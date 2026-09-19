// three.js rendering: robot built DIRECTLY from the compiled MuJoCo model's
// mesh buffers, checker floor, trails, footfall markers and a top-view
// minimap.
//
// Why not load the STL files? MuJoCo recomputes mesh frames at compile time
// (it recenters vertices and folds the mesh asset transform into
// geom_pos/geom_quat). The vertices in `model.mesh_vert` are therefore
// ALREADY in the frame that `geom_pos`/`geom_quat` expect, so applying those
// transforms directly is self-consistent — no manual compensation that can
// drift per-mesh (the "scattered arms" class of bugs). This mirrors the
// approach of zalo/mujoco_wasm and g1-kitchen-web.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Viewer3D {
  constructor(container, mujoco, model) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true,
      // the recorder composites this canvas into the video each frame; without
      // this flag the drawing buffer is cleared after compositing and the
      // recording would capture blank frames
      preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(innerWidth, innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b1226);
    this.scene.fog = new THREE.Fog(0x0b1226, 12, 40);

    const hemi = new THREE.HemisphereLight(0xbdd2ff, 0x1a2138, 0.9);
    this.scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-3, 6, 4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -6; sun.shadow.camera.right = 6;
    sun.shadow.camera.top = 6; sun.shadow.camera.bottom = -6;
    this.scene.add(sun);

    this.camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.05, 200);
    this.camera.position.set(-2.2, 1.4, 2.4);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.8, 0);
    this.followCam = true;

    this._buildFloor();
    this._buildRobot(mujoco, model);

    // trails
    this.comTrail = this._makeTrail(0xffffff, 4000, 0.011);
    this.planTrail = this._makeTrail(0x40ff50, 4000, 0.011);
    this.footfallGroup = new THREE.Group();
    this.zUpRoot.add(this.footfallGroup);

    addEventListener('resize', () => {
      this.camera.aspect = innerWidth / innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(innerWidth, innerHeight);
    });
  }

  _buildFloor() {
    // MuJoCo is z-up, three.js is y-up: everything lives under this root,
    // rotated -90° about X so MuJoCo +z becomes three.js +y.
    this.zUpRoot = new THREE.Group();
    this.zUpRoot.rotation.x = -Math.PI / 2;
    this.scene.add(this.zUpRoot);

    const c = document.createElement('canvas');
    c.width = c.height = 512;
    const g = c.getContext('2d');
    g.fillStyle = '#0d1526'; g.fillRect(0, 0, 512, 512);
    g.fillStyle = '#121d36';
    g.fillRect(0, 0, 256, 256); g.fillRect(256, 256, 256, 256);
    g.strokeStyle = 'rgba(140,165,220,0.55)'; g.lineWidth = 3;
    g.strokeRect(0, 0, 512, 512);
    g.strokeRect(256, 0, 256, 256); g.strokeRect(0, 256, 256, 256);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(14, 14);
    tex.anisotropy = 8;
    const mat = new THREE.MeshStandardMaterial({
      map: tex, roughness: 0.55, metalness: 0.25,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), mat);
    floor.receiveShadow = true;
    this.zUpRoot.add(floor);
  }

  _buildRobot(mujoco, model) {
    this.bodyGroups = new Map();
    const meshCache = new Map();
    const MESH = mujoco.mjtGeom.mjGEOM_MESH.value;
    const CYL = mujoco.mjtGeom.mjGEOM_CYLINDER.value;
    const SPH = mujoco.mjtGeom.mjGEOM_SPHERE.value;
    const BOX = mujoco.mjtGeom.mjGEOM_BOX.value;
    const PLANE = mujoco.mjtGeom.mjGEOM_PLANE.value;

    for (let g = 0; g < model.ngeom; g++) {
      if (model.geom_group[g] >= 3) continue;   // visual geoms only
      const type = model.geom_type[g];
      if (type === PLANE) continue;             // the floor is built below
      const size = [model.geom_size[g * 3], model.geom_size[g * 3 + 1],
                    model.geom_size[g * 3 + 2]];

      let geometry = null;
      if (type === MESH) {
        const mid = model.geom_dataid[g];
        if (!meshCache.has(mid)) {
          geometry = new THREE.BufferGeometry();
          const va = model.mesh_vertadr[mid], vn = model.mesh_vertnum[mid];
          const fa = model.mesh_faceadr[mid], fn = model.mesh_facenum[mid];
          // .slice() on the WASM heap views yields plain typed arrays (a
          // copy), which three.js needs — passing heap views directly breaks
          // its buffer bookkeeping (createBuffer expects .byteLength).
          const verts = model.mesh_vert.slice(va * 3, (va + vn) * 3);
          const faces = model.mesh_face.slice(fa * 3, (fa + fn) * 3);
          geometry.setAttribute('position',
            new THREE.BufferAttribute(new Float32Array(verts), 3));
          geometry.setIndex(Array.from(faces));
          geometry.computeVertexNormals();
          meshCache.set(mid, geometry);
        } else {
          geometry = meshCache.get(mid);
        }
      } else if (type === CYL) {
        geometry = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 24);
      } else if (type === SPH) {
        geometry = new THREE.SphereGeometry(size[0], 20, 14);
      } else if (type === BOX) {
        geometry = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
      } else {
        continue;
      }

      let rgba = [model.geom_rgba[g * 4], model.geom_rgba[g * 4 + 1],
                  model.geom_rgba[g * 4 + 2], model.geom_rgba[g * 4 + 3]];
      const matId = model.geom_matid[g];
      if (matId !== -1) {
        rgba = [model.mat_rgba[matId * 4], model.mat_rgba[matId * 4 + 1],
                model.mat_rgba[matId * 4 + 2], model.mat_rgba[matId * 4 + 3]];
      }
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        transparent: rgba[3] < 1, opacity: rgba[3],
        roughness: 0.45, metalness: 0.35,
      });
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.castShadow = mesh.receiveShadow = true;
      // geom local pose — the compiled vertices/frame match these directly
      mesh.position.set(model.geom_pos[g * 3], model.geom_pos[g * 3 + 1],
                        model.geom_pos[g * 3 + 2]);
      mesh.quaternion.set(model.geom_quat[g * 4 + 1], model.geom_quat[g * 4 + 2],
                          model.geom_quat[g * 4 + 3], model.geom_quat[g * 4]);

      const b = model.geom_bodyid[g];
      let grp = this.bodyGroups.get(b);
      if (!grp) {
        grp = new THREE.Group();
        this.bodyGroups.set(b, grp);
        this.zUpRoot.add(grp);
      }
      grp.add(mesh);
    }
  }

  _makeTrail(color, maxPts, width) {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxPts * 3), 3));
    geom.setDrawRange(0, 0);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
    const line = new THREE.Line(geom, mat);
    line.frustumCulled = false;
    this.zUpRoot.add(line);
    return { line, geom, n: 0, maxPts, last: null };
  }

  _pushTrail(trail, p) {
    if (trail.n >= trail.maxPts) return;
    if (trail.last && Math.hypot(p[0] - trail.last[0], p[1] - trail.last[1]) < 0.015) return;
    const arr = trail.geom.attributes.position.array;
    arr[trail.n * 3] = p[0]; arr[trail.n * 3 + 1] = p[1]; arr[trail.n * 3 + 2] = 0.02;
    trail.n++;
    trail.geom.setDrawRange(0, trail.n);
    trail.geom.attributes.position.needsUpdate = true;
    trail.last = p;
  }

  resetTrails(plan, mujocoTrailPts) {
    for (const tr of [this.comTrail, this.planTrail]) { tr.n = 0; tr.last = null; tr.geom.setDrawRange(0, 0); }
    // planned ZMP path (green)
    const dec = 2;
    for (let i = 0; i < mujocoTrailPts.length - dec; i += dec) {
      this._pushTrail(this.planTrail, [mujocoTrailPts[i][0], mujocoTrailPts[i][1]]);
    }
    this.footfallGroup.clear();
    for (const [x, y] of plan.footfallPoints()) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.016, 12, 10),
        new THREE.MeshBasicMaterial({ color: 0xffd24a }));
      s.position.set(x, y, 0.015);
      this.footfallGroup.add(s);
    }
  }
  syncFromData(data) {
    for (const [b, grp] of this.bodyGroups) {
      const o3 = b * 3, o4 = b * 4;
      grp.position.set(data.xpos[o3], data.xpos[o3 + 1], data.xpos[o3 + 2]);
      grp.quaternion.set(data.xquat[o4 + 1], data.xquat[o4 + 2], data.xquat[o4 + 3],
                         data.xquat[o4]);
    }
  }

  updateCamera(com) {
    if (this.followCam) {
      // MuJoCo (x, y, z) renders at three.js (x, z, -y)
      this.controls.target.set(com[0] + 0.45, com[2], -com[1]);
    }
    this.controls.update();
  }

  render() { this.renderer.render(this.scene, this.camera); }
}
