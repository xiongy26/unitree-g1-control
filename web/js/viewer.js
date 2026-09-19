// three.js rendering: robot from model_desc.json, checker floor, trails,
// footfall markers and a top-view minimap.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

export class Viewer3D {
  constructor(container, desc) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
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
    this._buildRobot(desc);

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

  _buildRobot(desc) {
    this.bodyGroups = new Map();
    const loader = new STLLoader();
    const meshCache = new Map();
    const getGeom = (name) => new Promise((resolve) => {
      if (meshCache.has(name)) { meshCache.get(name).then(resolve); return; }
      const p = loader.loadAsync(`./model/assets/${name}`).then((geo) => {
        geo.computeVertexNormals();
        meshCache.set(name, Promise.resolve(geo));
        return geo;
      });
      meshCache.set(name, p);
      p.then(resolve);
    });

    for (const body of desc.bodies) {
      const grp = new THREE.Group();
      for (const geo of body.geoms) {
        let mesh = null;
        const rgba = geo.rgba;
        const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
        const mat = new THREE.MeshStandardMaterial({
          color, roughness: 0.45, metalness: 0.35,
        });
        if (geo.type === 7) { // mesh
          getGeom(geo.mesh).then((geometry) => {
            const m = new THREE.Mesh(geometry, mat);
            this._applyLocal(m, geo);
            m.castShadow = true; m.receiveShadow = true;
            grp.add(m);
          });
          continue;
        } else if (geo.type === 5) { // cylinder (size: radius, half-length)
          mesh = new THREE.Mesh(
            new THREE.CylinderGeometry(geo.size[0], geo.size[0], 2 * geo.size[1], 24), mat);
        } else if (geo.type === 2) { // sphere
          mesh = new THREE.Mesh(new THREE.SphereGeometry(geo.size[0], 20, 14), mat);
        } else if (geo.type === 6) { // box
          mesh = new THREE.Mesh(
            new THREE.BoxGeometry(2 * geo.size[0], 2 * geo.size[1], 2 * geo.size[2]), mat);
        } else {
          continue;
        }
        this._applyLocal(mesh, geo);
        mesh.castShadow = true; mesh.receiveShadow = true;
        grp.add(mesh);
      }
      this.bodyGroups.set(body.id, grp);
      this.zUpRoot.add(grp);
    }
  }

  _applyLocal(mesh, geo) {
    mesh.position.set(geo.pos[0], geo.pos[1], geo.pos[2]);
    mesh.quaternion.set(geo.quat[1], geo.quat[2], geo.quat[3], geo.quat[0]);
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
  syncFromData(data, desc) {
    for (const body of desc.bodies) {
      const grp = this.bodyGroups.get(body.id);
      if (!grp) continue;
      const o3 = body.id * 3, o4 = body.id * 4;
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
