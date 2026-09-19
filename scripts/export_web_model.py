"""Prepare everything the web (MuJoCo WASM) app needs:

1. web/model/g1_wasm.xml  -- flattened, self-contained scene XML
2. web/model/assets/*.STL -- mesh files for the virtual FS
3. web/gains.json         -- ZMP preview-control gains (Kx, kr) computed with scipy
4. web/model_desc.json    -- body/geom description for the three.js renderer
"""
import json
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
MODEL_DIR = WEB / "model"
ASSET_DIR = MODEL_DIR / "assets"

MENAGERIE_G1 = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1"


class KajitaPreviewControl:
    """ZMP preview control of the linear inverted pendulum (Kajita et al. 2003).

    Cart-table model of the CoM:  p_ddddot = u (jerk input),
    p_zmp = p_com - (zc/g) * p_com_dd.  Discretised at `dt` and solved with
    the standard preview-control LQR: the future ZMP reference is part of an
    augmented state and the DARE yields the state feedback and preview gains.
    """

    def __init__(self, com_height: float, dt: float = 0.005,
                 preview_time: float = 1.5, q_zmp: float = 1.0,
                 r_jerk: float = 1e-6, g: float = 9.81):
        self.dt = dt
        self.zc = float(com_height)
        self.g = g

        A = np.array([[1.0, dt, 0.5 * dt * dt],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, 1.0]])
        B = np.array([[dt ** 3 / 6.0], [0.5 * dt * dt], [dt]])
        C = np.array([[1.0, 0.0, -self.zc / g]])

        n_prev = max(1, int(round(preview_time / dt)))
        N = 3 + n_prev
        Aa = np.zeros((N, N))
        Aa[:3, :3] = A
        Aa[3:, 3:] = np.eye(n_prev, k=1)     # shift future references
        Ba = np.zeros((N, 1))
        Ba[:3, 0] = B[:, 0]
        Qa = np.zeros((N, N))
        Qa[:3, :3] = C.T @ C * q_zmp
        Qa[:3, 3] = -C.T[:, 0] * q_zmp
        Qa[3, :3] = -C[0, :] * q_zmp
        Qa[3, 3] = q_zmp
        Ra = np.array([[r_jerk]])

        Pa = solve_discrete_are(Aa, Ba, Qa, Ra)
        Kbar = np.linalg.solve(Ra + Ba.T @ Pa @ Ba, Ba.T @ Pa @ Aa)
        self.Kx = Kbar[0, :3]                # u = -Kx x - kr . r(k:k+n)
        self.kr = Kbar[0, 3:]
        self.n_prev = n_prev


def build_model_xml() -> None:
    """Flatten scene + robot into one XML with meshdir="assets"."""
    robot_xml = (ROOT / "models" / "g1_walk.xml").read_text(encoding="utf-8")

    # extract top-level sections (2-space indented children of <mujoco>)
    sections = {}
    for m in re.finditer(r"^  <(\w+)[^>]*>", robot_xml, re.M):
        tag = m.group(1)
        close = f"\n  </{tag}>"
        end = robot_xml.find(close, m.start())
        if end < 0:
            continue
        body = robot_xml[m.end():end]
        sections.setdefault(tag, body)

    need = ["default", "asset", "worldbody", "actuator", "sensor", "keyframe"]
    missing = [t for t in need if t not in sections]
    assert not missing, f"sections missing: {missing}"

    scene = f"""<mujoco model="g1_zmp_walking_web">
  <compiler angle="radian" meshdir="assets"/>
  <option integrator="implicitfast" timestep="0.002"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.25 0.25 0.3" specular="0.8 0.8 0.8"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>

  <default>{sections["default"]}
  </default>

  <asset>{sections["asset"]}
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" friction="1.0"/>
{sections["worldbody"]}
  </worldbody>

  <actuator>{sections["actuator"]}
  </actuator>

  <sensor>{sections["sensor"]}
  </sensor>

  <keyframe>{sections["keyframe"]}
  </keyframe>
</mujoco>
"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "g1_wasm.xml").write_text(scene, encoding="utf-8")

    # meshes
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    shutil.copytree(MENAGERIE_G1 / "assets", ASSET_DIR)

    # sanity: load with desktop mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "g1_wasm.xml"))
    print(f"g1_wasm.xml OK: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")


def export_model_desc() -> None:
    """Body/geom description for the three.js side."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "g1_wasm.xml"))

    # mesh name -> asset filename (names may differ from files)
    mesh_file = {}
    asset_txt = sections_text(MODEL_DIR / "g1_wasm.xml", "asset")
    for m in re.finditer(r'<mesh\b[^>]*/>', asset_txt):
        tag = m.group(0)
        name_m = re.search(r'name="([^"]+)"', tag)
        file_m = re.search(r'file="([^"]+)"', tag)
        if not file_m:
            continue
        fname = file_m.group(1)
        mname = name_m.group(1) if name_m else fname.rsplit(".", 1)[0]
        mesh_file[mname] = fname

    meshes = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
              for i in range(model.nmesh)]

    bodies = []
    for b in range(1, model.nbody):          # skip world
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        geoms = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b:
                continue
            gtype = int(model.geom_type[g])
            mat_id = model.geom_matid[g]
            rgba = model.geom_rgba[g].tolist()
            if mat_id >= 0:
                rgba = model.mat_rgba[mat_id].tolist()
            entry = dict(
                id=g, type=gtype, pos=model.geom_pos[g].tolist(),
                quat=model.geom_quat[g].tolist(), rgba=rgba,
                group=int(model.geom_group[g]),
                size=model.geom_size[g].tolist(),
            )
            if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                entry["mesh"] = mesh_file[meshes[model.geom_dataid[g]]]
                # MuJoCo recenters mesh vertices at compile time: it folds the
                # mesh asset transform (mesh_pos, mesh_quat) into geom_pos/
                # geom_quat and stores *centered* vertices. three.js loads the
                # RAW STL, so the child transform must undo the asset
                # transform:  A = X_geom ∘ M⁻¹.
                mid = model.geom_dataid[g]
                mpos = model.mesh_pos[mid]
                mq = model.mesh_quat[mid]          # wxyz
                gq = model.geom_quat[g]            # wxyz
                # mq inverse
                mqi = [mq[0], -mq[1], -mq[2], -mq[3]]
                # quat = geom_quat ⊗ mesh_quat⁻¹
                w1, x1, y1, z1 = gq
                w2, x2, y2, z2 = mqi
                entry["quat"] = [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ]
                # pos = geom_pos - R(geom_quat) @ mesh_pos
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, gq)
                R = R.reshape(3, 3)
                entry["pos"] = (model.geom_pos[g] - R @ mpos).tolist()
            if int(model.geom_group[g]) in (0, 1) or gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue                      # skip non-visual helper geoms
            geoms.append(entry)
        bodies.append(dict(id=b, name=name, geoms=geoms))

    desc = dict(
        nq=int(model.nq), nv=int(model.nv), nu=int(model.nu),
        key_stand_qpos=model.key_qpos[0].tolist(),
        bodies=bodies,
    )
    (WEB / "model_desc.json").write_text(json.dumps(desc))
    n_meshes = sum(1 for b in bodies for g in b["geoms"] if "mesh" in g)
    print(f"model_desc.json: {len(bodies)} bodies, {n_meshes} visual mesh geoms")


def sections_text(path, tag):
    """Return the inner text of a top-level XML section."""
    xml = Path(path).read_text(encoding="utf-8")
    m = re.search(rf"^  <{tag}[^>]*>", xml, re.M)
    end = xml.find(f"\n  </{tag}>", m.start())
    return xml[m.end():end]


def export_gains() -> None:
    """ZMP preview-control gains for the online controller (fixed zc)."""
    zc = 0.685
    pc = KajitaPreviewControl(com_height=zc, dt=0.005, preview_time=1.5,
                              q_zmp=1.0, r_jerk=1e-5)
    gains = dict(
        zc=zc, dt=pc.dt, n_prev=int(pc.n_prev),
        Kx=pc.Kx.tolist(), kr=pc.kr.tolist(),
        g=9.81,
    )
    (WEB / "gains.json").write_text(json.dumps(gains))
    print(f"gains.json: Kx={np.round(pc.Kx, 3).tolist()} "
          f"kr[:3]={np.round(pc.kr[:3], 4).tolist()} (n_prev={pc.n_prev})")


if __name__ == "__main__":
    build_model_xml()
    export_model_desc()
    export_gains()
