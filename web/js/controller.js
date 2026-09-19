// Gait planner + online ZMP preview control + weighted whole-body IK,
// ported from the Python implementation (g1_zmp_walking/).
//
// References:
//  - Kajita et al. 2003, "ZMP-based walking with preview control"
//  - github.com/chauby/ZMP_preview_control, github.com/zanppa/WPG
//  - Weighted IK: github.com/kevinzakka/mink (same weighted-task formulation)

// ---------------------------------------------------------------- utils
// minimum-jerk quintic time scaling, s³(10-15s+6s²), 0→1 with zero vel/acc
// at both ends
export const minjerk = (s) => {
  s = Math.min(Math.max(s, 0), 1);
  return s * s * s * (10 - 15 * s + 6 * s * s);
};
export const smoothstep = (s) => {
  s = Math.min(Math.max(s, 0), 1);
  return s * s * (3 - 2 * s);
};

export function quatMul(a, b) { // wxyz
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}
export function quatConj(q) { return [q[0], -q[1], -q[2], -q[3]]; }
// world-frame rotation-vector log of R(q) (q = R_tgt * R_cur^-1)
export function quatLog3(q) {
  let [w, x, y, z] = q;
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const s = Math.sqrt(x * x + y * y + z * z);
  if (s < 1e-9) return [2 * x, 2 * y, 2 * z];
  const ang = 2 * Math.atan2(s, w);
  return [x / s * ang, y / s * ang, z / s * ang];
}

function solveLinear(H, g, n) { // Gaussian elimination w/ partial pivot
  const A = new Float64Array(H); // copy, row-major n×n
  const b = new Float64Array(g);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[piv * n + c])) piv = r;
    if (Math.abs(A[piv * n + c]) < 1e-12) continue;
    if (piv !== c) {
      for (let k = 0; k < n; k++) { const t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; }
      const t = b[c]; b[c] = b[piv]; b[piv] = t;
    }
    const d = A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / d;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * x[k];
    x[r] = Math.abs(A[r * n + r]) < 1e-12 ? 0 : s / A[r * n + r];
  }
  return x;
}

// ---------------------------------------------------------------- gait
export class GaitPlan {
  constructor(feet0, p) {
    // p: {nSteps, stepLength, tInit, tSS, tDS, tFinal, swingHeight,
    //     footCenterDx, stanceInset, firstSwing}
    const ySym = Math.max(0.05,
      0.5 * (Math.abs(feet0.left[1]) + Math.abs(feet0.right[1])) - p.stanceInset);
    this.footY = { left: ySym, right: -ySym };
    this.p = p;
    this.curX = { left: feet0.left[0], right: feet0.right[0] };
    this.steps = [];
    let swing = p.firstSwing;
    let t = p.tInit;
    for (let i = 0; i < p.nSteps; i++) {
      const other = swing === 'left' ? 'right' : 'left';
      const xTo = this.curX[other] + p.stepLength;
      this.steps.push({ swing, xFrom: this.curX[swing], xTo, tLift: t, tLand: t + p.tSS });
      this.curX[swing] = xTo;
      t += p.tSS + p.tDS;
      swing = other;
    }
    // closing step to bring feet side by side
    if (Math.abs(this.curX.left - this.curX.right) > 0.05) {
      const trail = this.curX.left < this.curX.right ? 'left' : 'right';
      const xTo = Math.max(this.curX.left, this.curX.right);
      this.steps.push({ swing: trail, xFrom: this.curX[trail], xTo, tLift: t, tLand: t + p.tSS });
      this.curX[trail] = xTo;
      t += p.tSS + p.tDS;
    }
    this.phases = [{ t0: 0, t1: p.tInit, kind: 'init' }];
    for (const st of this.steps) {
      const support = st.swing === 'left' ? 'right' : 'left';
      this.phases.push({ t0: st.tLift, t1: st.tLand, kind: 'ss', swing: st.swing, support });
      this.phases.push({
        t0: st.tLand, t1: st.tLand + p.tDS, kind: 'ds',
        from: this.footCenter(support, st.xFrom), to: this.footCenter(st.swing, st.xTo),
      });
      t = st.tLand + p.tDS;
    }
    this.phases.push({ t0: t, t1: t + p.tFinal, kind: 'final' });
    this.tEnd = t + p.tFinal;
  }

  footCenter(foot, x) { return [x + this.p.footCenterDx, this.footY[foot]]; }

  phaseAt(t) {
    for (const ph of this.phases) if (t < ph.t1) return ph;
    return this.phases[this.phases.length - 1];
  }
  phaseIndexAt(t) {
    for (let i = 0; i < this.phases.length; i++) if (t < this.phases[i].t1) return i;
    return this.phases.length - 1;
  }

  footX(foot, t) {
    // initial x of the foot = its first swing "from" (all feet swing);
    const first = this.steps.find((s) => s.swing === foot);
    let x = first ? first.xFrom : 0;
    for (const s of this.steps) {
      if (s.swing !== foot) continue;
      if (t >= s.tLand) x = s.xTo;
      else break;
    }
    return x;
  }

  // ankle-site pose [x,y,z] of `foot` at time t (z relative to standing)
  footPose(foot, t) {
    const ph = this.phaseAt(t);
    if (ph.kind === 'ss' && ph.swing === foot) {
      const st = this.steps.find((s) => s.tLift === ph.t0);
      const s = minjerk((t - ph.t0) / (ph.t1 - ph.t0));
      return [st.xFrom + (st.xTo - st.xFrom) * s, this.footY[foot],
              this.p.swingHeight * Math.sin(Math.PI * s)];
    }
    return [this.footX(foot, t), this.footY[foot], 0];
  }

  zmpRef(t) {
    const ph = this.phaseAt(t);
    if (ph.kind === 'ss') return this.footCenter(ph.support, this.footX(ph.support, t));
    if (ph.kind === 'ds') {
      const s = smoothstep((t - ph.t0) / (ph.t1 - ph.t0));
      return [ph.from[0] + (ph.to[0] - ph.from[0]) * s,
              ph.from[1] + (ph.to[1] - ph.from[1]) * s];
    }
    const x = 0.5 * (this.footX('left', t) + this.footX('right', t));
    return [x + this.p.footCenterDx, 0];
  }

  footfallPoints() {
    const pts = [];
    for (const foot of ['left', 'right']) {
      const first = this.steps.find((s) => s.swing === foot);
      pts.push([first ? first.xFrom : 0, this.footY[foot]]);
    }
    for (const s of this.steps) pts.push([s.xTo, this.footY[s.swing]]);
    return pts;
  }
}

// ------------------------------------------------- online preview control
export class OnlinePreview {
  // gains: {Kx[3], kr[N], dt, n_prev, zc}
  constructor(gains, plan, tEnd, resimWindow = 0.6) {
    this.Kx = Float64Array.from(gains.Kx);
    this.kr = Float64Array.from(gains.kr);
    this.dt = gains.dt;
    this.nPrev = gains.n_prev;
    this.zc = gains.zc;
    this.g = 9.81;
    // reference sampled at preview dt, padded at the end
    const nRef = Math.round((tEnd + 2.0) / this.dt) + 1;
    this.refX = new Float64Array(nRef + this.nPrev + 2);
    this.refY = new Float64Array(nRef + this.nPrev + 2);
    for (let k = 0; k < nRef; k++) {
      const r = plan.zmpRef(k * this.dt);
      this.refX[k] = r[0]; this.refY[k] = r[1];
    }
    for (let k = nRef; k < this.refX.length; k++) {
      this.refX[k] = this.refX[nRef - 1]; this.refY[k] = this.refY[nRef - 1];
    }
    // offline closed-loop plan (for the analysis panel & target tube)
    const nPlan = Math.round(tEnd / this.dt) + 1;
    this.planT = new Float64Array(nPlan);
    this.planX = new Float64Array(nPlan);
    this.planY = new Float64Array(nPlan);
    let sx = this.refX[0], svx = 0, sax = 0;
    let sy = this.refY[0], svy = 0, say = 0;
    const A = [1, this.dt, 0.5 * this.dt * this.dt];
    for (let k = 0; k < nPlan; k++) {
      this.planT[k] = k * this.dt;
      this.planX[k] = sx; this.planY[k] = sy;
      const kx = Math.min(k, this.refX.length - this.nPrev - 1);
      const ux = -(this.Kx[0] * sx + this.Kx[1] * svx + this.Kx[2] * sax)
        - this.dotRef(this.kr, this.refX, kx);
      const uy = -(this.Kx[0] * sy + this.Kx[1] * svy + this.Kx[2] * say)
        - this.dotRef(this.kr, this.refY, kx);
      sax += ux * this.dt; svx += sax * this.dt; sx += svx * this.dt;
      say += uy * this.dt; svy += say * this.dt; sy += svy * this.dt;
    }
    // receding-horizon re-simulation buffers (stabilized closed-loop from
    // the measured CoM state; see resim())
    this.nW = Math.max(2, Math.round(resimWindow / this.dt));
    this.rsX = new Float64Array(this.nW + 2);
    this.rsY = new Float64Array(this.nW + 2);
  }
  dotRef(kr, ref, k) {
    let s = 0;
    for (let i = 0; i < this.nPrev; i++) s += kr[i] * ref[k + i];
    return s;
  }
  planAt(t) { // interpolated offline plan
    const k = Math.min(Math.max(t / this.dt, 0), this.planT.length - 1.001);
    const i = Math.floor(k), f = k - i;
    const j = Math.min(i + 1, this.planT.length - 1);
    return [this.planX[i] + (this.planX[j] - this.planX[i]) * f,
            this.planY[i] + (this.planY[j] - this.planY[i]) * f];
  }
  // Re-simulate the *stabilized* closed-loop cart-table forward from the
  // measured CoM state (pos, vel, acc) at walk time t, following the ZMP
  // reference with the preview law. The LQR term makes this trajectory
  // converge back onto the offline plan, so it is dynamically consistent
  // AND corrective: sampling it ahead of "now" yields a CoM command with
  // both feedforward (reference preview) and feedback (measured state).
  resim(t, pos, vel, acc) {
    const dt = this.dt, nW = this.nW;
    const kMax = this.refX.length - this.nPrev - 1;
    const k0 = Math.min(Math.max(Math.round(t / dt), 0), kMax);
    let sx = pos[0], svx = vel[0], sax = acc[0];
    let sy = pos[1], svy = vel[1], say = acc[1];
    for (let i = 0; i <= nW; i++) {
      this.rsX[i] = sx; this.rsY[i] = sy;
      const k = Math.min(k0 + i, kMax);
      const ux = -(this.Kx[0] * sx + this.Kx[1] * svx + this.Kx[2] * sax)
        - this.dotRef(this.kr, this.refX, k);
      const uy = -(this.Kx[0] * sy + this.Kx[1] * svy + this.Kx[2] * say)
        - this.dotRef(this.kr, this.refY, k);
      sax += ux * dt; svx += sax * dt; sx += svx * dt;
      say += uy * dt; svy += say * dt; sy += svy * dt;
    }
  }
  // sample the re-simulated trajectory at t + tau (tau <= window; past the
  // window the re-sim has converged onto the offline plan, so blend there)
  resimAt(tau) {
    const i = Math.min(Math.max(tau / this.dt, 0), this.nW);
    const i0 = Math.floor(i), f = i - i0;
    const j = Math.min(i0 + 1, this.nW);
    return [this.rsX[i0] + (this.rsX[j] - this.rsX[i0]) * f,
            this.rsY[i0] + (this.rsY[j] - this.rsY[i0]) * f];
  }
  // state = [x, vx, ax, y, vy, ay]; refIdx k ↔ time t
  jerk(state, k) {
    const ax = state[0], vx = state[1], aax = state[2];
    const ay = state[3], vy = state[4], aay = state[5];
    const kx = Math.min(k, this.refX.length - this.nPrev - 1);
    const ux = -(this.Kx[0] * ax + this.Kx[1] * vx + this.Kx[2] * aax)
      - this.dotRef(this.kr, this.refX, kx);
    const uy = -(this.Kx[0] * ay + this.Kx[1] * vy + this.Kx[2] * aay)
      - this.dotRef(this.kr, this.refY, kx);
    return [ux, uy];
  }
}

// ---------------------------------------------------------------- WBC
export class WalkingController {
  constructor(mujoco, model, data, gains, params) {
    this.mj = mujoco; this.model = model; this.data = data;
    this.gains = gains; this.p = params;
    const nv = model.nv, nq = model.nq;
    this.nv = nv; this.nq = nq;

    this.pelvisId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis');
    this.siteId = {
      left: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'left_foot'),
      right: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'right_foot'),
    };
    // actuator -> qpos address
    this.ctrlQadr = new Int32Array(model.nu);
    for (let i = 0; i < model.nu; i++) {
      this.ctrlQadr[i] = model.jnt_qposadr[model.actuator_trnid[i * 2]];
    }
    // hinge joints (dofadr, qposadr) for the posture task
    this.hinges = [];
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] === mujoco.mjtJoint.mjJNT_HINGE.value) {
        this.hinges.push([model.jnt_dofadr[j], model.jnt_qposadr[j]]);
      }
    }
    // knee joints (singularity protection)
    this.kneeQads = ['left_knee_joint', 'right_knee_joint'].map((n) =>
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n)]);

    // bent-knee posture (feet flat: ankle = -(hip+knee))
    this.qCrouch = Float64Array.from(model.key_qpos.slice(0, nq));
    const jadr = (n) => model.jnt_qposadr[
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n)];
    for (const side of ['left', 'right']) {
      this.qCrouch[jadr(`${side}_hip_pitch_joint`)] = params.crouchHip;
      this.qCrouch[jadr(`${side}_knee_joint`)] = params.crouchKnee;
      this.qCrouch[jadr(`${side}_ankle_pitch_joint`)] = params.crouchAnkle;
    }
    this.ctrlCrouch = new Float64Array(model.nu);
    for (let i = 0; i < model.nu; i++) this.ctrlCrouch[i] = this.qCrouch[this.ctrlQadr[i]];

    // separate data for the IK kinematics
    this.dIk = new mujoco.MjData(model);
    this.qRef = Float64Array.from(this.qCrouch);

    // jacobian buffers (3*nv each), allocated once
    this.jacBufs = [];
    for (let i = 0; i < 4; i++) {
      this.jacBufs.push(mujoco.DoubleBuffer.FromArray(new Array(3 * nv).fill(0)));
    }
    this.forceBuf = mujoco.DoubleBuffer.FromArray(new Array(6).fill(0));
    this.H = new Float64Array(nv * nv);
    this.g = new Float64Array(nv);
    this.ctrl = new Float64Array(model.nu);

    // weights (relative task costs)
    this.W = params.weights;
    this.damping = params.ikDamping;

    // gait / state
    this.t = 0; this.stepCount = 0; this.kIter = 0;
    this.phaseIdx = -1; this.hold = {}; this.swingStart = null;
    this.estVel = [0, 0]; this.estAcc = [0, 0];
    this.prevPos = [0, 0]; this.prevVel = [0, 0];
    this.comCmd = [0, 0]; this.zmpMeas = [NaN, NaN];
    this.fell = false; this.done = false;
    this.status = 'init';
  }

  com() { // measured whole-body CoM
    return [this.data.subtree_com[0], this.data.subtree_com[1], this.data.subtree_com[2]];
  }
  sitePos(foot) {
    const o = this.siteId[foot] * 3;
    return [this.data.site_xpos[o], this.data.site_xpos[o + 1], this.data.site_xpos[o + 2]];
  }

  // ------------------------------------------------------------- settle
  settle() {
    const dt = this.model.opt.timestep;
    // Start directly in the crouch pose with the feet resting on the floor:
    // dropping from the straight-legged keyframe while commanding a deep
    // crouch makes the robot free-fall and slam into the ground.
    this.data.qpos.set(this.qCrouch);
    this.mj.mj_kinematics(this.model, this.data);
    this.mj.mj_comPos(this.model, this.data);
    const fz = 0.5 * (this.data.site_xpos[this.siteId.left * 3 + 2] +
                      this.data.site_xpos[this.siteId.right * 3 + 2]);
    this.data.qpos[2] += 0.035 - fz;   // sole spheres touch ground at 0.035
    this.mj.mj_forward(this.model, this.data);
    // brief hold at the crouch posture
    for (let k = 0; k < Math.round(this.p.dropTime / dt); k++) {
      this.data.ctrl.set(this.ctrlCrouch);
      this.mj.mj_step(this.model, this.data);
    }
    // stage B: whole-body control. Balance the CoM over the foot CENTERS:
    // full pitch authority in both directions at walk start, and the ZMP
    // reference during the initial double support equals exactly this point,
    // so walking starts without any transient.
    const feetMeas = { left: this.sitePos('left'), right: this.sitePos('right') };
    this.feet0 = { left: feetMeas.left.slice(0, 2), right: feetMeas.right.slice(0, 2) };
    this.footZ = 0.5 * (feetMeas.left[2] + feetMeas.right[2]);
    this.footTargets = { left: feetMeas.left, right: feetMeas.right };
    this.com0 = this.com();
    this.zc = this.com0[2];
    const balX = 0.5 * (this.feet0.left[0] + this.feet0.right[0]) + this.p.footCenterDx;
    const balY = 0.5 * (this.feet0.left[1] + this.feet0.right[1]);
    this.balPos = [balX, balY];
    this.comTarget = [balX, balY, this.zc];
    this.plan = new GaitPlan(this.feet0, this.p);
    this.preview = new OnlinePreview(this.gains, this.plan, this.plan.tEnd);
    this.prevPos = null;                 // first walk update initializes it
    this.comCmd = [balX, balY];
    this.walkWarmup = 0;
    this.estVel = [0, 0]; this.estAcc = [0, 0]; this.prevVel = [0, 0];
    for (let k = 0; k < Math.round(this.p.settleTime / dt); k++) {
      this.settling = true;
      if (k % this.p.ikRate === 0) this.updateIK(dt * this.p.ikRate);
      this.data.ctrl.set(this.ctrl);
      this.mj.mj_step(this.model, this.data);
    }
    this.settling = false;
    if (this.isFallen()) {
      this.status = 'FELL';
      this.fell = true;
      return;
    }
    this.t = 0; this.phaseIdx = -1; this.tPlan = 0;
    this.status = 'walking';
  }

  // ---------------------------------------------------------- targets
  footTargetsUpdate(t) {
    const idx = this.plan.phaseIndexAt(t);
    const ph = this.plan.phases[idx];
    if (idx !== this.phaseIdx) {
      this.phaseIdx = idx;
      this.swingStart = null;
      this.hold = { left: this.virtFoot('left'), right: this.virtFoot('right') };
      // the foot that just finished swinging holds its PLANNED footfall,
      // not wherever the virtual config happens to be (after an
      // early-touchdown clock jump it can still be mid-air there)
      const prev = this.plan.phases[idx - 1];
      if (prev && prev.kind === 'ss') {
        const st = this.plan.steps.find((sp) => sp.tLift === prev.t0);
        this.hold[st.swing] = [st.xTo, this.plan.footY[st.swing], this.footZ];
      }
    }
    if (ph.kind === 'ss') {
      const st = this.plan.steps.find((s) => s.tLift === ph.t0);
      if (!this.swingStart) this.swingStart = this.virtFoot(ph.swing);
      // swing completes within `swingEndFrac` of the phase; the foot is then
      // held at its landed pose for the remainder, giving the position
      // actuators time to seat the leg firmly before weight transfer
      const sSw = Math.min(1, (t - ph.t0) / ((ph.t1 - ph.t0) * (this.p.swingEndFrac ?? 0.85)));
      const s = minjerk(sSw);
      const goal = [st.xTo, this.plan.footY[st.swing], this.footZ];
      const pos = [0, 1, 2].map((i) =>
        this.swingStart[i] + (goal[i] - this.swingStart[i]) * s);
      pos[2] = this.swingStart[2] + this.p.swingHeight * Math.sin(Math.PI * s);
      this.footTargets[ph.swing] = pos;
      this.footTargets[ph.support] = this.hold[ph.support];
      // Early-touchdown re-timing: with position actuators the swing foot
      // physically lands before the phase boundary, and if the ZMP reference
      // keeps saying "CoP on the old foot" the actual CoP jumps to the new
      // foot anyway -- from then on plan and reality disagree and the walk
      // dies. Once the swing is well underway (past 60% of its path) and the
      // foot is physically down, fast-forward the plan clock to the
      // double-support phase so the reference transfer starts when the
      // transfer really happens. (The planned CoM trajectory is continuous
      // across the boundary, so the CoM command does not jump.)
      if (sSw > 0.6 && this.tPlan < ph.t1 - 0.02) {
        const swz = this.sitePos(ph.swing)[2];
        if (swz < this.footZ + 0.005) {
          this.hold[ph.swing] = [st.xTo, this.plan.footY[st.swing], this.footZ];
          this.footTargets[ph.swing] = this.hold[ph.swing];
          this.tPlan = ph.t1;
        }
      }
    } else {
      this.footTargets.left = this.hold.left;
      this.footTargets.right = this.hold.right;
    }
  }

  comTargetUpdate(t, dti) {
    // Receding-horizon online ZMP preview control. The CoM state is estimated
    // from the measured whole-body CoM, the *stabilized* closed-loop
    // cart-table model is re-simulated forward from that state (~0.6 s) and
    // the CoM command is a point `comLead` seconds ahead on the re-simulated
    // trajectory. Because the re-simulation starts from the measured state
    // and its LQR converges onto the ZMP reference, the command carries both
    // feedforward (reference preview) and feedback (measured state): any
    // tracking error immediately bends the future trajectory back toward the
    // reference instead of hugging the robot's own fall. `t` is the paced
    // plan-clock time (see updatePlanClock), not the wall time.
    const c = this.com();
    const pos = [c[0], c[1]];

    if (!this.prevPos) {                 // first update after settle
      this.prevPos = pos;
      this.comCmd = pos.slice();
      this.comTarget = [pos[0], pos[1], this.zc];
      return this.comTarget;
    }
    // warm-up: hold still for the first updates so estimator transients
    // right after the settle phase cannot kick the robot
    this.walkWarmup = (this.walkWarmup || 0) + 1;
    if (this.walkWarmup <= 10) {
      this.prevPos = pos;
      this.estVel = [0, 0]; this.estAcc = [0, 0]; this.prevVel = [0, 0];
      this.comCmd = [pos[0], pos[1]];
      this.comTarget = [pos[0], pos[1], this.zc];
      return this.comTarget;
    }
    const velRaw = [(pos[0] - this.prevPos[0]) / dti, (pos[1] - this.prevPos[1]) / dti];
    this.prevPos = pos;
    const sym = (v, lim) => Math.max(-lim, Math.min(lim, v));
    this.estVel = [sym(0.5 * this.estVel[0] + 0.5 * velRaw[0], 1.5),
                   sym(0.5 * this.estVel[1] + 0.5 * velRaw[1], 1.5)];
    const accRaw = [(this.estVel[0] - this.prevVel[0]) / dti,
                    (this.estVel[1] - this.prevVel[1]) / dti];
    this.prevVel = this.estVel.slice();
    this.estAcc = [sym(0.3 * this.estAcc[0] + 0.7 * accRaw[0], 8),
                   sym(0.3 * this.estAcc[1] + 0.7 * accRaw[1], 8)];

    const pc = this.preview;
    pc.resim(t, pos, this.estVel, this.estAcc);
    // The lead compensates the command→actual transport lag (IK virtual
    // configuration + position actuators); the lateral direction propagates
    // through hip roll and lags a bit more. The re-simulated trajectory
    // starts at the measured state, so at t+lead it is still displaced from
    // the plan by exp(-lead/τ_LQR) ≈ 0.6 of the current state error. That
    // deviation is REFLECTED about the reference (`plan - resim`): the robot
    // trailing the plan yields a command ahead of the plan, an overshooting
    // robot brakes — a proper negative state feedback whose position AND
    // velocity AND acceleration components are shaped by the preview LQR
    // closed-loop modes.
    const lead = [this.p.comLeadX ?? 0.10, this.p.comLeadY ?? 0.14];
    const beta = this.p.comFeedback ?? 2.0;
    const plL = [pc.planAt(t + lead[0]), pc.planAt(t + lead[1])];
    const rsX = pc.resimAt(lead[0]);
    const rsY = pc.resimAt(lead[1]);
    const tx = plL[0][0] + beta * (plL[0][0] - rsX[0]);
    const ty = plL[1][1] + beta * (plL[1][1] - rsY[1]);
    // safety tube around the offline plan bounds the correction authority
    const pl = pc.planAt(t);
    const tube = this.p.comTube ?? 0.15;
    const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    let cx = cl(tx, pl[0] - tube, pl[0] + tube);
    let cy = cl(ty, pl[1] - tube, pl[1] + tube);
    this.rawCmd = [tx, ty];
    // Measured-CoP feedback: the CoP responds quasi-instantaneously to the
    // commanded pelvis position (much faster than the CoM), so pushing the
    // CoM command toward the ZMP reference error drives the weight transfer
    // (CoP stuck on the old foot → shift the pelvis onto the new foot) and
    // keeps the CoP away from the foot edges (ankle strategy).
    const ref = this.plan.zmpRef(t);
    if (isFinite(this.zmpMeas[0])) {
      const kc = this.p.copFeedback ?? 0;
      const klim = this.p.copFeedbackLim ?? 0.04;
      cx = cl(cx + kc * (ref[0] - this.zmpMeas[0]), pl[0] - tube, pl[0] + tube);
      cy = cl(cy + kc * (ref[1] - this.zmpMeas[1]), pl[1] - tube, pl[1] + tube);
    }
    this.comCmd = [cx, cy];
    this.comTarget = [cx, cy, this.zc];
    return this.comTarget;
  }

  // ------------------------------------------------------- whole-body IK
  virtFoot(foot) {
    const o = this.siteId[foot] * 3;
    const d = this.dIk;
    return [d.site_xpos[o], d.site_xpos[o + 1], d.site_xpos[o + 2]];
  }

  updateIK(dt) {
    const mj = this.mj, model = this.model, d = this.dIk;
    const nv = this.nv;

    for (let iter = 0; iter < 2; iter++) {
      d.qpos.set(this.qRef);
      mj.mj_kinematics(model, d);
      mj.mj_comPos(model, d);

      const H = this.H, g = this.g;
      H.fill(0); g.fill(0);

      // ---- foot 6D pose tasks
      for (const foot of ['left', 'right']) {
        const isSupport = this.phaseAtNow() !== 'swing_' + foot;
        const wPos = isSupport ? this.W.supportPos : this.W.swingPos;
        const wOri = isSupport ? this.W.supportOri : this.W.swingOri;
        const o = this.siteId[foot] * 3;
        const cur = [d.site_xpos[o], d.site_xpos[o + 1], d.site_xpos[o + 2]];
        const tgt = this.footTargets[foot];
        const eo = [tgt[0] - cur[0], tgt[1] - cur[1], tgt[2] - cur[2]];
        const o9 = this.siteId[foot] * 9;
        const xm = d.site_xmat;
        const qCurRot = matToQuat([xm[o9], xm[o9 + 1], xm[o9 + 2],
                                   xm[o9 + 3], xm[o9 + 4], xm[o9 + 5],
                                   xm[o9 + 6], xm[o9 + 7], xm[o9 + 8]]);
        const er = quatLog3(quatConj(qCurRot)); // R_tgt=I, error = log(R_cur^T)
        const [jp, jr] = this.jacSite(foot);
        this.addTask(jp, eo, wPos);
        this.addTask(jr, er, wOri);
      }

      // ---- CoM task
      {
        const c = this.comTarget;
        const cur = [d.subtree_com[0], d.subtree_com[1], d.subtree_com[2]];
        const e = [c[0] - cur[0], c[1] - cur[1], c[2] - cur[2]];
        const jp = this.jacSubtreeCom();
        this.addTask(jp, e, this.W.com);
      }

      // ---- pelvis orientation task (upright)
      {
        const o4 = this.pelvisId * 4;
        const q = [d.xquat[o4], d.xquat[o4 + 1], d.xquat[o4 + 2], d.xquat[o4 + 3]];
        const er = quatLog3(quatMul([1, 0, 0, 0], quatConj(q)));
        const jr = this.jacPelvisOri();
        this.addTask(jr, er, this.W.pelvisOri);
      }

      // ---- posture task (bent-knee nominal, hinges only)
      // e = target - current (same convention as every other task):
      // pushes joint targets TOWARD the nominal posture.
      for (const [dof, qad] of this.hinges) {
        const e = this.qCrouch[qad] - this.qRef[qad];
        H[dof * nv + dof] += this.W.posture;
        g[dof] += this.W.posture * e;
      }

      // ---- damping / regularization
      for (let i = 0; i < nv; i++) H[i * nv + i] += this.damping;

      const dq = solveLinear(H, g, nv);
      // convergence gain: apply a fixed fraction of the task error per
      // iteration (mink applies dt≈0.016 -> ~1 s time constant, far too slow
      // to track a swing trajectory from a virtual configuration). During
      // the settle phase the slow rate is deliberate: it avoids resonating
      // with the position actuators.
      this.integrateQ(dq, this.settling ? 0.03 : this.p.ikGain);
      // keep the knees away from full extension: near 0 rad the leg Jacobian
      // is singular and the least-squares IK can run the hips away
      const kneeMin = this.p.kneeMin || 0.15;
      for (const qa of this.kneeQads) {
        this.qRef[qa] = Math.max(kneeMin, this.qRef[qa]);
      }
      // NaN guard: a single bad joint target must never reach the physics
      for (let i = 0; i < this.model.nu; i++) {
        const qa = this.ctrlQadr[i];
        if (!isFinite(this.qRef[qa])) this.qRef[qa] = this.qCrouch[qa];
      }
    }
    // actuator targets from the integrated virtual configuration, plus
    // servo-lag compensation: the position actuators track their target with
    // a first-order lag (gravity/load included), and that lag is what lets
    // the CoM trail the plan until the lateral whip misses its window every
    // step. Commanding  target + (target - actual) * k  cancels the steady
    // lag; dynamically it scales the servo stiffness by (1+k) while the
    // damping ratio drops only by 1/sqrt(1+k) (kp=1600, dampratio=1 -> 0.77
    // at k=0.7, still well damped).
    const kServo = this.p.servoComp ?? 0.7;
    for (let i = 0; i < model.nu; i++) {
      const qa = this.ctrlQadr[i];
      const tgt = this.qRef[qa];
      this.ctrl[i] = kServo > 0
        ? tgt + (tgt - this.data.qpos[qa]) * kServo
        : tgt;
    }
  }

  phaseAtNow() { // helper marking which foot is swinging right now
    const ph = this.plan.phaseAt(this.t);
    return ph.kind === 'ss' ? 'swing_' + ph.swing : 'none';
  }

  addTask(J, e, w) { // J: Float64Array 3*nv row-major, e: [3], w: scalar or [3]
    const nv = this.nv, H = this.H, g = this.g;
    const w0 = Array.isArray(w) ? w[0] : w;
    const w1 = Array.isArray(w) ? w[1] : w;
    const w2 = Array.isArray(w) ? w[2] : w;
    for (let r = 0; r < 3; r++) {
      const wr = r === 0 ? w0 : r === 1 ? w1 : w2;
      if (wr === 0) continue;
      const er = wr * e[r];
      for (let c = 0; c < nv; c++) {
        const j = J[r * nv + c];
        if (j === 0) continue;
        g[c] += j * er;
        for (let k = c; k < nv; k++) {
          const jk = J[r * nv + k];
          if (jk === 0) continue;
          H[c * nv + k] += wr * j * jk;
        }
      }
    }
    // mirror upper triangle
    for (let c = 0; c < nv; c++)
      for (let k = 0; k < c; k++) H[c * nv + k] = H[k * nv + c];
  }

  jacSite(foot) {
    const jp = this.jacBufs[0], jr = this.jacBufs[1];
    this.mj.mj_jacSite(this.model, this.dIk, jp, jr, this.siteId[foot]);
    return [jp.GetView(), jr.GetView()];
  }
  jacSubtreeCom() {
    const jp = this.jacBufs[2];
    this.mj.mj_jacSubtreeCom(this.model, this.dIk, jp, 0);
    return jp.GetView();
  }
  jacPelvisOri() {
    const jp = this.jacBufs[2], jr = this.jacBufs[3];
    this.mj.mj_jacBody(this.model, this.dIk, jp, jr, this.pelvisId);
    return jr.GetView();
  }

  integrateQ(dq, dt) { // hinge: q += dq; free: pos += dq, quat integrate
    const q = this.qRef;
    const m = this.model;
    for (let j = 0; j < m.njnt; j++) {
      const t = m.jnt_type[j];
      const qa = m.jnt_qposadr[j], da = m.jnt_dofadr[j];
      if (t === this.mj.mjtJoint.mjJNT_FREE.value) {
        q[qa] += dq[da] * dt; q[qa + 1] += dq[da + 1] * dt; q[qa + 2] += dq[da + 2] * dt;
        const w = [dq[da + 3] * dt, dq[da + 4] * dt, dq[da + 5] * dt];
        const half = [0.5 * w[0], 0.5 * w[1], 0.5 * w[2]];
        const dq4 = [1, half[0], half[1], half[2]]; // small-angle quat
        const n = Math.hypot(dq4[1], dq4[2], dq4[3]);
        let rot;
        if (n < 1e-12) rot = [1, 0, 0, 0];
        else {
          const ang = 2 * Math.asin(Math.min(1, n));
          rot = [Math.cos(ang / 2), dq4[1] / n * Math.sin(ang / 2),
                 dq4[2] / n * Math.sin(ang / 2), dq4[3] / n * Math.sin(ang / 2)];
        }
        const cur = [q[qa + 3], q[qa + 4], q[qa + 5], q[qa + 6]];
        const nw = quatMul(rot, cur);
        const nl = Math.hypot(nw[0], nw[1], nw[2], nw[3]);
        q[qa + 3] = nw[0] / nl; q[qa + 4] = nw[1] / nl;
        q[qa + 5] = nw[2] / nl; q[qa + 6] = nw[3] / nl;
      } else if (t === this.mj.mjtJoint.mjJNT_HINGE.value) {
        q[qa] += dq[da] * dt;
      }
    }
  }

  // ------------------------------------------------------------- physics
  measureZmp() {
    const d = this.data;
    let fz = 0, px = 0, py = 0;
    const buf = this.forceBuf;
    const n = Math.min(d.ncon, 64);
    for (let i = 0; i < n; i++) {
      const c = d.contact.get(i);
      this.mj.mj_contactForce(this.model, d, i, buf);
      const f = buf.GetView();
      const fn = f[0];
      if (fn > 0) { fz += fn; px += fn * c.pos[0]; py += fn * c.pos[1]; }
      c.delete();
    }
    this.zmpMeas = fz > 1e-6 ? [px / fz, py / fz] : [NaN, NaN];
  }

  isFallen() {
    const z = this.data.xpos[this.pelvisId * 3 + 2];
    const r22 = this.data.xmat[this.pelvisId * 9 + 8];
    const tilt = Math.acos(Math.max(-1, Math.min(1, r22))) * 180 / Math.PI;
    return !isFinite(z) || z < this.p.fallZ || tilt > this.p.fallTilt;
  }

  reset(paramsPatch) {
    Object.assign(this.p, paramsPatch || {});
    const dt = this.model.opt.timestep;
    this.mj.mj_resetDataKeyframe(this.model, this.data, 0);
    this.mj.mj_forward(this.model, this.data);
    this.qRef = Float64Array.from(this.qCrouch);
    this.t = 0; this.kIter = 0; this.phaseIdx = -1;
    this.tPlan = 0; this.clockRate = 1;
    this.fell = false; this.done = false;
    this.settle();
  }

  updatePlanClock(dti) {
    // Adaptive pacing of the gait-plan clock. The plan timeline and the
    // robot's actual CoM progress drift apart (position-actuator lag,
    // per-step landing disturbances); running the plan on the wall clock
    // lets that gap accumulate until the legs cannot span it anymore. Pace
    // the clock by BOTH the sagittal position error and the velocity
    // mismatch: a trailing or slow robot slows the whole gait down until it
    // catches up, so long walks keep plan and robot consistent.
    const pc = this.preview;
    const pl = pc.planAt(this.tPlan);
    const pl2 = pc.planAt(this.tPlan + 0.1);
    const vPlan = (pl2[0] - pl[0]) / 0.1;
    const e = this.com()[0] - pl[0];
    const vErr = (this.estVel ? this.estVel[0] : 0) - vPlan;
    const k = this.p.clockGain ?? 0;
    const kv = this.p.clockVelGain ?? 0;
    this.clockRate = Math.max(0.6, Math.min(1.3, 1 + k * e + kv * vErr));
    this.tPlan += dti * this.clockRate;
  }

  stepPhysics() { // one mujoco step (dt=0.002)
    if (this.fell || this.done) return;
    const dt = this.model.opt.timestep;
    const k = ++this.kIter;
    if ((k - 1) % this.p.ikRate === 0) {
      const dti = dt * this.p.ikRate;
      this.updatePlanClock(dti);
      this.footTargetsUpdate(this.tPlan);
      this.comTargetUpdate(this.tPlan, dti);
      this.updateIK(dti);
    }
    this.data.ctrl.set(this.ctrl);
    this.mj.mj_step(this.model, this.data);
    this.t += dt;
    this.measureZmp();
    if (this.isFallen()) { this.fell = true; this.status = 'FELL'; }
    else if (this.tPlan >= this.plan.tEnd + this.p.extraHold) {
      this.done = true; this.status = 'finished';
    }
  }
}

function matToQuat(m) { // row-major 3x3 -> wxyz quaternion
  const tr = m[0] + m[4] + m[8];
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2;
    w = (m[7] - m[5]) / s; x = 0.25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s;
  } else if (m[4] > m[8]) {
    const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2;
    w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25 * s; z = (m[5] + m[7]) / s;
  } else {
    const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2;
    w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z);
  return [w / n, x / n, y / n, z / n];
}
