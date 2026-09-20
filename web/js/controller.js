// Gait planner + [online ZMP preview control | LIPM-ZMP linear MPC] +
// weighted whole-body IK, ported from the Python implementation
// (g1_zmp_walking/).
//
// References:
//  - Kajita et al. 2003, "ZMP-based walking with preview control"
//  - github.com/chauby/ZMP_preview_control, github.com/zanppa/WPG
//  - MPC: Wieber 2006 / Herdt et al. 2010, predictive control of the linear
//    inverted pendulum with ZMP-inside-support-polygon constraints
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

// LU factorization with partial pivoting (in place, row-major), returning
// the pivot permutation. luSolve then applies it: one O(n²) triangular
// solve per right-hand side — used by the MPC whose KKT matrix is constant
// across all solves of a walk.
function luFactor(M, n) {
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  for (let k = 0; k < n; k++) {
    let p = k, mx = Math.abs(M[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(M[i * n + k]);
      if (v > mx) { mx = v; p = i; }
    }
    if (p !== k) {
      for (let j = 0; j < n; j++) {
        const t = M[k * n + j]; M[k * n + j] = M[p * n + j]; M[p * n + j] = t;
      }
      const tp = piv[k]; piv[k] = piv[p]; piv[p] = tp;
    }
    const d = M[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = M[i * n + k] / d;
      M[i * n + k] = f;
      if (f !== 0) for (let j = k + 1; j < n; j++) M[i * n + j] -= f * M[k * n + j];
    }
  }
  return piv;
}
function luSolve(M, piv, n, b, out) {
  for (let i = 0; i < n; i++) out[i] = b[piv[i]];
  for (let i = 1; i < n; i++) {
    let s = out[i];
    for (let j = 0; j < i; j++) s -= M[i * n + j] * out[j];
    out[i] = s;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = out[i];
    for (let j = i + 1; j < n; j++) s -= M[i * n + j] * out[j];
    out[i] = s / M[i * n + i];
  }
  return out;
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
// Shared by both CoM generators: sample the ZMP reference at the preview
// dt and roll the preview-LQR closed-loop cart-table forward from rest —
// the canonical nominal CoM plan of this project (used by updatePlanClock
// pacing and by the headless tracking metrics).
function buildNominalPlan(gains, plan, tEnd) {
  const dt = gains.dt, nPrev = gains.n_prev;
  const Kx = gains.Kx, kr = gains.kr;
  // reference sampled at preview dt, padded at the end
  const nRef = Math.round((tEnd + 2.0) / dt) + 1;
  const refX = new Float64Array(nRef + nPrev + 2);
  const refY = new Float64Array(nRef + nPrev + 2);
  for (let k = 0; k < nRef; k++) {
    const r = plan.zmpRef(k * dt);
    refX[k] = r[0]; refY[k] = r[1];
  }
  for (let k = nRef; k < refX.length; k++) {
    refX[k] = refX[nRef - 1]; refY[k] = refY[nRef - 1];
  }
  // offline closed-loop plan (for the analysis panel & target tube)
  const nPlan = Math.round(tEnd / dt) + 1;
  const planT = new Float64Array(nPlan);
  const planX = new Float64Array(nPlan);
  const planY = new Float64Array(nPlan);
  const kMax = refX.length - nPrev - 1;
  const dotRef = (ref, k) => {
    let s = 0;
    for (let i = 0; i < nPrev; i++) s += kr[i] * ref[k + i];
    return s;
  };
  let sx = refX[0], svx = 0, sax = 0;
  let sy = refY[0], svy = 0, say = 0;
  for (let k = 0; k < nPlan; k++) {
    planT[k] = k * dt;
    planX[k] = sx; planY[k] = sy;
    const kx = Math.min(k, kMax);
    const ux = -(Kx[0] * sx + Kx[1] * svx + Kx[2] * sax) - dotRef(refX, kx);
    const uy = -(Kx[0] * sy + Kx[1] * svy + Kx[2] * say) - dotRef(refY, kx);
    sax += ux * dt; svx += sax * dt; sx += svx * dt;
    say += uy * dt; svy += say * dt; sy += svy * dt;
  }
  return { dt, refX, refY, planT, planX, planY };
}

export class OnlinePreview {
  // gains: {Kx[3], kr[N], dt, n_prev, zc}
  constructor(gains, plan, tEnd, resimWindow = 0.6) {
    this.Kx = Float64Array.from(gains.Kx);
    this.kr = Float64Array.from(gains.kr);
    this.dt = gains.dt;
    this.nPrev = gains.n_prev;
    this.zc = gains.zc;
    this.g = 9.81;
    const nominal = buildNominalPlan(gains, plan, tEnd);
    this.refX = nominal.refX; this.refY = nominal.refY;
    this.planT = nominal.planT;
    this.planX = nominal.planX; this.planY = nominal.planY;
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

// ------------------------------------------------- LIPM-ZMP linear MPC
// Receding-horizon CoM trajectory generation (Wieber 2006 / Herdt 2010
// family). The DECISION VARIABLE is the future ZMP sequence over a ~1.6 s
// horizon — constrained pointwise to the support polygon (SS: support-foot
// rectangle, DS/init/final: both-feet hull) — and the CoM follows through
// the EXACT zero-order-hold discretization of the linear inverted pendulum
// (c̈ = ω²(c − p)). The objective tracks the canonical nominal CoM plan
// (position AND velocity — without the velocity term the divergent LIPM
// mode hides excess sway speed behind an on-reference position) plus a
// weak ZMP-reference term, with a first-difference penalty keeping the
// ZMP smooth. Box constraints on the decision variable itself make the
// ADMM projection exact and M = H + ρI perfectly conditioned, so the
// warm-started iterations converge cleanly every cycle; M is factored
// ONCE per walk. The optimized trajectory starts at the MEASURED state
// and is both dynamically consistent and constraint-aware — the command
// law samples it exactly like the preview control's re-simulation.
export class MpcPreview {
  // gains: {zc, dt, ...} (dt feeds the nominal plan only), plan: GaitPlan,
  // zc: measured CoM height from settle(), p: params
  constructor(gains, plan, tEnd, zc, p) {
    this.plan = plan;
    this.zc = zc;
    this.g = 9.81;
    // canonical nominal CoM plan (same LQR rollout as OnlinePreview) for
    // updatePlanClock pacing and the headless tracking metrics
    const nominal = buildNominalPlan(gains, plan, tEnd);
    this.dtNom = nominal.dt;
    this.planT = nominal.planT;
    this.planX = nominal.planX; this.planY = nominal.planY;

    this.dt = p.mpcDt ?? 0.04;
    this.N = Math.max(8, Math.round((p.mpcHorizon ?? 1.6) / this.dt));
    this.qc = p.mpcQCom ?? 10.0;  // CoM position tracking (nominal plan)
    this.qv = p.mpcQVel ?? 10.0;  // CoM velocity tracking (nominal plan)
    this.q = p.mpcQZmp ?? 0.2;    // ZMP reference tracking (kept weak)
    this.r = p.mpcR ?? 1e-2;      // ZMP first-difference (smoothness) weight
    this.margin = p.mpcFootMargin ?? 0.015;
    this.iters = p.mpcAdmmIters ?? 40;
    // foot sole rectangle around the ankle joint (g1_walk.xml foot spheres):
    // x ∈ [-0.05, 0.12], y ∈ ±0.03; GaitPlan.footCenter() is its midpoint
    this.halfFootX = 0.085;
    this.halfFootY = 0.03;

    const N = this.N, dt = this.dt;
    const w = this.w = Math.sqrt(this.g / zc);   // LIPM eigenfrequency
    const th = w * dt, ch = this.ch = Math.cosh(th), sh = this.sh = Math.sinh(th);
    // exact ZOH of c̈ = ω²(c − p) under piecewise-constant p:
    //   X_{k+1} = Al·X_k + Bl·p_k,  X = [c, ċ]
    //   Al = [[ch, sh/w], [w·sh, ch]],  Bl = [1 − ch, −w·sh]
    const Al = [[ch, sh / w], [w * sh, ch]];
    const Bl = [1 - ch, -w * sh];
    // Toeplitz kernels of the p → (c, ċ) maps and homogeneous rows e·Al^{k+1}
    this.kc = new Float64Array(N);
    this.kv = new Float64Array(N);
    this.cR = new Float64Array(2 * N);
    this.vR = new Float64Array(2 * N);
    let v = [Bl[0], Bl[1]];                 // Al^0·Bl
    let r1 = [1, 0], r2 = [0, 1];           // e1, e2
    for (let k = 0; k < N; k++) {
      this.kc[k] = v[0];
      this.kv[k] = v[1];
      const r1n = [r1[0] * ch + r1[1] * w * sh, r1[0] * (sh / w) + r1[1] * ch];
      const r2n = [r2[0] * ch + r2[1] * w * sh, r2[0] * (sh / w) + r2[1] * ch];
      this.cR[2 * k] = r1n[0]; this.cR[2 * k + 1] = r1n[1];
      this.vR[2 * k] = r2n[0]; this.vR[2 * k + 1] = r2n[1];
      v = [ch * v[0] + (sh / w) * v[1], w * sh * v[0] + ch * v[1]];
      r1 = r1n; r2 = r2n;
    }
    const Kc = new Float64Array(N * N);
    const Kv = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j <= i; j++) {
        Kc[i * N + j] = this.kc[i - j];
        Kv[i * N + j] = this.kv[i - j];
      }
    }
    this.Kc = Kc; this.Kv = Kv;
    // Hessian H = 2(qc·KcᵀKc + qv·KvᵀKv + q·I) + 2r·DᵀD   (D = first difference)
    const H = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let sc = 0, sv = 0;
        for (let m = 0; m < N; m++) {
          sc += Kc[m * N + i] * Kc[m * N + j];
          sv += Kv[m * N + i] * Kv[m * N + j];
        }
        H[i * N + j] = 2 * (this.qc * sc + this.qv * sv + (i === j ? this.q : 0));
      }
    }
    for (let i = 0; i < N; i++) {
      H[i * N + i] += 2 * this.r * (i === 0 || i === N - 1 ? 1 : 2);
      if (i > 0) {
        H[i * N + i - 1] -= 2 * this.r;
        H[(i - 1) * N + i] -= 2 * this.r;
      }
    }
    // ADMM with the box ON the decision variable:
    //   U ← (H + ρI)⁻¹(−grad + ρ(z − y));  z ← clamp(U + y);  y ← y + U − z
    const rho = this.rho = 1.0;
    const M = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) M[i] = H[i];
    for (let i = 0; i < N; i++) M[i * N + i] += rho;
    this.Mlu = M;
    this.Mpiv = luFactor(M, N);
    // per-axis workspaces + warm-start state
    this.axes = [0, 1].map(() => ({
      U: new Float64Array(N), z: new Float64Array(N), y: new Float64Array(N),
      lb: new Float64Array(N), ub: new Float64Array(N), ref: new Float64Array(N),
      cref: new Float64Array(N), vref: new Float64Array(N),
      cR0: new Float64Array(N), vR0: new Float64Array(N),
      grad: new Float64Array(N), rhs: new Float64Array(N),
      traj: new Float64Array(N + 1), trajV: new Float64Array(N + 1),
      zmp: new Float64Array(N),
    }));
  }
  planAt(t) { // nominal plan, same semantics as OnlinePreview.planAt
    const n = this.planT.length;
    const k = Math.min(Math.max(t / this.dtNom, 0), n - 1.001);
    const i = Math.floor(k), f = k - i;
    const j = Math.min(i + 1, n - 1);
    return [this.planX[i] + (this.planX[j] - this.planX[i]) * f,
            this.planY[i] + (this.planY[j] - this.planY[i]) * f];
  }
  // support-polygon bounds [xlo, xhi, ylo, yhi] at plan-clock time t
  boundsAt(t) {
    const hx = this.halfFootX - this.margin, hy = this.halfFootY - this.margin;
    const ph = this.plan.phaseAt(t);
    if (ph.kind === 'ss') {
      const c = this.plan.zmpRef(t);   // SS: the support-foot center
      return [c[0] - hx, c[0] + hx, c[1] - hy, c[1] + hy];
    }
    // init / ds / final: both feet down -> hull of the two sole rectangles
    const cl = this.plan.footCenter('left', this.plan.footX('left', t));
    const cr = this.plan.footCenter('right', this.plan.footX('right', t));
    return [Math.min(cl[0], cr[0]) - hx, Math.max(cl[0], cr[0]) + hx,
            Math.min(cl[1], cr[1]) - hy, Math.max(cl[1], cr[1]) + hy];
  }
  // Re-solve both axis QPs from the measured CoM state. Decision variable
  // U = ZMP sequence: U[k] holds over [t + k·dt, t + (k+1)·dt); outputs are
  // sampled at t + (k+1)·dt. Fills traj[0..N] (CoM, traj[0] = measured) —
  // sample with trajAt().
  solve(t, pos, vel, acc) {
    const N = this.N, dt = this.dt;
    const Kc = this.Kc, Kv = this.Kv, cR = this.cR, vR = this.vR;
    const rho = this.rho, ch = this.ch, sh = this.sh, w = this.w;
    for (let ax = 0; ax < 2; ax++) {
      const a = this.axes[ax];
      // warm start: shift the previous solution one step left (done FIRST
      // so that after solve() returns, U[0]/traj all describe the solution
      // just computed)
      for (let k = 0; k < N - 1; k++) {
        a.U[k] = a.U[k + 1]; a.z[k] = a.z[k + 1]; a.y[k] = a.y[k + 1];
      }
      const x0 = pos[ax], v0 = vel[ax];
      for (let k = 0; k < N; k++) {
        const bm = this.boundsAt(t + (k + 0.5) * dt);
        a.lb[k] = bm[2 * ax]; a.ub[k] = bm[2 * ax + 1];
        const tk = t + (k + 1) * dt;
        a.ref[k] = this.plan.zmpRef(tk)[ax];
        a.cref[k] = this.planAt(tk)[ax];
        a.vref[k] = (this.planAt(tk + dt)[ax] - a.cref[k]) / dt;
        a.cR0[k] = cR[2 * k] * x0 + cR[2 * k + 1] * v0;
        a.vR0[k] = vR[2 * k] * x0 + vR[2 * k + 1] * v0;
      }
      // objective gradient at U = 0:
      //   2qc·Kcᵀ(cR0 − cref) + 2qv·Kvᵀ(vR0 − vref) − 2q·zmpRef
      for (let i = 0; i < N; i++) {
        let sc = 0, sv = 0;
        for (let j = i; j < N; j++) {
          sc += Kc[j * N + i] * (a.cR0[j] - a.cref[j]);
          sv += Kv[j * N + i] * (a.vR0[j] - a.vref[j]);
        }
        a.grad[i] = 2 * (this.qc * sc + this.qv * sv) - 2 * this.q * a.ref[i];
      }
      for (let it = 0; it < this.iters; it++) {
        for (let i = 0; i < N; i++) a.rhs[i] = -a.grad[i] + rho * (a.z[i] - a.y[i]);
        luSolve(this.Mlu, this.Mpiv, N, a.rhs, a.U);
        for (let i = 0; i < N; i++) {
          const zi = Math.min(a.ub[i], Math.max(a.lb[i], a.U[i] + a.y[i]));
          a.y[i] += a.U[i] - zi;
          a.z[i] = zi;
        }
      }
      // roll the CoM trajectory under the optimal ZMP (exact LIPM ZOH)
      let cx = x0, cv = v0;
      a.traj[0] = cx;
      a.trajV[0] = cv;
      for (let k = 0; k < N; k++) {
        const p = a.U[k];
        const cn = ch * cx + (sh / w) * cv + (1 - ch) * p;
        cv = w * sh * cx + ch * cv - w * sh * p;
        cx = cn;
        a.traj[k + 1] = cx;
        a.trajV[k + 1] = cv;
        a.zmp[k] = p;
      }
    }
  }
  // sample the last-solved optimized CoM trajectory tau seconds ahead of
  // the solve origin → [x, y]
  trajAt(tau) {
    const k = Math.min(Math.max(tau / this.dt, 0), this.N);
    const i = Math.floor(k), f = k - i;
    const j = Math.min(i + 1, this.N);
    return [this.axes[0].traj[i] + (this.axes[0].traj[j] - this.axes[0].traj[i]) * f,
            this.axes[1].traj[i] + (this.axes[1].traj[j] - this.axes[1].traj[i]) * f];
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
    this.pushFy = 0; this.pushLeft = 0; this.captureUntil = -1;
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
    this.preview = this.p.controller === 'mpc'
      ? new MpcPreview(this.gains, this.plan, this.plan.tEnd, this.zc, this.p)
      : new OnlinePreview(this.gains, this.plan, this.plan.tEnd);
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
      // LATERAL CAPTURE, footstep re-placement (MPC mode): if the CoM
      // enters the swing clearly displaced from the plan's lane, shift the
      // swing foot's footfall toward it — the capture foot must be UNDER
      // the falling CoM, not 4 cm off to the side. plan.footY is read live
      // by zmpRef(), boundsAt() and the swing-goal code below, so the ZMP
      // reference and the MPC support constraints follow the new landing
      // automatically. Only the SWING foot's reference moves (the stance
      // foot did not move); the other foot re-centers the lane at its own
      // next swing. Timing alone (the clock pacing in updatePlanClock)
      // cannot recover a push whose capture point lies outside the lane.
      if (ph.kind === 'ss' && this.p.controller === 'mpc'
          && this.t < this.captureUntil) {
        const eLat = this.com()[1] - this.preview.planAt(t)[1];
        if (Math.abs(eLat) > 0.04) {
          const shift =
            Math.max(-0.05, Math.min(0.05, 0.7 * (eLat - Math.sign(eLat) * 0.04)));
          this.plan.footY[ph.swing] += shift;
          // NOTE: the frozen nominal CoM plan deliberately stays on the old
          // lane — the command law's beta·(plan − mpcTraj) term NEEDS the
          // uncompromised plan as its reference; re-rolling the plan toward
          // the robot would cancel the very correction that performs the
          // recovery.
        }
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

    if (this.p.controller === 'mpc') {
      // LIPM-ZMP linear MPC: re-solve the support-polygon-constrained
      // receding-horizon QP from the measured state — see MpcPreview. The
      // command law mirrors the proven preview controller exactly, with
      // the MPC's constraint-aware trajectory substituted for the LQR
      // re-simulation:
      //   cmd = plan(t+lead) + beta·(plan(t+lead) − mpcTraj(t+lead))
      // The plan carries the deep countdown (the QP re-solved from rest is
      // always lazier than the plan); the MPC trajectory starts at the
      // measured state and respects the upcoming support polygons, so the
      // correction bends the command back onto a feasible path. Tube-
      // clamped around the plan like comTube in the preview mode.
      const mpc = this.preview;
      mpc.solve(t, pos, this.estVel, this.estAcc);
      const leadX = this.p.mpcLeadX ?? 0.04, leadY = this.p.mpcLeadY ?? 0.12;
      const beta = this.p.mpcFeedback ?? 1.2;
      const plX = mpc.planAt(t + leadX), plY = mpc.planAt(t + leadY);
      const trX = mpc.trajAt(leadX)[0], trY = mpc.trajAt(leadY)[1];
      const tx = plX[0] + beta * (plX[0] - trX);
      const ty = plY[1] + beta * (plY[1] - trY);
      const pl = mpc.planAt(t);
      const tube = this.p.mpcTube ?? 0.05;
      const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      const cx = cl(tx, pl[0] - tube, pl[0] + tube);
      const cy = cl(ty, pl[1] - tube, pl[1] + tube);
      this.rawCmd = [tx, ty];
      this.comCmd = [cx, cy];
      this.comTarget = [cx, cy, this.zc];
      return this.comTarget;
    }

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
    //
    // MPC mode adds CAPTURE-STEP PACING: a lateral CoM deviation from the
    // plan means the fixed footstep timing is wrong for the disturbance —
    // falling toward the SWING side needs the swing foot down NOW (end the
    // single support early), falling toward the STANCE side needs the
    // stance foot to keep decelerating (stretch it). The effect of the
    // lateral error on the clock therefore flips with the swing side.
    // Without this the QP can only watch its own predicted fall: the
    // support-polygon constraints come from the planned timeline, and the
    // one transition that could save the robot arrives too late.
    const pc = this.preview;
    const pl = pc.planAt(this.tPlan);
    const pl2 = pc.planAt(this.tPlan + 0.1);
    const vPlan = (pl2[0] - pl[0]) / 0.1;
    const c = this.com();
    const e = c[0] - pl[0];
    const vErr = (this.estVel ? this.estVel[0] : 0) - vPlan;
    const k = this.p.clockGain ?? 0;
    const kv = this.p.clockVelGain ?? 0;
    let rate = 1 + k * e + kv * vErr;
    let clampHi = 1.3, clampLo = 0.6;
    // CAPTURE-STEP PACING (MPC, always on): falling toward the swing side
    // compresses the swing (land early — its ZMP authority is needed now);
    // velocity (not position) error — after a capture the plan's sway
    // phase stays offset from the robot, and a position term would stay
    // biased. Nominal gait tolerates this: verified PASS with unchanged
    // nominal tracking. A lateral recovery also steals forward momentum,
    // so the sagittal pacing gets extra room to wait (clampLo 0.5).
    if (this.p.controller === 'mpc') {
      clampHi = 1.5;
      const ph = this.plan.phaseAt(this.tPlan);
      if (ph.kind === 'ss') {
        const vPlanY = (pc.planAt(this.tPlan + 0.1)[1] - pl[1]) / 0.1;
        const vLatErr = (this.estVel ? this.estVel[1] : 0) - vPlanY;
        const dir = ph.swing === 'left' ? 1 : -1;   // +vy = toward left foot
        const dead = Math.sign(vLatErr) *
          Math.max(0, Math.abs(vLatErr * dir) - 0.12);
        rate += (this.p.clockLatGain ?? 1.5) * dead;
      }
    }
    this.clockRate = Math.max(clampLo, Math.min(clampHi, rate));
    this.tPlan += dti * this.clockRate;
  }

  // horizontal push disturbance: apply force fy (N, world y) to the pelvis
  // for dur seconds — the standard way to probe walking robustness (the
  // MPC's support-polygon constraints exist exactly for this). Also ARMS
  // the capture machinery (footstep re-placement) for a window: nominal
  // sway reaches the same CoM deviations a real push leaves behind, so the
  // capture must never trigger on undisturbed walking.
  applyPush(fy, dur = 0.15) {
    this.pushFy = fy;
    this.pushLeft = dur;
    this.captureUntil = (this.t ?? 0) + 2.5;
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
    if (this.pushLeft > 0) {
      this.data.xfrc_applied[this.pelvisId * 6 + 1] = this.pushFy;
      this.pushLeft -= dt;
    } else if (this.pushFy) {
      this.data.xfrc_applied[this.pelvisId * 6 + 1] = 0;
      this.pushFy = 0;
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
