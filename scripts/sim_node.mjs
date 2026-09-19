// Headless harness: run the EXACT browser controller (web/js/controller.js)
// under Node with the same MuJoCo WASM build, for fast quantitative iteration.
//
// Usage:
//   node scripts/sim_node.mjs [--steps N] [--len L] [--ss T] [--ds T]
//                             [--log-every T] [--quiet] [--dump FILE]
//
// Prints a per-phase summary and an overall PASS/FAIL verdict.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '../web/lib/package/mujoco.js';
import { WalkingController } from '../web/js/controller.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const WEB = path.join(ROOT, 'web');

// ------------------------------------------------------------------ CLI args
const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : dflt;
};
const has = (name) => args.includes(name);

// ------------------------------------------------------------------- params
// mirror main.js defaults
const params = {
  nSteps: parseInt(argVal('--steps', 16)), stepLength: parseFloat(argVal('--len', 0.12)),
  tInit: 1.0, tSS: parseFloat(argVal('--ss', 0.4)), tDS: parseFloat(argVal('--ds', 0.10)),
  tFinal: 1.0, extraHold: 0.5, swingHeight: parseFloat(argVal('--sh', 0.10)), footCenterDx: 0.035,
  stanceInset: 0.02, firstSwing: 'left',
  ikRate: 8, ikDamping: 1e-3, ikGain: parseFloat(argVal('--ikgain', 0.12)),
  kneeMin: parseFloat(argVal('--kneemin', 0.4)), horizonSteps: 80,
  crouchHip: -0.3, crouchKnee: 0.6, crouchAnkle: -0.3,
  dropTime: 0.8, settleTime: 1.5, tube: 0.04,
  fallZ: 0.40, fallTilt: 60,
  comLeadX: parseFloat(argVal('--leadx', 0.04)),
  comLeadY: parseFloat(argVal('--leady', 0.12)),
  comFeedback: parseFloat(argVal('--beta', 1.2)),
  comTube: parseFloat(argVal('--tube', 0.05)),
  copFeedback: parseFloat(argVal('--copfb', 0)),
  copFeedbackLim: parseFloat(argVal('--coplim', 0.04)),
  swingEndFrac: parseFloat(argVal('--swingend', 0.85)),
  servoComp: parseFloat(argVal('--servocomp', 0.7)),
  clockGain: parseFloat(argVal('--clockgain', 2)),
  clockVelGain: parseFloat(argVal('--clockvel', 0)),
  stanceInset: parseFloat(argVal('--inset', 0.02)),
  weights: {
    supportPos: parseFloat(argVal('--wsup', 400)), supportOri: 120,
    swingPos: parseFloat(argVal('--wsw', 250)), swingOri: 60,
    com: [40, 80, 120], pelvisOri: 30,
    posture: parseFloat(argVal('--wpost', 5)),
  },
};
const logEvery = parseFloat(argVal('--log-every', 0.1));
const quiet = has('--quiet');

// ------------------------------------------------------------- load mujoco
const mujoco = await loadMujoco({
  locateFile: (f) => path.join(WEB, 'lib/package', f),
});

const modelPath = path.join(WEB, 'model', 'g1_wasm.xml');
const assetDir = path.join(WEB, 'model', 'assets');
mujoco.FS.mkdirTree('/model/assets');
mujoco.FS.writeFile('/model/g1_wasm.xml', new Uint8Array(fs.readFileSync(modelPath)));
for (const f of fs.readdirSync(assetDir)) {
  mujoco.FS.writeFile(`/model/assets/${f}`, new Uint8Array(fs.readFileSync(path.join(assetDir, f))));
}

const model = mujoco.MjModel.from_xml_path('/model/g1_wasm.xml');
const data = new mujoco.MjData(model);
mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);

const gains = JSON.parse(fs.readFileSync(path.join(WEB, 'gains.json'), 'utf8'));

// ------------------------------------------------------------------- run
const ctrl = new WalkingController(mujoco, model, data, gains, params);
ctrl.reset();

const log = [];
const perPhase = new Map();   // phaseIdx -> {sumXY:[..], sumSq:[..], n, maxCopDev}
let maxLatErr = 0;            // |com_y - plan_y| overall
let maxSagErr = 0;

function record() {
  const t = ctrl.t;
  const ph = ctrl.plan.phaseAt(t);
  const c = ctrl.com();
  const ref = ctrl.plan.zmpRef(t);
  const pl = ctrl.preview.planAt(t);
  const lf = ctrl.sitePos('left'), rf = ctrl.sitePos('right');
  // virtual (IK) foot poses + swing target
  const vlf = ctrl.virtFoot('left'), vrf = ctrl.virtFoot('right');
  const tlf = ctrl.footTargets ? ctrl.footTargets.left : [NaN, NaN, NaN];
  const zerr = isFinite(ctrl.zmpMeas[0])
    ? [ref[0] - ctrl.zmpMeas[0], ref[1] - ctrl.zmpMeas[1]] : [NaN, NaN];
  // left leg joint angles: actual (qpos) vs commanded (ctrl via ctrlQadr)
  const jn = (n) => {
    const j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n);
    const qa = model.jnt_qposadr[j];
    return [data.qpos[qa], ctrl.qRef ? ctrl.qRef[qa] : NaN];
  };
  const [kA, kC] = jn('left_knee_joint');
  const [hA, hC] = jn('left_hip_pitch_joint');
  const [rA, rC] = jn('left_hip_roll_joint');
  const [apA, apC] = jn('right_ankle_pitch_joint');
  // pelvis actual vs virtual: pos + roll
  const pid = ctrl.pelvisId;
  const pAct = [data.xpos[pid * 3], data.xpos[pid * 3 + 1], data.xpos[pid * 3 + 2]];
  // pelvis roll from xmat (col-ish): use atan2 of the y-axis z comp
  const rollA = Math.atan2(data.xmat[pid * 9 + 2], data.xmat[pid * 9 + 5]);
  const vq = ctrl.dIk.xquat; const vq4 = pid * 4;
  // roll of virtual pelvis quat (w,x,y,z): rotation about x
  const rollV = Math.atan2(2 * (vq[vq4] * vq[vq4 + 1] + vq[vq4 + 2] * vq[vq4 + 3]),
                           1 - 2 * (vq[vq4 + 1] ** 2 + vq[vq4 + 2] ** 2));
  const tph = ctrl.plan.phaseAt(ctrl.tPlan ?? ctrl.t);
  log.push([t, ph.kind === 'ss' ? 1 : 0, c[0], c[1], c[2], ref[0], ref[1],
            ctrl.zmpMeas[0], ctrl.zmpMeas[1], pl[0], pl[1],
            lf[0], lf[1], lf[2], rf[0], rf[1], rf[2],
            vlf[2], vrf[2], tlf[2], vlf[0], tlf[0],
            ctrl.comCmd ? ctrl.comCmd[0] : NaN, ctrl.comCmd ? ctrl.comCmd[1] : NaN,
            kA, kC, hA, hC, rA, rC,
            pAct[0], pAct[1], pAct[2], rollA, rollV, apA, apC,
            ctrl.tPlan ?? ctrl.t, tph.kind === 'ss' ? 1 : (tph.kind === 'ds' ? 2 : 0),
            ctrl.clockRate ?? 1]);
  if (ph.kind === 'ss') {
    let st = perPhase.get(ctrl.phaseIdx);
    if (!st) { st = { sum: [0, 0], max: [0, 0], n: 0 }; perPhase.set(ctrl.phaseIdx, st); }
    for (const [i, v] of zerr.entries()) {
      if (isFinite(v)) { st.sum[i] += v; st.max[i] = Math.max(st.max[i], Math.abs(v)); }
    }
    st.n++;
  }
  maxLatErr = Math.max(maxLatErr, Math.abs(c[1] - pl[1]));
  maxSagErr = Math.max(maxSagErr, Math.abs(c[0] - pl[0]));
}

// settle already ran inside reset(); now walk
let nextLog = 0;
while (!ctrl.fell && !ctrl.done) {
  ctrl.stepPhysics();
  if (ctrl.t >= nextLog) { record(); nextLog += logEvery; }
  if (ctrl.fell || ctrl.done) break;
}
record();

// ---------------------------------------------------------------- summary
const nSS = [...perPhase.keys()].length;
const completedSS = nSS - (ctrl.fell && ctrl.status === 'FELL' ? 1 : 0);
console.log('== headless walk ==');
console.log(`params: ${params.nSteps} steps, len ${params.stepLength}, SS ${params.tSS}s, DS ${params.tDS}s, ikGain ${params.ikGain}`);
console.log(`result: status=${ctrl.status} t=${ctrl.t.toFixed(2)}s fell=${ctrl.fell} ` +
  `ss-phases-entered=${nSS} planned-steps=${ctrl.plan.steps.length}`);
console.log(`tracking: max|com_y-plan_y|=${(maxLatErr * 100).toFixed(2)}cm ` +
  `max|com_x-plan_x|=${(maxSagErr * 100).toFixed(2)}cm`);

let i = 0;
for (const [idx, st] of [...perPhase.entries()].sort((a, b) => a[0] - b[0])) {
  const ph = ctrl.plan.phases[idx];
  const mx = st.n ? st.sum[0] / st.n : NaN, my = st.n ? st.sum[1] / st.n : NaN;
  if (!quiet) {
    console.log(`  ss#${String(i++).padStart(2)} swing=${ph.swing.padEnd(5)} ` +
      `copErr mean(x,y)=(${(mx * 100).toFixed(1)}, ${(my * 100).toFixed(1)})cm ` +
      `max=(${(st.max[0] * 100).toFixed(1)}, ${(st.max[1] * 100).toFixed(1)})cm`);
  }
}

if (has('--dump')) {
  const out = argVal('--dump', 'out/sim_log.json');
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify({ params, log, fell: ctrl.fell }, null, 1));
  console.log(`logged ${log.length} samples -> ${out}`);
}

const pass = !ctrl.fell && ctrl.status === 'finished';
console.log(pass ? 'VERDICT: PASS' : 'VERDICT: FAIL');
process.exit(pass ? 0 : 1);
