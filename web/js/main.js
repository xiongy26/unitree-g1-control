// App bootstrap: MuJoCo WASM + model loading (MEMFS) + control loop + UI.
import loadMujoco from '../lib/package/mujoco.js';
import { Viewer3D } from './viewer.js';
import { WalkingController, GaitPlan } from './controller.js';

// ------------------------------------------------------------------ params
const params = {
  nSteps: 16, stepLength: 0.12, tInit: 1.0, tSS: 0.4, tDS: 0.10, tFinal: 1.0,
  extraHold: 0.5, swingHeight: 0.10, footCenterDx: 0.035, stanceInset: 0.02,
  firstSwing: 'left',
  ikRate: 8, ikDamping: 1e-3, ikGain: 0.12, kneeMin: 0.4, horizonSteps: 80,
  crouchHip: -0.3, crouchKnee: 0.6, crouchAnkle: -0.3,
  dropTime: 0.8, settleTime: 1.5, tube: 0.04,
  fallZ: 0.40, fallTilt: 60,
  // receding-horizon CoM law: cmd = plan(t+lead) + beta*(plan(t+lead) - resim(t+lead))
  comLeadX: 0.04, comLeadY: 0.12, comFeedback: 1.2, comTube: 0.05,
  servoComp: 0.7,          // servo-lag compensation on the actuator targets
  clockGain: 2, clockVelGain: 0, // adaptive plan-clock pacing
  swingEndFrac: 0.85,      // swing path completes within this fraction of SS
  copFeedback: 0, copFeedbackLim: 0.04,
  weights: {
    supportPos: 400, supportOri: 120, swingPos: 250, swingOri: 60,
    com: [40, 80, 120], pelvisOri: 30, posture: 5,
  },
};

const $ = (id) => document.getElementById(id);
const statusEl = $('status');

window.addEventListener('error', (e) => {
  statusEl.textContent = '错误: ' + e.message + '\n' + (e.filename || '') + ':' + e.lineno;
});
window.addEventListener('unhandledrejection', (e) => {
  statusEl.textContent = '错误: ' + (e.reason && e.reason.message ? e.reason.message : e.reason);
});

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function main() {
  statusEl.textContent = '加载 MuJoCo WASM …';
  const mujoco = await loadMujoco({
    locateFile: (f) => `./lib/package/${f}`,
  });

  statusEl.textContent = '加载模型与网格 …';
  const xmlBytes = await fetchBytes('./model/g1_wasm.xml');
  // all mesh assets referenced by the XML must be in the virtual FS before
  // compilation (the renderer reads the COMPILED buffers, not these files)
  const xmlText = new TextDecoder().decode(xmlBytes);
  const meshNames = new Set();
  for (const m of xmlText.matchAll(/file="([^"]+)"/g)) meshNames.add(m[1]);

  mujoco.FS.mkdirTree('/model/assets');
  mujoco.FS.writeFile('/model/g1_wasm.xml', xmlBytes);
  let i = 0;
  for (const name of meshNames) {
    statusEl.textContent = `加载网格 ${++i}/${meshNames.size} …`;
    const bytes = await fetchBytes(`./model/assets/${name}`);
    mujoco.FS.writeFile(`/model/assets/${name}`, bytes);
  }

  statusEl.textContent = '编译模型 …';
  const model = mujoco.MjModel.from_xml_path('/model/g1_wasm.xml');
  const data = new mujoco.MjData(model);
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);

  const gains = await (await fetch('./gains.json')).json();
  const viewer = new Viewer3D($('app'), mujoco, model);
  const minimap = $('minimap');
  const mmCtx = minimap.getContext('2d');

  // title panel values
  $('t-ss').textContent = params.tSS;
  $('t-ds').textContent = params.tDS;
  $('t-len').textContent = `${params.stepLength} m`;

  let ctrl = null;
  let paused = false;

  function buildController() {
    ctrl = new WalkingController(mujoco, model, data, gains, params);
    window.ctrl = ctrl; window.mujoco = mujoco; window.model = model; window.data = data;
    statusEl.textContent = '初始化（落地 + 平衡整定）…';
  }

  function resetSim() {
    params.nSteps = parseInt($('in-steps').value) || 16;
    params.stepLength = parseFloat($('in-len').value) || 0.12;
    params.tSS = parseFloat($('in-ss').value) || 0.4;
    params.tDS = parseFloat($('in-ds').value) || 0.1;
    $('t-ss').textContent = params.tSS;
    $('t-ds').textContent = params.tDS;
    $('t-len').textContent = `${params.stepLength} m`;
    $('banner').style.display = 'none';
    paused = false;
    $('btn-pause').textContent = '⏸ 暂停';
    buildController();
    // let the UI paint, then run the (blocking) settle phase
    setTimeout(() => {
      reanchorTicker();
      ctrl.reset();
      viewer.resetTrails(ctrl.plan, samplePlanPath(ctrl.plan));
      statusEl.textContent = '开始行走';
    }, 50);
  }

  function samplePlanPath(plan) {
    const pts = [];
    for (let t = 0; t < plan.tEnd; t += 0.02) {
      const r = plan.zmpRef(t);
      pts.push(r);
    }
    return pts;
  }

  // ---------------------------------------------------------------- loop
  // Physics is driven by a Web Worker timer: window timers are throttled to
  // ~1 Hz for background tabs, which would freeze the robot; worker timers
  // keep firing. Rendering stays on a window timer capped at ~30 fps.
  // The number of physics steps per tick is set by the WALL clock so the
  // simulation runs in real time — a fixed batch of 20 steps per 4 ms tick
  // runs the walk at ~3-10x speed, which reads as "being shoved forward".
  const simDt = model.opt.timestep;
  let tickAnchor = null;
  const tickerBlob = new Blob(
    ['setInterval(() => postMessage(0), 4);'],
    { type: 'application/javascript' });
  const ticker = new Worker(URL.createObjectURL(tickerBlob));
  ticker.onmessage = () => {
    if (!ctrl || paused || ctrl.status === 'init') return;
    const now = performance.now();
    if (tickAnchor === null) { tickAnchor = now; return; }
    let nSteps = Math.floor((now - tickAnchor) / 1000 / simDt);
    if (nSteps <= 0) return;
    nSteps = Math.min(nSteps, 16);   // cap catch-up bursts (tab stalls)
    tickAnchor += nSteps * simDt * 1000;
    if (now - tickAnchor > 500) tickAnchor = now;  // drop long backlogs
    for (let n = 0; n < nSteps; n++) {
      ctrl.stepPhysics();
      if (ctrl.fell || ctrl.done) break;
    }
  };
  const reanchorTicker = () => { tickAnchor = null; };
  setInterval(() => {
    try {
      if (!ctrl) return;
      viewer.syncFromData(data);
      const com = ctrl.com();
      viewer._pushTrail(viewer.comTrail, [com[0], com[1]]);
      viewer.updateCamera(com);
      viewer.render();
      drawMinimap(mmCtx, ctrl, minimap);

      // status line — the plan clock (tPlan), not the wall time, drives the
      // gait phases (see updatePlanClock), so readouts must use it too
      const tp = ctrl.tPlan ?? ctrl.t;
      const ph = ctrl.plan ? ctrl.plan.phaseAt(tp) : null;
      const zerr = ctrl.plan && isFinite(ctrl.zmpMeas[0])
        ? (Math.hypot(ctrl.plan.zmpRef(tp)[0] - ctrl.zmpMeas[0],
                      ctrl.plan.zmpRef(tp)[1] - ctrl.zmpMeas[1]) * 100).toFixed(1)
        : '--';
      statusEl.textContent =
        `t = ${tp.toFixed(2)} s   阶段 = ${ph ? ph.kind : '-'}   ` +
        `步态周期 ${params.tSS + params.tDS} s\n` +
        `CoP 误差 = ${zerr} cm   步数 = ${ctrl.plan ? ctrl.plan.steps.length : 0}\n` +
        `状态: ${ctrl.status}${paused ? ' (已暂停)' : ''}`;
      if (ctrl.fell) {
        $('banner').style.display = 'block';
        $('banner').textContent = '跌倒了 —— 点击“重置”重新开始';
      }
    } catch (e) {
      if (!window._frameErr) {
        window._frameErr = true;
        statusEl.textContent = '渲染错误: ' + e.message + '\n' + (e.stack || '').slice(0, 300);
      }
    }
  }, 33);
  window.viewer = viewer;

  // ---------------------------------------------------------------- UI
  $('btn-pause').onclick = () => {
    paused = !paused;
    $('btn-pause').textContent = paused ? '▶ 继续' : '⏸ 暂停';
    reanchorTicker();
  };
  $('btn-reset').onclick = resetSim;
  $('btn-cam').onclick = () => {
    viewer.followCam = !viewer.followCam;
    $('btn-cam').textContent = viewer.followCam ? '📷 相机跟随' : '📷 自由视角';
  };

  // first boot
  buildController();
  setTimeout(() => {
    reanchorTicker();
    ctrl.reset();
    viewer.resetTrails(ctrl.plan, samplePlanPath(ctrl.plan));
  }, 50);
}

function drawMinimap(ctx, ctrl, canvas) {
  if (!ctrl.plan) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const com = ctrl.com();
  const cx = com[0], cy = com[1];
  const scale = 88; // px per meter
  const W = canvas.width, H = canvas.height;
  const px = (x, y) => [W / 2 + (y - cy) * scale, H - 30 - (x - cx) * scale];

  // grid
  ctx.strokeStyle = 'rgba(90,110,160,0.25)';
  ctx.lineWidth = 1;
  for (let dx = -3; dx <= 3; dx++) {
    const [x1, y1] = px(Math.round(cx) + dx, cy - 3);
    const [x2, y2] = px(Math.round(cx) + dx, cy + 3);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const [u1, v1] = px(cx - 3, Math.round(cy) + dx);
    const [u2, v2] = px(cx + 3, Math.round(cy) + dx);
    ctx.beginPath(); ctx.moveTo(u1, v1); ctx.lineTo(u2, v2); ctx.stroke();
  }
  // planned zmp path
  ctx.strokeStyle = '#40ff50'; ctx.lineWidth = 3;
  ctx.beginPath();
  let first = true;
  for (let t = 0; t < ctrl.plan.tEnd; t += 0.04) {
    const r = ctrl.plan.zmpRef(t);
    const [x, y] = px(r[0], r[1]);
    if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // footfalls
  ctx.fillStyle = '#ffd24a';
  for (const [fx, fy] of ctrl.plan.footfallPoints()) {
    const [x, y] = px(fx, fy);
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill();
  }
  // com trail
  const tr = ctrl.comTrailPx || (ctrl.comTrailPx = []);
  tr.push([com[0], com[1]]);
  if (tr.length > 400) tr.shift();
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3;
  ctx.beginPath();
  tr.forEach(([x, y], i) => {
    const [pxx, pyy] = px(x, y);
    if (i === 0) ctx.moveTo(pxx, pyy); else ctx.lineTo(pxx, pyy);
  });
  ctx.stroke();
  // measured zmp dot
  if (isFinite(ctrl.zmpMeas[0])) {
    const [x, y] = px(ctrl.zmpMeas[0], ctrl.zmpMeas[1]);
    ctx.fillStyle = '#ff5a5a';
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fill();
  }
  ctx.fillStyle = '#9fb0d8';
  ctx.font = '11px sans-serif';
  ctx.fillText('俯视图 (前=上)', 10, 16);
}

main().catch((e) => {
  statusEl.textContent = '错误: ' + e.message;
  console.error(e);
});
