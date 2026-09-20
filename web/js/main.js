// App bootstrap: MuJoCo WASM + model loading (MEMFS) + control loop + UI.
import loadMujoco from '../lib/package/mujoco.js';
import { Viewer3D } from './viewer.js';
import { WalkingController, GaitPlan } from './controller.js';

// ------------------------------------------------------------------ params
const params = {
  nSteps: 16, stepLength: 0.12, tInit: 1.0, tSS: 0.4, tDS: 0.10, tFinal: 1.0,
  extraHold: 0.5, swingHeight: 0.10, footCenterDx: 0.035, stanceInset: 0.02,
  firstSwing: 'left',
  // CoM trajectory generator: 'zmp' (Kajita preview control) | 'mpc'
  // (LIPM-ZMP linear MPC, see MpcPreview in controller.js)
  controller: 'zmp',
  mpcDt: 0.04, mpcHorizon: 1.6, mpcQCom: 10.0, mpcQVel: 10.0, mpcQZmp: 0.2,
  mpcR: 1e-2, mpcFootMargin: 0.015, mpcLeadX: 0.06, mpcLeadY: 0.12,
  mpcFeedback: 1.2, mpcTube: 0.09, mpcAdmmIters: 40,
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
    params.controller = $('in-ctrl').value === 'mpc' ? 'mpc' : 'zmp';
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
        `步态周期 ${params.tSS + params.tDS} s   控制器 = ` +
        `${params.controller === 'mpc' ? 'MPC' : 'ZMP 预观'}\n` +
        `CoP 误差 = ${zerr} cm   步数 = ${ctrl.plan ? ctrl.plan.steps.length : 0}\n` +
        `状态: ${ctrl.status}${paused ? ' (已暂停)' : ''}`;
      if (ctrl.fell) {
        $('banner').style.display = 'block';
        $('banner').textContent = '跌倒了 —— 点击“重置”重新开始';
      }
      // while recording, composite this displayed frame into the video
      if (rec) drawCompositeFrame();
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
  // lateral push disturbances — the standard probe for walking robustness.
  // 40 N ≈ 0.17 m/s CoM velocity change: inside both controllers' recovery
  // envelope (MPC demonstrably recovers where preview control sometimes
  // falls); each click adds another push, so repeated clicks escalate past
  // the envelope (~50 N single / more when repeated) and topple either mode
  const PUSH_N = 40, PUSH_S = 0.15;
  $('btn-pushl').onclick = () => {
    if (ctrl && ctrl.status === 'walking') ctrl.applyPush(-PUSH_N, PUSH_S);
  };
  $('btn-pushr').onclick = () => {
    if (ctrl && ctrl.status === 'walking') ctrl.applyPush(PUSH_N, PUSH_S);
  };
  $('btn-cam').onclick = () => {
    viewer.followCam = !viewer.followCam;
    $('btn-cam').textContent = viewer.followCam ? '📷 相机跟随' : '📷 自由视角';
  };

  // -------------------------------------------------------- video recording
  // Records EXACTLY what the user sees: every displayed frame is composited
  // onto one canvas (3D view + minimap + all text panels, drawn by
  // drawCompositeFrame below) and MediaRecorder encodes that canvas,
  // preferring the browser's MP4/H.264 path (falls back to WebM).
  const recCanvas = document.createElement('canvas');
  const recCtx = recCanvas.getContext('2d');
  window._recCanvas = recCanvas;   // exposed for debugging/testing
  let rec = null;            // MediaRecorder while recording
  let recMime = '';
  let recChunks = [];
  let recTimer = null;
  let recT0 = 0;

  function drawCompositeFrame() {
    const W = innerWidth, H = innerHeight, dpr = Math.min(devicePixelRatio, 2);
    const pw = Math.round(W * dpr), ph = Math.round(H * dpr);
    if (recCanvas.width !== pw || recCanvas.height !== ph) {
      recCanvas.width = pw; recCanvas.height = ph;
    }
    const ctx = recCtx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(viewer.renderer.domElement, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // ---- DOM overlays, repainted to match the page CSS
    const FONT = '13px "Segoe UI",Arial,"Microsoft YaHei",sans-serif';
    const roundRect = (x, y, w, h, r) => {
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
      else ctx.rect(x, y, w, h);
    };
    const panel = (x, y, w, h) => {
      ctx.fillStyle = 'rgba(8,12,26,0.72)';
      ctx.strokeStyle = 'rgba(38,50,90,1)';
      ctx.lineWidth = 1;
      roundRect(x, y, w, h, 10); ctx.fill(); ctx.stroke();
    };

    // title panel (top-left)
    ctx.font = FONT;
    const titleLines = $('title').innerText.split('\n');
    const tlh = 21, tw = Math.max(...titleLines.map((l) => ctx.measureText(l).width));
    panel(16, 14, tw + 30, 14 + titleLines.length * tlh + 2);
    ctx.textBaseline = 'top'; ctx.textAlign = 'left';
    ctx.fillStyle = '#e8eeff';
    ctx.font = '600 17px "Segoe UI",Arial,sans-serif';
    ctx.fillText(titleLines[0], 30, 22);
    ctx.font = FONT;
    titleLines.slice(1).forEach((l, i) => {
      const y = 22 + 25 + i * tlh;
      if (l.includes('Step Length:')) {           // value is highlighted red
        const head = l.slice(0, l.lastIndexOf(':') + 1);
        ctx.fillStyle = '#e8eeff'; ctx.fillText(head, 30, y);
        ctx.fillStyle = '#ff5a5a';
        ctx.fillText(l.slice(head.length), 30 + ctx.measureText(head).width, y);
        ctx.fillStyle = '#e8eeff';
      } else {
        ctx.fillStyle = l.includes('MuJoCo') ? '#9fb0d8' : '#e8eeff';
        ctx.fillText(l, 30, y);
      }
    });

    // control panel (top-right): buttons + parameter fields + selects, 2-row wrap
    const btns = [...document.querySelectorAll('#ctrl button')];
    const fields = [...document.querySelectorAll('#ctrl label')];
    const sels = [...document.querySelectorAll('#ctrl select')];
    ctx.font = FONT;
    const btnW = btns.map((b) => ctx.measureText(b.textContent).width + 28);
    const fldW = fields.map((f) => ctx.measureText(f.childNodes[0].textContent).width + 4 + 64);
    const selW = sels.map((s) => ctx.measureText(s.selectedOptions[0].textContent).width + 24);
    const rows = [[]];
    let rowW = 0;
    const lay = btns.map((b, i) => ({ kind: 'b', i, w: btnW[i] }))
      .concat(fields.map((f, i) => ({ kind: 'f', i, w: fldW[i] })))
      .concat(sels.map((s, i) => ({ kind: 's', i, w: selW[i] })));
    for (const it of lay) {
      if (rowW + it.w > 420 && rows[rows.length - 1].length) { rows.push([]); rowW = 0; }
      rows[rows.length - 1].push(it);
      rowW += it.w + 8;
    }
    const cw = Math.max(...rows.map((r) => r.reduce((a, it) => a + it.w + 8, 0)));
    const ch = rows.length * 28 + (rows.length - 1) * 8;
    const cx0 = W - 16 - cw - 28, cy0 = 14;
    panel(cx0, cy0, cw + 28, ch + 20);
    rows.forEach((row, ri) => {
      let x = cx0 + 14;
      const y = cy0 + 10 + ri * 36;
      for (const it of row) {
        if (it.kind === 'b') {
          const b = btns[it.i];
          ctx.fillStyle = b.classList.contains('rec') ? '#8f2727' : '#274b8f';
          roundRect(x, y, it.w, 28, 6); ctx.fill();
          ctx.fillStyle = '#fff';
          ctx.fillText(b.textContent, x + 14, y + 7);
        } else if (it.kind === 's') {
          const s = sels[it.i];
          ctx.fillStyle = '#141c33';
          ctx.strokeStyle = '#2a3a66';
          roundRect(x, y, it.w, 26, 6); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#e8eeff';
          ctx.fillText(s.selectedOptions[0].textContent, x + 8, y + 7);
        } else {
          const f = fields[it.i];
          ctx.fillStyle = '#9fb0d8';
          ctx.font = '12px "Segoe UI",Arial,sans-serif';
          ctx.fillText(f.childNodes[0].textContent, x, y + 8);
          ctx.font = FONT;
          const bx = x + it.w - 64;
          ctx.fillStyle = '#141c33';
          ctx.strokeStyle = '#2a3a66';
          roundRect(bx, y, 64, 26, 6); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#e8eeff';
          ctx.fillText(f.querySelector('input').value, bx + 6, y + 7);
        }
        x += it.w + 8;
      }
    });

    // minimap (right, under the controls)
    ctx.drawImage(minimap, W - 16 - minimap.width, 168);

    // status panel (bottom-left)
    ctx.font = FONT;
    const statusLines = statusEl.textContent.split('\n');
    const sw = Math.max(...statusLines.map((l) => ctx.measureText(l).width));
    const sh = statusLines.length * 20 + 16;
    panel(16, H - 14 - sh, sw + 28, sh);
    ctx.fillStyle = '#8fe08f';
    statusLines.forEach((l, i) => ctx.fillText(l, 30, H - 14 - sh + 10 + i * 20));

    // legend (bottom-right)
    const legendRows = [...document.querySelectorAll('#legend div')];
    const legendSw = ['#40ff50', '#ffffff', '#ffd24a', '#ffa030'];
    ctx.font = FONT;
    const lw = Math.max(...legendRows.map((r) => ctx.measureText(r.textContent).width)) + 30;
    const lh = legendRows.length * 21 + 16;
    const lx = W - 16 - lw, ly = H - 14 - lh;
    panel(lx, ly, lw + 4, lh);
    legendRows.forEach((r, i) => {
      const y = ly + 10 + i * 21;
      if (i < legendSw.length) {
        ctx.fillStyle = legendSw[i];
        roundRect(lx + 14, y + 7, 18, 3, 2); ctx.fill();
      }
      ctx.fillStyle = '#9fb0d8';
      ctx.fillText(r.textContent, lx + 40, y);
    });

    // fall banner
    const banner = $('banner');
    if (banner.style.display === 'block') {
      ctx.font = '700 26px "Segoe UI",Arial,sans-serif';
      ctx.fillStyle = '#ffd25a';
      ctx.textAlign = 'center';
      ctx.fillText(banner.textContent, W / 2, H * 0.38);
      ctx.textAlign = 'left';
    }

    // blinking REC dot
    if (Math.floor((performance.now() - recT0) / 600) % 2 === 0) {
      ctx.fillStyle = '#ff4040';
      ctx.beginPath(); ctx.arc(W - 24, H - 8, 5, 0, 7); ctx.fill();
    }
  }

  function toggleRecording() {
    if (!rec) {
      // prefer MP4/H.264; some browsers only mux WebM — the extension follows
      for (const c of ['video/mp4;codecs=avc1.42E01E', 'video/mp4;codecs=avc1',
                       'video/mp4', 'video/webm;codecs=vp9', 'video/webm']) {
        if (MediaRecorder.isTypeSupported(c)) { recMime = c; break; }
      }
      recCanvas.width = Math.round(innerWidth * Math.min(devicePixelRatio, 2));
      recCanvas.height = Math.round(innerHeight * Math.min(devicePixelRatio, 2));
      rec = new MediaRecorder(recCanvas.captureStream(30), {
        mimeType: recMime || undefined, videoBitsPerSecond: 8_000_000,
      });
      recChunks = [];
      rec.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
      rec.onstop = () => {
        const ext = recMime.startsWith('video/mp4') ? 'mp4' : 'webm';
        const blob = new Blob(recChunks, { type: recMime || 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `g1_walk_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.${ext}`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        window._lastRecording = { size: blob.size, ms: Math.round(performance.now() - recT0), ext };
      };
      rec.start(250);
      const t0 = recT0 = performance.now();
      $('btn-rec').classList.add('rec');
      const tick = () => {
        if (!rec) return;
        $('btn-rec').textContent = `⏹ 停止录制 ${((performance.now() - t0) / 1000).toFixed(0)}s`;
      };
      tick();
      recTimer = setInterval(tick, 500);
    } else {
      rec.stop();
      rec = null;
      clearInterval(recTimer);
      $('btn-rec').textContent = '⏺ 录制';
      $('btn-rec').classList.remove('rec');
    }
  }
  $('btn-rec').onclick = toggleRecording;

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
  // MPC predicted ZMP over the horizon (orange) — shows the planning
  // process: in recovery the prediction hugs the support-polygon edge
  if (ctrl.preview && ctrl.preview.axes) {
    ctx.strokeStyle = '#ffa030'; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let k = 0; k < ctrl.preview.N; k++) {
      const [x, y] = px(ctrl.preview.axes[0].zmp[k], ctrl.preview.axes[1].zmp[k]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.fillStyle = '#9fb0d8';
  ctx.font = '11px sans-serif';
  ctx.fillText('俯视图 (前=上)', 10, 16);
}

main().catch((e) => {
  statusEl.textContent = '错误: ' + e.message;
  console.error(e);
});
