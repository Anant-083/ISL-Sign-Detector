// Camera + MediaPipe Hands wiring. Depends on model.js being loaded first.

const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const bigLetter = document.getElementById("bigLetter");
const confText = document.getElementById("confText");
const barsEl = document.getElementById("bars");
const statusText = document.getElementById("statusText");
const liveDot = document.getElementById("liveDot");
const badgeText = document.getElementById("badgeText");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let camera = null;
let hands = null;
let smoothed = null;
let lockedIdx = null;
let dwell = 0;
let smoothedLandmarks = [];
const SMOOTH = 0.35;
function resizeCanvas() {
  canvasEl.width = videoEl.videoWidth || canvasEl.clientWidth;
  canvasEl.height = videoEl.videoHeight || canvasEl.clientHeight;
}

// Normalize a single hand's 21 landmarks: translate so the wrist (landmark 0)
// is the origin, then scale so the farthest landmark from the wrist is at
// distance 1. This makes features invariant to where the hand sits in frame
// and how close it is to the camera -- MUST exactly match the normalization
// used in scripts/extract_landmarks.py, or live predictions won't match
// what the model was trained on.
function normalizeHand(lm) {
  const wrist = lm[0];
  const translated = lm.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));
  let maxDist = 0;
  for (const p of translated) {
    const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (d > maxDist) maxDist = d;
  }
  const scale = maxDist > 1e-6 ? maxDist : 1e-6;
  return translated.map((p) => ({ x: p.x / scale, y: p.y / scale, z: p.z / scale }));
}

// Assumption: left hand -> features[0:63], right hand -> features[63:126].
// Change here if your training script used a different convention.
function landmarksToFeatures(results) {
  const feat = new Array(126).fill(0);
  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return null;

  const slots = { Left: null, Right: null };
  results.multiHandLandmarks.forEach((lm, idx) => {
    const handedness =
      results.multiHandedness && results.multiHandedness[idx]
        ? results.multiHandedness[idx].label
        : idx === 0
        ? "Left"
        : "Right";
    slots[handedness] = lm;
  });

  function fill(lm, offset) {
    if (!lm) return;
    const norm = normalizeHand(lm);
    for (let i = 0; i < 21; i++) {
      feat[offset + i * 3 + 0] = norm[i].x;
      feat[offset + i * 3 + 1] = norm[i].y;
      feat[offset + i * 3 + 2] = norm[i].z;
    }
  }
  fill(slots.Left, 0);
  fill(slots.Right, 63);
  return feat;
}

function renderBars(probs) {
  const idxs = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
  barsEl.innerHTML = "";
  idxs.forEach(([p, i]) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<div class="bar-letter">${Model.LABELS[i]}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
      <div class="bar-val">${(p * 100).toFixed(0)}%</div>`;
    barsEl.appendChild(row);
  });
}

function onResults(results) {
  resizeCanvas();
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

 if (results.multiHandLandmarks) {
  results.multiHandLandmarks.forEach((lm, handIdx) => {
    if (!smoothedLandmarks[handIdx]) {
      smoothedLandmarks[handIdx] = lm.map(p => ({ ...p }));
    } else {
      smoothedLandmarks[handIdx] = smoothedLandmarks[handIdx].map((p, i) => ({
        x: p.x * 0.6 + lm[i].x * 0.4,
        y: p.y * 0.6 + lm[i].y * 0.4,
        z: p.z * 0.6 + lm[i].z * 0.4,
      }));
    }
    const drawLm = smoothedLandmarks[handIdx];
    if (window.drawConnectors) drawConnectors(ctx, drawLm, HAND_CONNECTIONS, { color: "#f2a72e", lineWidth: 4 });
    if (window.drawLandmarks) drawLandmarks(ctx, drawLm, { color: "#2f9e94", lineWidth: 2, radius: 4 });
  });
} else {
  smoothedLandmarks = [];
}
  ctx.restore();

  const feat = landmarksToFeatures(results);
  if (feat) {
  const probs = Model.predict(feat);
  smoothed = smoothed ? smoothed.map((v, i) => v * (1 - 0.3) + probs[i] * 0.3) : probs;
  let best = 0, bi = 0;
  smoothed.forEach((p, i) => { if (p > best) { best = p; bi = i; } });

  // dwell-time lock: require 6 frames + 55% confidence before switching
  if (best >= 0.55) {
    if (!lockedIdx || lockedIdx === bi) { dwell = 0; lockedIdx = bi; }
    else if (++dwell >= 6) { lockedIdx = bi; dwell = 0; }
  }
  bigLetter.textContent = lockedIdx ? Model.LABELS[lockedIdx] : "—";
  confText.textContent = lockedIdx ? `confidence ${(best*100).toFixed(1)}%` : "hold steady…";
  renderBars(smoothed);
} else {
  lockedIdx = null; dwell = 0; smoothed = null;
  bigLetter.textContent = "—";
  confText.textContent = "no hand in frame";
  barsEl.innerHTML = "";
}

}

async function startCamera() {
  startBtn.disabled = true;
  statusText.textContent = "Requesting camera access...";
  try {
    await Model.loadModel("weights.json");

    hands = new Hands({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6, selfieMode: false});
    hands.onResults(onResults);

    camera = new Camera(videoEl, {
      onFrame: async () => { await hands.send({ image: videoEl }); },
      width: 1280, height: 720,
    });
    await camera.start();

    liveDot.classList.add("live");
    badgeText.textContent = "live";
    stopBtn.disabled = false;
    statusText.textContent = "Camera running. Show one or two hands, palm toward the camera.";
  } catch (err) {
    statusText.textContent = "Camera access failed: " + err.message;
    startBtn.disabled = false;
  }
}

function stopCamera() {
  if (camera) { camera.stop(); camera = null; }
  const stream = videoEl.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  videoEl.srcObject = null;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  liveDot.classList.remove("live");
  badgeText.textContent = "camera off";
  bigLetter.textContent = "—";
  confText.textContent = "confidence —";
  barsEl.innerHTML = "";
  smoothed = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusText.textContent = 'Click "Start camera" and allow access.';
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
window.addEventListener("resize", resizeCanvas);
