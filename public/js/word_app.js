const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const wordOutput = document.getElementById("wordOutput");
const wordConf = document.getElementById("wordConf");
const debugGuesses = document.getElementById("debugGuesses");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");
const liveDot = document.getElementById("liveDot");
const badgeText = document.getElementById("badgeText");

let camera = null;
let hands = null;
let pose = null;
let latestHands = null;
let latestPose = null;
let predicting = false;
let loopHandle = null;
let currentPrediction = { word: null, confidence: 0 };

let stableWord = null;
let stableCount = 0;
const STABLE_THRESHOLD = 3;   // same top word for 3 cycles in a row
const CONF_THRESHOLD = 0.55;  // minimum confidence to count as "final"

function resizeCanvas() {
  canvasEl.width = videoEl.videoWidth || canvasEl.clientWidth;
  canvasEl.height = videoEl.videoHeight || canvasEl.clientHeight;
}

function tryCombine() {
  if (!latestHands || !latestPose) return;
  onResults({
    poseLandmarks: latestPose.poseLandmarks,
    leftHandLandmarks: latestHands.multiHandLandmarks?.[latestHands.multiHandedness?.findIndex(h => h.label === "Left")],
    rightHandLandmarks: latestHands.multiHandLandmarks?.[latestHands.multiHandedness?.findIndex(h => h.label === "Right")],
  });
}

function onResults(results) {
  pushFrame(results);

  resizeCanvas();
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  if (results.poseLandmarks) {
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#2f9e94", lineWidth: 3 });
    drawLandmarks(ctx, results.poseLandmarks, { color: "#f2a72e", lineWidth: 1, radius: 3 });
  }
  if (results.leftHandLandmarks) {
    drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#f2a72e", lineWidth: 3 });
    drawLandmarks(ctx, results.leftHandLandmarks, { color: "#2f9e94", lineWidth: 1, radius: 3 });
  }
  if (results.rightHandLandmarks) {
    drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#f2a72e", lineWidth: 3 });
    drawLandmarks(ctx, results.rightHandLandmarks, { color: "#2f9e94", lineWidth: 1, radius: 3 });
  }

  // Bounding box + label around detected hand(s)
  const allPoints = [];
  if (results.leftHandLandmarks) allPoints.push(...results.leftHandLandmarks);
  if (results.rightHandLandmarks) allPoints.push(...results.rightHandLandmarks);

  if (allPoints.length > 0 && currentPrediction.word) {
    const xs = allPoints.map(p => p.x * canvasEl.width);
    const ys = allPoints.map(p => p.y * canvasEl.height);
    const minX = Math.min(...xs) - 30, maxX = Math.max(...xs) + 30;
    const minY = Math.min(...ys) - 30, maxY = Math.max(...ys) + 30;

    ctx.strokeStyle = "#f2a72e";
    ctx.lineWidth = 3;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

    const label = `${currentPrediction.word}: ${(currentPrediction.confidence * 100).toFixed(1)}%`;
    ctx.font = "bold 18px sans-serif";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = "#f2a72e";
    ctx.fillRect(minX, minY - 28, textWidth + 16, 28);

    ctx.fillStyle = "#1a1a1a";
    ctx.fillText(label, minX + 8, minY - 8);
  }

  ctx.restore();
}

async function startCamera() {
  startBtn.disabled = true;
  try {
    statusText.textContent = "Loading word model...";
    await loadWordModel();

    statusText.textContent = "Word model loaded. Starting MediaPipe...";
    hands = new Hands({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    });
    hands.setOptions({ modelComplexity: 0, maxNumHands: 2 });
    hands.onResults((r) => { latestHands = r; tryCombine(); });

    pose = new Pose({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`
    });
    pose.setOptions({ modelComplexity: 0 });
    pose.onResults((r) => { latestPose = r; tryCombine(); });

    statusText.textContent = "Requesting camera...";
    camera = new Camera(videoEl, {
      onFrame: async () => {
        await hands.send({ image: videoEl });
        await pose.send({ image: videoEl });
      },
      width: 480, height: 360,
    });
    await camera.start();

    liveDot.classList.add("live");
    badgeText.textContent = "live";
    statusText.textContent = "Camera running. Perform a sign.";
    stopBtn.disabled = false;
    loopHandle = setInterval(predictLoop, 700);
  } catch (err) {
    statusText.textContent = "ERROR: " + (err && err.message ? err.message : String(err));
    console.error("startCamera failed:", err);
    startBtn.disabled = false;
  }
}

async function predictLoop() {
  if (predicting) return;

  const sample = getNormalizedSample();
  if (!sample) {
    wordOutput.textContent = "—";
    wordConf.textContent = "show your hands";
    debugGuesses.textContent = "";
    statusText.textContent = `Buffering frames (${frameWindow.length}/${SEQ_LEN})...`;
    stableWord = null;
    stableCount = 0;
    return;
  }

  predicting = true;
  statusText.textContent = "Predicting...";
  try {
    const { top, word, confidence } = await predictWord(sample);
    currentPrediction = { word, confidence };

    debugGuesses.innerHTML =
      "guesses: " + top.map(t => `${t.word} (${(t.confidence * 100).toFixed(1)}%)`).join(", ");

    if (word === stableWord) {
      stableCount++;
    } else {
      stableWord = word;
      stableCount = 1;
    }

    if (stableCount >= STABLE_THRESHOLD && confidence >= CONF_THRESHOLD) {
      wordOutput.textContent = word;
      wordConf.textContent = `FINAL — confidence ${(confidence * 100).toFixed(1)}%`;
      statusText.textContent = `Locked in: "${word}"`;
    } else {
      wordOutput.textContent = word + " ...";
      wordConf.textContent = `guessing — ${(confidence * 100).toFixed(1)}%`;
      statusText.textContent = `Stabilizing (${stableCount}/${STABLE_THRESHOLD} matches on "${word}")...`;
    }
  } catch (err) {
    console.error("predictWord failed (skipped this cycle):", err);
    statusText.textContent = "Prediction hiccup, retrying next cycle...";
  }
  predicting = false;
}

function stopCamera() {
  if (camera) { camera.stop(); camera = null; }
  const stream = videoEl.srcObject;
  if (stream) stream.getTracks().forEach(t => t.stop());
  videoEl.srcObject = null;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  clearInterval(loopHandle);
  frameWindow = [];
  latestHands = null;
  latestPose = null;
  stableWord = null;
  stableCount = 0;
  currentPrediction = { word: null, confidence: 0 };
  liveDot.classList.remove("live");
  badgeText.textContent = "camera off";
  wordOutput.textContent = "—";
  wordConf.textContent = "confidence —";
  debugGuesses.textContent = "";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusText.textContent = 'Click "Start camera" and allow access.';
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
window.addEventListener("resize", resizeCanvas);
