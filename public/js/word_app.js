const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const wordOutput = document.getElementById("wordOutput");
const wordConf = document.getElementById("wordConf");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");
const liveDot = document.getElementById("liveDot");
const badgeText = document.getElementById("badgeText");

let camera = null;
let holistic = null;
let predicting = false;
let loopHandle = null;

function resizeCanvas() {
  canvasEl.width = videoEl.videoWidth || canvasEl.clientWidth;
  canvasEl.height = videoEl.videoHeight || canvasEl.clientHeight;
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

  ctx.restore();
}

async function startCamera() {
  startBtn.disabled = true;
  try {
    statusText.textContent = "Loading word model...";
    await loadWordModel();

    statusText.textContent = "Word model loaded. Starting MediaPipe...";
    holistic = new Holistic({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
    });
    holistic.setOptions({ modelComplexity: 0, smoothLandmarks: true, refineFaceLandmarks: false });
    holistic.onResults(onResults);

    statusText.textContent = "Requesting camera...";
    camera = new Camera(videoEl, {
      onFrame: async () => { await holistic.send({ image: videoEl }); },
      width: 640, height: 480,
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
    return;
  }
  predicting = true;
  try {
    const { word, confidence } = await predictWord(sample);
    wordOutput.textContent = word;
    wordConf.textContent = `confidence ${(confidence * 100).toFixed(1)}%`;
  } catch (err) {
    wordOutput.textContent = "ERROR";
    wordConf.textContent = String(err && err.message ? err.message : err);
    console.error("predictWord failed:", err);
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
  liveDot.classList.remove("live");
  badgeText.textContent = "camera off";
  wordOutput.textContent = "—";
  wordConf.textContent = "confidence —";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusText.textContent = 'Click "Start camera" and allow access.';
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
window.addEventListener("resize", resizeCanvas);
