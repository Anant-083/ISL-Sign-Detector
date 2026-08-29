const videoEl = document.getElementById("video");
const wordOutput = document.getElementById("wordOutput");
const wordConf = document.getElementById("wordConf");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");

let camera = null;
let holistic = null;
let predicting = false;
let loopHandle = null;

function onResults(results) {
  pushFrame(results);
}

async function startCamera() {
  startBtn.disabled = true;
  statusText.textContent = "Loading word model...";
  await loadWordModel();

  holistic = new Holistic({
    locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
  });
  holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, refineFaceLandmarks: false });
  holistic.onResults(onResults);

  camera = new Camera(videoEl, {
    onFrame: async () => { await holistic.send({ image: videoEl }); },
    width: 640, height: 480,
  });
  await camera.start();

  statusText.textContent = "Camera running. Perform a sign.";
  stopBtn.disabled = false;
  loopHandle = setInterval(predictLoop, 700);
}

async function predictLoop() {
  if (predicting) return;
  const sample = getNormalizedSample();
  if (!sample) return;
  predicting = true;
  const { word, confidence } = await predictWord(sample);
  wordOutput.textContent = word;
  wordConf.textContent = `confidence ${(confidence).toFixed(2)}`;
  predicting = false;
}

function stopCamera() {
  if (camera) { camera.stop(); camera = null; }
  const stream = videoEl.srcObject;
  if (stream) stream.getTracks().forEach(t => t.stop());
  videoEl.srcObject = null;
  clearInterval(loopHandle);
  frameWindow = [];
  wordOutput.textContent = "—";
  wordConf.textContent = "confidence —";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusText.textContent = 'Click "Start camera" and allow access.';
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
