const POSE_INDICES = [0, 2, 5, 11, 12, 13, 14];
const HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];
const SEQ_LEN = 120;
const MIN_FRAMES = 8;
const NO_HAND_CONFIRM = 3;

let frameWindow = [];
let missedFrameStreak = 0;
let signComplete = false;

function extract27Points(results) {
  const points = [];
  POSE_INDICES.forEach(i => {
    const lm = results.poseLandmarks?.[i];
    points.push(lm ? [lm.x, lm.y] : [0, 0]);
  });
  HAND_INDICES.forEach(i => {
    const lm = results.leftHandLandmarks?.[i];
    points.push(lm ? [lm.x, lm.y] : [0, 0]);
  });
  HAND_INDICES.forEach(i => {
    const lm = results.rightHandLandmarks?.[i];
    points.push(lm ? [lm.x, lm.y] : [0, 0]);
  });
  return points;
}

function pushFrame(results) {
  const hasHand = !!(results.leftHandLandmarks || results.rightHandLandmarks);

  if (hasHand) {
    missedFrameStreak = 0;
    signComplete = false;
    const raw = extract27Points(results);
    frameWindow.push(raw);
    if (frameWindow.length > SEQ_LEN * 2) frameWindow.shift();
    return;
  }

  missedFrameStreak++;

  if (missedFrameStreak >= NO_HAND_CONFIRM && frameWindow.length >= MIN_FRAMES && !signComplete) {
    signComplete = true;
  }

  if (missedFrameStreak > 20) {
    frameWindow = [];
    signComplete = false;
  }
}

function uniformSample(frames) {
  const out = [];
  for (let i = 0; i < SEQ_LEN; i++) {
    const idx = Math.min(frames.length - 1, Math.floor(i * frames.length / SEQ_LEN));
    out.push(frames[idx]);
  }
  return out;
}

function centerAndScaleClip(frames) {
  const LS = 3, RS = 4;
  let sumCx = 0, sumCy = 0, sumDist = 0;
  frames.forEach(f => {
    const [lx, ly] = f[LS];
    const [rx, ry] = f[RS];
    sumCx += (lx + rx) / 2;
    sumCy += (ly + ry) / 2;
    sumDist += Math.hypot(lx - rx, ly - ry);
  });
  const n = frames.length;
  const cx = sumCx / n, cy = sumCy / n;
  const meanDist = sumDist / n;
  const scale = meanDist > 1e-6 ? 1 / meanDist : 1;
  return frames.map(f => f.map(([x, y]) => [(x - cx) * scale, (y - cy) * scale]));
}

function getNormalizedSample() {
  if (frameWindow.length < MIN_FRAMES) return null;
  const sampled = uniformSample(frameWindow);
  return centerAndScaleClip(sampled);
}

function consumeSignComplete() {
  if (!signComplete) return false;
  signComplete = false;
  return true;
}

function resetBuffer() {
  frameWindow = [];
  signComplete = false;
  missedFrameStreak = 0;
}
