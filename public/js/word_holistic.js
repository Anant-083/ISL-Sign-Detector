const POSE_INDICES = [0, 2, 5, 11, 12, 13, 14];
const HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];
const SEQ_LEN = 120;

let frameWindow = [];

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
  if (!hasHand) {
    frameWindow = [];
    return;
  }
  const raw = extract27Points(results);
  frameWindow.push(raw);
  if (frameWindow.length > SEQ_LEN * 2) frameWindow.shift();
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
  if (frameWindow.length < SEQ_LEN) return null;
  const sampled = uniformSample(frameWindow);
  return centerAndScaleClip(sampled);
}
