const SEQ_LEN = 123;
const FEATURE_DIM = 225; // 75 points * 3 (x,y,z)

let frameWindow = [];
let missedFrameStreak = 0;
const MAX_MISSED_STREAK = 15;

function extractRawLandmarks(results) {
  const pose = [];
  for (let i = 0; i < 33; i++) {
    const lm = results.poseLandmarks?.[i];
    pose.push(lm ? [lm.x, lm.y, lm.z || 0] : [0, 0, 0]);
  }
  const handA = [];
  for (let i = 0; i < 21; i++) {
    const lm = results.leftHandLandmarks?.[i];
    handA.push(lm ? [lm.x, lm.y, lm.z || 0] : [0, 0, 0]);
  }
  const handB = [];
  for (let i = 0; i < 21; i++) {
    const lm = results.rightHandLandmarks?.[i];
    handB.push(lm ? [lm.x, lm.y, lm.z || 0] : [0, 0, 0]);
  }
  return { pose, handA, handB };
}

function normalizeFrame({ pose, handA, handB }) {
  function normHand(hand) {
    const wrist = hand[0];
    const rel = hand.map(p => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]);
    let maxDist = 0;
    rel.forEach(p => {
      const d = Math.hypot(p[0], p[1], p[2]);
      if (d > maxDist) maxDist = d;
    });
    const scale = maxDist === 0 ? 1 : maxDist;
    return rel.map(p => [p[0] / scale, p[1] / scale, p[2] / scale]);
  }

  const hip23 = pose[23], hip24 = pose[24];
  const hipCenter = [(hip23[0] + hip24[0]) / 2, (hip23[1] + hip24[1]) / 2, (hip23[2] + hip24[2]) / 2];
  const poseRel = pose.map(p => [p[0] - hipCenter[0], p[1] - hipCenter[1], p[2] - hipCenter[2]]);

  const handANorm = normHand(handA);
  const handBNorm = normHand(handB);

  return [...poseRel, ...handANorm, ...handBNorm]; // 75 points, x/y/z each
}

function pushFrame(results) {
  const hasHand = !!(results.leftHandLandmarks || results.rightHandLandmarks);

  if (!hasHand) {
    missedFrameStreak++;
    if (missedFrameStreak > MAX_MISSED_STREAK) {
      frameWindow = [];
      missedFrameStreak = 0;
      return;
    }
    if (frameWindow.length > 0) frameWindow.push(frameWindow[frameWindow.length - 1]);
    return;
  }

  missedFrameStreak = 0;
  const raw = extractRawLandmarks(results);
  const normalized = normalizeFrame(raw);
  frameWindow.push(normalized);
  if (frameWindow.length > SEQ_LEN * 2) frameWindow.shift();
}

function getNormalizedSample() {
  if (frameWindow.length < 15) return null;
  const length = Math.min(frameWindow.length, SEQ_LEN);
  const padded = new Array(SEQ_LEN).fill(null).map(() => new Array(75).fill([0, 0, 0]));
  for (let t = 0; t < length; t++) padded[t] = frameWindow[t];
  return { frames: padded, length };
}
