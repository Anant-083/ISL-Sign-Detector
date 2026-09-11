const SEQ_LEN = 160;
const FEATURE_DIM = 228;

let frameWindow = [];
let missedFrameStreak = 0;
const MAX_MISSED_STREAK = 15;

function extractRawLandmarks(results) {
  const flat = [];

  for (let i = 0; i < 33; i++) {
    const lm = results.poseLandmarks?.[i];
    flat.push(lm ? lm.x : 0, lm ? lm.y : 0, lm ? (lm.z || 0) : 0);
  }
  for (let i = 0; i < 21; i++) {
    const lm = results.leftHandLandmarks?.[i];
    flat.push(lm ? lm.x : 0, lm ? lm.y : 0, lm ? (lm.z || 0) : 0);
  }
  for (let i = 0; i < 21; i++) {
    const lm = results.rightHandLandmarks?.[i];
    flat.push(lm ? lm.x : 0, lm ? lm.y : 0, lm ? (lm.z || 0) : 0);
  }

  // matches the constant tail observed in training data
  flat.push(1, 1, 1);

  return flat; // length 228
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
  const flat = extractRawLandmarks(results);
  frameWindow.push(flat);
  if (frameWindow.length > SEQ_LEN * 2) frameWindow.shift();
}

function getNormalizedSample() {
  if (frameWindow.length < 15) return null;
  const length = Math.min(frameWindow.length, SEQ_LEN);
  const recent = frameWindow.slice(-length);
  const padded = new Array(SEQ_LEN).fill(null).map(() => new Array(FEATURE_DIM).fill(0));
  for (let t = 0; t < length; t++) padded[t] = recent[t];
  return { frames: padded, length };
}

function resetFrameWindow() {
  frameWindow = [];
  missedFrameStreak = 0;
}
