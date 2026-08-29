let ortSession = null;

async function loadWordModel() {
  ortSession = await ort.InferenceSession.create('models/isl_word_bert.onnx', {
    executionProviders: ['wasm'],
  });
  console.log('Word model loaded');
}

async function predictWord(frameBuffer) {
  const data = new Float32Array(2 * 120 * 27);
  for (let t = 0; t < 120; t++) {
    for (let p = 0; p < 27; p++) {
      data[t * 27 + p] = frameBuffer[t][p][0];
      data[120 * 27 + t * 27 + p] = frameBuffer[t][p][1];
    }
  }
  const tensor = new ort.Tensor('float32', data, [2, 120, 27]);
  const results = await ortSession.run({ frames: tensor });
  const logits = results.logits.data;
  let maxIdx = 0, maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
  }
  return { word: WORD_LABELS[maxIdx], confidence: maxVal };
}
