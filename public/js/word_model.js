let ortSession = null;

ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.simd = true;

async function loadWordModel() {
  ortSession = await ort.InferenceSession.create('models/include50.onnx', {
    executionProviders: ['wasm'],
  });
  console.log('Word model loaded. Input names:', ortSession.inputNames, 'Output names:', ortSession.outputNames);
}

async function predictWord(sample) {
  const { frames, length } = sample;
  const data = new Float32Array(123 * 225);
  for (let t = 0; t < 123; t++) {
    for (let p = 0; p < 75; p++) {
      data[t * 225 + p * 3 + 0] = frames[t][p][0];
      data[t * 225 + p * 3 + 1] = frames[t][p][1];
      data[t * 225 + p * 3 + 2] = frames[t][p][2];
    }
  }

  const inputTensor = new ort.Tensor('float32', data, [1, 123, 225]);
  const lengthTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(length)]), [1]);

  const results = await ortSession.run({ input: inputTensor, length: lengthTensor });
  const outputName = ortSession.outputNames[0];
  const logits = Array.from(results[outputName].data);

  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExp);

  const ranked = probs
    .map((p, i) => ({ word: WORD_LABELS[i], confidence: p }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  return { top: ranked, word: ranked[0].word, confidence: ranked[0].confidence };
}
