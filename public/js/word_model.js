let ortSession = null;

ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.simd = true;
async function loadWordModel() {
  ortSession = await ort.InferenceSession.create('models/sign_model.onnx', {
    executionProviders: ['wasm'],
    externalData: [
      {
        path: 'sign_model.onnx.data',
        data: 'models/sign_model.onnx.data',
      },
    ],
  });
  console.log('Word model loaded. Input names:', ortSession.inputNames, 'Output names:', ortSession.outputNames);
}

async function predictWord(sample) {
  const { frames, length } = sample;
  const MAX_LEN = 160;
  const FEAT_DIM = 228;

  const data = new Float32Array(MAX_LEN * FEAT_DIM);
  const maskData = new Uint8Array(MAX_LEN);

  for (let t = 0; t < length; t++) {
    for (let p = 0; p < FEAT_DIM; p++) {
      data[t * FEAT_DIM + p] = frames[t][p];
    }
    maskData[t] = 1;
  }

  const inputTensor = new ort.Tensor('float32', data, [1, MAX_LEN, FEAT_DIM]);
  const maskTensor = new ort.Tensor('bool', maskData, [1, MAX_LEN]);

  const results = await ortSession.run({ input: inputTensor, mask: maskTensor });
  const outputName = ortSession.outputNames[0]; // 'logits'
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
