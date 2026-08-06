// Forward pass for the exported Keras MLP, using weights.json.
// Loaded as a global `Model` object; app.js calls Model.load() then Model.predict().

const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

let W = null;

function matVecAdd(input, kernel, bias) {
  const outDim = bias.length;
  const out = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = bias[o];
    for (let i = 0; i < input.length; i++) sum += input[i] * kernel[i][o];
    out[o] = sum;
  }
  return out;
}

function relu(v) {
  for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0;
  return v;
}

function batchNorm(v, gamma, beta, mean, varr, eps) {
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) {
    out[i] = gamma[i] * ((v[i] - mean[i]) / Math.sqrt(varr[i] + eps)) + beta[i];
  }
  return out;
}

function softmax(v) {
  const m = Math.max(...v);
  const ex = v.map((x) => Math.exp(x - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / s);
}

async function loadModel(url = "weights.json") {
  const res = await fetch(url);
  W = await res.json();
}

function predict(input126) {
  let x = matVecAdd(input126, W.dense_kernel, W.dense_bias);
  x = batchNorm(x, W.bn1_gamma, W.bn1_beta, W.bn1_mean, W.bn1_var, 0.001);
  x = relu(x);
  x = matVecAdd(x, W.dense1_kernel, W.dense1_bias);
  x = batchNorm(x, W.bn2_gamma, W.bn2_beta, W.bn2_mean, W.bn2_var, 0.001);
  x = relu(x);
  x = matVecAdd(x, W.dense2_kernel, W.dense2_bias);
  x = relu(x);
  x = matVecAdd(x, W.dense3_kernel, W.dense3_bias);
  return softmax(Array.from(x));
}

window.Model = { loadModel, predict, LABELS };
