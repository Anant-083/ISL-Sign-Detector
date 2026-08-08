// Forward pass for the exported Keras MLP, using weights.json.
// Loaded as a global `Model` object; app.js calls Model.load() then Model.predict().

// Order must exactly match the sorted class order used in scripts/train_model.py
// (ASCII sort of folder names: '0'-'9' then 'A'-'Z').
const LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

let W = null;
let SCALER = null;

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

async function loadModel(url = "weights.json", scalerUrl = "scaler.json") {
  const [res, scalerRes] = await Promise.all([fetch(url), fetch(scalerUrl)]);
  W = await res.json();
  SCALER = await scalerRes.json();
}

function scaleInput(input126) {
  const out = new Float32Array(126);
  for (let i = 0; i < 126; i++) {
    out[i] = (input126[i] - SCALER.mean[i]) / SCALER.scale[i];
  }
  return out;
}

function predict(input126) {
  const scaled = scaleInput(input126);
  let x = matVecAdd(scaled, W.dense_kernel, W.dense_bias);
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
