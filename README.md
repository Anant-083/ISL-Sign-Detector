# ISL Alphabet Recognizer

Real-time Indian Sign Language (A–Z) recognition, running entirely in the browser.
MediaPipe Hands extracts 21 landmarks per hand; a small MLP (trained in Keras,
exported to JSON) classifies the pose. No backend, no upload — inference runs
client-side.

## Architecture

Input(126) → Dense(256) → BatchNorm → ReLU → Dropout
           → Dense(128) → BatchNorm → ReLU → Dropout
           → Dense(64)  → ReLU → Dropout
           → Dense(26, softmax)

Input is 2 hands × 21 landmarks × (x, y, z) = 126 features, left hand first,
zero-padded if a hand is absent. Output is A–Z.

## Run locally

```bash
cd public
python3 -m http.server 8000
# open http://localhost:8000
```

Camera access requires a secure context (localhost or https).

## Re-export weights after retraining

```bash
pip install h5py
python3 scripts/export_weights.py model/isl_model.h5 public/weights.json
```

## Preprocessing assumptions

These must match your training pipeline exactly or predictions will be wrong:

- Landmark order: left hand (indices 0–62), then right hand (63–125)
- Missing hand → zeros, not omitted
- Class order: A–Z alphabetical, index 0–25

If your training notebook used a different convention, edit the mapping in
`public/js/app.js` (`landmarksToFeatures`) and `public/js/model.js` (`LABELS`).

## Deploy

Static site — works on GitHub Pages, Netlify, Vercel, etc. Just serve `public/`.
