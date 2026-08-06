"""
Export a Keras .h5 model's weights to a flat JSON file the browser can load
directly, no TensorFlow.js conversion needed.

Usage:
    python export_weights.py model/isl_model.h5 public/weights.json
"""
import sys
import json
import h5py

LAYER_MAP = {
    "dense_kernel": "model_weights/dense/sequential/dense/kernel",
    "dense_bias": "model_weights/dense/sequential/dense/bias",
    "bn1_gamma": "model_weights/batch_normalization/sequential/batch_normalization/gamma",
    "bn1_beta": "model_weights/batch_normalization/sequential/batch_normalization/beta",
    "bn1_mean": "model_weights/batch_normalization/sequential/batch_normalization/moving_mean",
    "bn1_var": "model_weights/batch_normalization/sequential/batch_normalization/moving_variance",
    "dense1_kernel": "model_weights/dense_1/sequential/dense_1/kernel",
    "dense1_bias": "model_weights/dense_1/sequential/dense_1/bias",
    "bn2_gamma": "model_weights/batch_normalization_1/sequential/batch_normalization_1/gamma",
    "bn2_beta": "model_weights/batch_normalization_1/sequential/batch_normalization_1/beta",
    "bn2_mean": "model_weights/batch_normalization_1/sequential/batch_normalization_1/moving_mean",
    "bn2_var": "model_weights/batch_normalization_1/sequential/batch_normalization_1/moving_variance",
    "dense2_kernel": "model_weights/dense_2/sequential/dense_2/kernel",
    "dense2_bias": "model_weights/dense_2/sequential/dense_2/bias",
    "dense3_kernel": "model_weights/dense_3/sequential/dense_3/kernel",
    "dense3_bias": "model_weights/dense_3/sequential/dense_3/bias",
}

def main():
    if len(sys.argv) != 3:
        print("Usage: python export_weights.py <input.h5> <output.json>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    with h5py.File(src, "r") as f:
        weights = {key: f[path][:].tolist() for key, path in LAYER_MAP.items()}

    with open(dst, "w") as out:
        json.dump(weights, out)

    print(f"Wrote {dst}")

if __name__ == "__main__":
    main()
