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
    "dense_kernel": "model_weights/dense/dense/kernel:0",
    "dense_bias": "model_weights/dense/dense/bias:0",
    "bn1_gamma": "model_weights/batch_normalization/batch_normalization/gamma:0",
    "bn1_beta": "model_weights/batch_normalization/batch_normalization/beta:0",
    "bn1_mean": "model_weights/batch_normalization/batch_normalization/moving_mean:0",
    "bn1_var": "model_weights/batch_normalization/batch_normalization/moving_variance:0",
    "dense1_kernel": "model_weights/dense_1/dense_1/kernel:0",
    "dense1_bias": "model_weights/dense_1/dense_1/bias:0",
    "bn2_gamma": "model_weights/batch_normalization_1/batch_normalization_1/gamma:0",
    "bn2_beta": "model_weights/batch_normalization_1/batch_normalization_1/beta:0",
    "bn2_mean": "model_weights/batch_normalization_1/batch_normalization_1/moving_mean:0",
    "bn2_var": "model_weights/batch_normalization_1/batch_normalization_1/moving_variance:0",
    "dense2_kernel": "model_weights/dense_2/dense_2/kernel:0",
    "dense2_bias": "model_weights/dense_2/dense_2/bias:0",
    "dense3_kernel": "model_weights/dense_3/dense_3/kernel:0",
    "dense3_bias": "model_weights/dense_3/dense_3/bias:0",
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
