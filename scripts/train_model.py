"""
Train the ISL landmark classifier from data/landmarks.csv and produce:
    model/isl_model.h5     -- layer-naming export_weights.py expects
    public/scaler.json     -- StandardScaler mean/scale, read by public/js/model.js
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def validate_h5_against_exporter(h5_path: Path, export_script: Path):
    import h5py
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_weights", export_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with h5py.File(h5_path, "r") as f:
        missing = [key for key, path in mod.LAYER_MAP.items() if path not in f]

    if missing:
        raise RuntimeError(
            f"Saved model.h5 does NOT match export_weights.py's expected layout "
            f"(missing: {missing})."
        )
    print("h5 layout OK -- matches export_weights.py's LAYER_MAP.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/landmarks.csv")
    ap.add_argument("--model_out", default="model/isl_model.h5")
    ap.add_argument("--scaler_out", default="public/scaler.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--test_size", type=float, default=0.15)
    args = ap.parse_args()

    try:
        import tf_keras as keras
    except ImportError:
        sys.exit("tf_keras not installed. pip install tf_keras")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(args.csv)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype("float32")
    y_raw = df["label"].astype(str).values

    classes = sorted(np.unique(y_raw).tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[v] for v in y_raw], dtype="int64")

    print(f"{len(df)} samples, {len(classes)} classes: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    num_classes = len(classes)
    model = keras.Sequential([
        keras.layers.Input(shape=(126,)),
        keras.layers.Dense(256),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    model.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    print(f"Saved {model_out}")

    validate_h5_against_exporter(model_out, Path("scripts/export_weights.py"))

    scaler_out = Path(args.scaler_out)
    scaler_out.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_out, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)
    print(f"Saved {scaler_out}")

    js_labels = json.dumps(classes)
    print("\nNext steps:")
    print(f"  1. python scripts/export_weights.py {model_out} public/weights.json")
    print(f"  2. In public/js/model.js, set:\n     const LABELS = {js_labels};")


if __name__ == "__main__":
    main()
