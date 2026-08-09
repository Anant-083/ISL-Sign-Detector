"""
Convert the Kaggle ISL image dataset into a CSV of 126-dim MediaPipe hand-landmark
features, using the exact same feature convention as public/js/app.js:

    features[0:63]   = left hand  (21 landmarks x [x, y, z]), zeros if absent
    features[63:126] = right hand (21 landmarks x [x, y, z]), zeros if absent

Uses the MediaPipe Tasks API (HandLandmarker) since mp.solutions was removed
in recent mediapipe releases. Requires hand_landmarker.task in the working dir:
    wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

VALID_LABELS = set("0123456789") | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dirs(root: Path, max_depth: int = 4):
    by_label = defaultdict(list)
    root = root.resolve()

    def walk(d: Path, depth: int):
        if depth > max_depth:
            return
        try:
            children = list(d.iterdir())
        except PermissionError:
            return
        label = d.name.strip().upper()
        if label in VALID_LABELS:
            imgs = [p for p in children if p.is_file() and p.suffix.lower() in IMG_EXTS]
            if imgs:
                by_label[label].extend(imgs)
        for c in children:
            if c.is_dir():
                walk(c, depth + 1)

    walk(root, 0)
    return by_label


def normalize_hand(lm):
    """Translate to wrist-relative coords, scale so the farthest landmark
    from the wrist is at distance 1. MUST exactly match normalizeHand() in
    public/js/app.js -- this is what makes the model invariant to hand
    position/size in frame instead of memorizing absolute pixel coordinates."""
    wrist = lm[0]
    translated = [(pt.x - wrist.x, pt.y - wrist.y, pt.z - wrist.z) for pt in lm]
    max_dist = max((x * x + y * y + z * z) ** 0.5 for x, y, z in translated)
    scale = max_dist if max_dist > 1e-6 else 1e-6
    return [(x / scale, y / scale, z / scale) for x, y, z in translated]


def landmarks_to_features(hand_landmarks_list, handedness_list):
    feat = [0.0] * 126
    slots = {"Left": None, "Right": None}
    for lm, hd in zip(hand_landmarks_list, handedness_list):
        label = hd[0].category_name  # "Left" or "Right"
        slots[label] = lm

    def fill(lm, offset):
        if lm is None:
            return
        norm = normalize_hand(lm)
        for i, (x, y, z) in enumerate(norm):
            feat[offset + i * 3 + 0] = x
            feat[offset + i * 3 + 1] = y
            feat[offset + i * 3 + 2] = z

    fill(slots["Left"], 0)
    fill(slots["Right"], 63)
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="data/landmarks.csv")
    ap.add_argument("--min-detection-confidence", type=float, default=0.5)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--model", default="hand_landmarker.task")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        sys.exit(f"--data_dir does not exist: {root}")

    by_label = find_class_dirs(root)
    if not by_label:
        sys.exit("No class-named directories (0-9, A-Z) found under --data_dir.")

    print(f"Found {len(by_label)} classes:")
    for label in sorted(by_label):
        print(f"  {label}: {len(by_label[label])} images")

    if args.list:
        return

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not Path(args.model).exists():
        sys.exit(f"Model bundle not found: {args.model} (download hand_landmarker.task first)")

    base_options = mp_python.BaseOptions(model_asset_path=args.model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written, n_skipped = 0, 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(126)] + ["label"])

        for label in sorted(by_label):
            paths = by_label[label]
            if args.limit_per_class:
                paths = paths[: args.limit_per_class]
            for p in paths:
                img = cv2.imread(str(p))
                if img is None:
                    n_skipped += 1
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)
                if not result.hand_landmarks:
                    n_skipped += 1
                    continue
                feat = landmarks_to_features(result.hand_landmarks, result.handedness)
                writer.writerow(feat + [label])
                n_written += 1

        print(f"\nWrote {n_written} rows to {out_path} ({n_skipped} images skipped)")


if __name__ == "__main__":
    main()
