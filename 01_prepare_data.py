"""Step 1: images -> MediaPipe landmarks -> 33 features -> train/test CSVs.

Train set is augmented (gaussian landmark noise); test set is not.
"""
import csv
import os

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

import config as C
from features import (CSV_HEADER, augment_landmarks, extract_features,
                      landmarks_to_array, normalize_landmarks)

IMG_EXT = (".jpg", ".jpeg", ".png")


def label_from_filename(name):
    """P1_Alpadma_n_3.jpg -> 'Alpadma' (also fixes known typos)."""
    parts = os.path.splitext(name)[0].split("_")
    label = parts[1] if len(parts) >= 2 else parts[0]
    return C.LABEL_FIXES.get(label, label)


def collect_images():
    X, y = [], []
    for item in sorted(os.listdir(C.DATASET_PATH)):
        path = C.DATASET_PATH / item
        if path.is_dir():                       # class-per-folder layout
            for img in sorted(os.listdir(path)):
                if img.lower().endswith(IMG_EXT):
                    X.append(str(path / img))
                    y.append(C.LABEL_FIXES.get(item, item))
        elif item.lower().endswith(IMG_EXT):    # flat layout
            X.append(str(path))
            y.append(label_from_filename(item))
    return X, y


def detect_hand(hands, image):
    """The dataset images are small (200x200) and tightly cropped, so MediaPipe
    often misses the hand. Retry with upscaling + edge-replicate padding.
    Returns the landmark list of the first hand found, or None."""
    attempts = [(1, 0), (2, 200), (2, 100), (3, 300)]   # (scale, padding)
    for scale, pad in attempts:
        img = image
        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if pad:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            return res.multi_hand_landmarks[0]
    return None


def process_split(hands, paths, labels, csv_path, failed_path, augment, rng):
    stats = dict(total=0, read=0, detected=0, rows=0)
    failed = []
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for path, label in zip(paths, labels):
            stats["total"] += 1
            image = cv2.imread(path)
            if image is None:
                failed.append(path)
                continue
            stats["read"] += 1
            hand = detect_hand(hands, image)
            if hand is None:
                failed.append(path)
                continue
            stats["detected"] += 1
            norm = normalize_landmarks(landmarks_to_array(hand))
            if norm is None:
                failed.append(path)
                continue
            writer.writerow([label] + extract_features(norm))
            stats["rows"] += 1
            if augment:
                for _ in range(C.AUGMENTATION_COUNT):
                    aug = augment_landmarks(norm, C.NOISE_LEVEL, rng)
                    writer.writerow([label] + extract_features(aug))
                    stats["rows"] += 1
    with open(failed_path, "w") as fh:
        fh.write("\n".join(failed))
    return stats, failed


def main():
    print(f"Scanning dataset: {C.DATASET_PATH}")
    X, y = collect_images()
    print(f"Found {len(X)} images across {len(set(y))} classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y)
    print(f"Split: {len(X_train)} train / {len(X_test)} test images.")

    rng = np.random.default_rng(C.RANDOM_STATE)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=C.MP_DETECTION_CONFIDENCE)
    try:
        for name, paths, labels, csv_path, failed_path, aug in (
            ("TRAIN", X_train, y_train, C.TRAIN_CSV, C.FAILED_TRAIN_TXT, True),
            ("TEST", X_test, y_test, C.TEST_CSV, C.FAILED_TEST_TXT, False),
        ):
            print(f"\nProcessing {name} -> {csv_path.name}")
            s, failed = process_split(hands, paths, labels, csv_path, failed_path, aug, rng)
            print(f"  images: {s['total']}  read: {s['read']}  "
                  f"hand detected: {s['detected']}  rows written: {s['rows']}  "
                  f"failed: {len(failed)}")
    finally:
        hands.close()
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
