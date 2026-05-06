#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np


NAMES = {
    0: "motorcycle",
    1: "pedestrian",
    2: "traffic_light",
    3: "traffic_sign",
    4: "vehicle",
}

MOTORCYCLE = 0
PEDESTRIAN = 1
TRAFFIC_LIGHT = 2
TRAFFIC_SIGN = 3
VEHICLE = 4

CRITICAL_CLASSES = {MOTORCYCLE, PEDESTRIAN, TRAFFIC_SIGN}


def write_data_yaml(out_root):
    text = f"""path: {out_root.resolve()}

train: images/train
val: images/val
test: images/test

names:
  0: motorcycle
  1: pedestrian
  2: traffic_light
  3: traffic_sign
  4: vehicle
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def read_yolo_labels(label_path):
    labels = []

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        labels.append([cls, x, y, w, h])

    return labels


def write_yolo_labels(label_path, labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in labels:
            x = max(0.0, min(1.0, float(x)))
            y = max(0.0, min(1.0, float(y)))
            w = max(0.0, min(1.0, float(w)))
            h = max(0.0, min(1.0, float(h)))
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def find_image(label_path, image_dir):
    stem = label_path.stem

    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


def class_counts_from_labels(labels):
    c = Counter()

    for lab in labels:
        c[int(lab[0])] += 1

    return c


def count_dataset(root):
    total = Counter()

    for split in ["train", "val", "test"]:
        label_dir = root / "labels" / split

        if not label_dir.exists():
            continue

        counts = Counter()
        files = list(label_dir.glob("*.txt"))

        for f in files:
            labels = read_yolo_labels(f)

            for lab in labels:
                counts[NAMES[int(lab[0])]] += 1
                total[NAMES[int(lab[0])]] += 1

        print("SPLIT:", split)
        print("images:", len(files))
        print(dict(counts))
        print()

    print("TOTAL:", dict(total))


def copy_pair(label_path, image_path, out_label_dir, out_image_dir, new_stem=None):
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    if new_stem is None:
        out_label = out_label_dir / label_path.name
        out_image = out_image_dir / image_path.name
    else:
        out_label = out_label_dir / f"{new_stem}.txt"
        out_image = out_image_dir / f"{new_stem}{image_path.suffix.lower()}"

    shutil.copy2(label_path, out_label)
    shutil.copy2(image_path, out_image)


def copy_split_all(src_root, out_root, split):
    src_label_dir = src_root / "labels" / split
    src_image_dir = src_root / "images" / split

    if not src_label_dir.exists():
        return

    out_label_dir = out_root / "labels" / split
    out_image_dir = out_root / "images" / split

    copied = 0

    for label_path in sorted(src_label_dir.glob("*.txt")):
        image_path = find_image(label_path, src_image_dir)

        if image_path is None:
            continue

        copy_pair(label_path, image_path, out_label_dir, out_image_dir)
        copied += 1

    print(f"[COPY {split}] {copied} image")


def hflip_labels(labels):
    flipped = []

    for cls, x, y, w, h in labels:
        flipped.append([cls, 1.0 - x, y, w, h])

    return flipped


def augment_image(image, labels, aug_type, rng):
    if aug_type == "hflip":
        aug_img = cv2.flip(image, 1)
        aug_labels = hflip_labels(labels)
        return aug_img, aug_labels

    if aug_type == "bright":
        alpha = rng.uniform(0.85, 1.20)
        beta = rng.uniform(-18, 18)
        aug_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return aug_img, labels

    if aug_type == "blur_noise":
        aug_img = image.copy()

        if rng.random() < 0.50:
            aug_img = cv2.GaussianBlur(aug_img, (3, 3), 0)

        noise = rng.normal(0, 6, aug_img.shape).astype(np.int16)
        aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return aug_img, labels

    if aug_type == "gamma":
        gamma = rng.uniform(0.75, 1.35)
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        aug_img = cv2.LUT(image, table)
        return aug_img, labels

    return image, labels


def build_selected_train(src_root, vehicle_target, light_target, seed):
    rng = random.Random(seed)

    src_label_dir = src_root / "labels" / "train"
    src_image_dir = src_root / "images" / "train"

    critical_items = []
    vehicle_light_items = []
    other_items = []

    current_counts = Counter()

    for label_path in sorted(src_label_dir.glob("*.txt")):
        image_path = find_image(label_path, src_image_dir)

        if image_path is None:
            continue

        labels = read_yolo_labels(label_path)
        classes = {int(l[0]) for l in labels}
        counts = class_counts_from_labels(labels)

        item = {
            "label_path": label_path,
            "image_path": image_path,
            "labels": labels,
            "classes": classes,
            "counts": counts,
        }

        if classes & CRITICAL_CLASSES:
            critical_items.append(item)

            for cls, n in counts.items():
                current_counts[cls] += n

        elif VEHICLE in classes or TRAFFIC_LIGHT in classes:
            vehicle_light_items.append(item)

        else:
            other_items.append(item)

    rng.shuffle(vehicle_light_items)

    selected = list(critical_items)

    for item in vehicle_light_items:
        vehicle_now = current_counts[VEHICLE]
        light_now = current_counts[TRAFFIC_LIGHT]

        need_vehicle = vehicle_now < vehicle_target
        need_light = light_now < light_target

        item_has_vehicle = item["counts"][VEHICLE] > 0
        item_has_light = item["counts"][TRAFFIC_LIGHT] > 0

        if (need_vehicle and item_has_vehicle) or (need_light and item_has_light):
            selected.append(item)

            for cls, n in item["counts"].items():
                current_counts[cls] += n

        if current_counts[VEHICLE] >= vehicle_target and current_counts[TRAFFIC_LIGHT] >= light_target:
            break

    print("==========================================")
    print("[TRAIN SELECTION]")
    print("critical images kept:", len(critical_items))
    print("vehicle/light candidate images:", len(vehicle_light_items))
    print("other images ignored:", len(other_items))
    print("selected train images before aug:", len(selected))
    print("counts before aug:", {NAMES[k]: v for k, v in current_counts.items()})
    print("NOTE: vehicle/light 5000 hedefi, critical görüntülerdeki ek etiketlerden dolayı aşılabilir.")
    print("==========================================")

    return selected


def create_augmented_train(selected_items, out_root, aug_per_image, seed):
    rng = np.random.default_rng(seed)

    out_label_dir = out_root / "labels" / "train"
    out_image_dir = out_root / "images" / "train"

    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    aug_types = ["hflip", "bright", "blur_noise", "gamma"]

    copied = 0
    augmented = 0

    for idx, item in enumerate(selected_items):
        image_path = item["image_path"]
        label_path = item["label_path"]
        labels = item["labels"]
        classes = item["classes"]

        new_stem = f"base_train_{idx:07d}"
        copy_pair(label_path, image_path, out_label_dir, out_image_dir, new_stem=new_stem)
        copied += 1

        should_aug = bool(classes & {MOTORCYCLE, PEDESTRIAN})

        if not should_aug:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        for aug_idx in range(aug_per_image):
            aug_type = aug_types[aug_idx % len(aug_types)]

            aug_img, aug_labels = augment_image(
                image=image,
                labels=[x[:] for x in labels],
                aug_type=aug_type,
                rng=rng,
            )

            aug_stem = f"aug_{aug_type}_{idx:07d}_{aug_idx:02d}"
            aug_img_path = out_image_dir / f"{aug_stem}.jpg"
            aug_lab_path = out_label_dir / f"{aug_stem}.txt"

            cv2.imwrite(str(aug_img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo_labels(aug_lab_path, aug_labels)
            augmented += 1

    print("==========================================")
    print("[AUGMENTATION]")
    print("base copied:", copied)
    print("augmented images:", augmented)
    print("aug_per_person_motor_image:", aug_per_image)
    print("==========================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vehicle-target", type=int, default=5000)
    parser.add_argument("--light-target", type=int, default=5000)
    parser.add_argument("--aug-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()

    if out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_root)

    selected_items = build_selected_train(
        src_root=src_root,
        vehicle_target=args.vehicle_target,
        light_target=args.light_target,
        seed=args.seed,
    )

    create_augmented_train(
        selected_items=selected_items,
        out_root=out_root,
        aug_per_image=args.aug_per_image,
        seed=args.seed,
    )

    copy_split_all(src_root, out_root, "val")
    copy_split_all(src_root, out_root, "test")

    print("==========================================")
    print("[FINAL COUNTS]")
    count_dataset(out_root)
    print("DATA YAML:", out_root / "data.yaml")
    print("==========================================")


if __name__ == "__main__":
    main()
