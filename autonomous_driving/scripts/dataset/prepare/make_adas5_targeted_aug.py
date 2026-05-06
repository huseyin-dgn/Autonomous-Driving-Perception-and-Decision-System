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


def read_labels(label_path):
    labels = []

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue

        p = line.strip().split()

        if len(p) < 5:
            continue

        labels.append([
            int(float(p[0])),
            float(p[1]),
            float(p[2]),
            float(p[3]),
            float(p[4]),
        ])

    return labels


def write_labels(label_path, labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in labels:
            if w <= 0 or h <= 0:
                continue

            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            if w <= 0.001 or h <= 0.001:
                continue

            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def find_image(label_path, image_dir):
    stem = label_path.stem

    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


def count_labels(labels):
    c = Counter()
    for lab in labels:
        c[int(lab[0])] += 1
    return c


def copy_pair(label_path, image_path, out_label_dir, out_image_dir, new_stem=None):
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    if new_stem is None:
        shutil.copy2(label_path, out_label_dir / label_path.name)
        shutil.copy2(image_path, out_image_dir / image_path.name)
    else:
        shutil.copy2(label_path, out_label_dir / f"{new_stem}.txt")
        shutil.copy2(image_path, out_image_dir / f"{new_stem}{image_path.suffix.lower()}")


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


def yolo_to_xyxy(label, img_w, img_h):
    cls, cx, cy, bw, bh = label

    x1 = (cx - bw / 2.0) * img_w
    y1 = (cy - bh / 2.0) * img_h
    x2 = (cx + bw / 2.0) * img_w
    y2 = (cy + bh / 2.0) * img_h

    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, crop_w, crop_h):
    x1 = max(0.0, min(float(crop_w - 1), x1))
    y1 = max(0.0, min(float(crop_h - 1), y1))
    x2 = max(0.0, min(float(crop_w - 1), x2))
    y2 = max(0.0, min(float(crop_h - 1), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    if bw < 4 or bh < 4:
        return None

    cx = (x1 + x2) / 2.0 / crop_w
    cy = (y1 + y2) / 2.0 / crop_h
    nw = bw / crop_w
    nh = bh / crop_h

    return [cls, cx, cy, nw, nh]


def make_crop_box(target_label, img_w, img_h, rng):
    cls, x1, y1, x2, y2 = yolo_to_xyxy(target_label, img_w, img_h)

    bw = x2 - x1
    bh = y2 - y1

    scale = rng.uniform(2.0, 3.2)

    crop_w = max(160, int(bw * scale))
    crop_h = max(160, int(bh * scale))

    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    cx = (x1 + x2) / 2.0 + rng.uniform(-0.25, 0.25) * bw
    cy = (y1 + y2) / 2.0 + rng.uniform(-0.25, 0.25) * bh

    cx1 = int(cx - crop_w / 2)
    cy1 = int(cy - crop_h / 2)
    cx2 = cx1 + crop_w
    cy2 = cy1 + crop_h

    if cx1 < 0:
        cx2 -= cx1
        cx1 = 0

    if cy1 < 0:
        cy2 -= cy1
        cy1 = 0

    if cx2 > img_w:
        shift = cx2 - img_w
        cx1 -= shift
        cx2 = img_w

    if cy2 > img_h:
        shift = cy2 - img_h
        cy1 -= shift
        cy2 = img_h

    cx1 = max(0, cx1)
    cy1 = max(0, cy1)
    cx2 = min(img_w, cx2)
    cy2 = min(img_h, cy2)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    return cx1, cy1, cx2, cy2


def labels_in_crop(labels, crop_box, img_w, img_h):
    cx1, cy1, cx2, cy2 = crop_box
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    new_labels = []

    for lab in labels:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(lab, img_w, img_h)

        ix1 = max(x1, cx1)
        iy1 = max(y1, cy1)
        ix2 = min(x2, cx2)
        iy2 = min(y2, cy2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        old_area = max(1.0, (x2 - x1) * (y2 - y1))
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        keep_ratio = inter_area / old_area

        if keep_ratio < 0.45:
            continue

        nx1 = ix1 - cx1
        ny1 = iy1 - cy1
        nx2 = ix2 - cx1
        ny2 = iy2 - cy1

        yolo = xyxy_to_yolo(cls, nx1, ny1, nx2, ny2, crop_w, crop_h)

        if yolo is not None:
            new_labels.append(yolo)

    return new_labels


def hflip_labels(labels):
    out = []
    for cls, x, y, w, h in labels:
        out.append([cls, 1.0 - x, y, w, h])
    return out


def augment_crop(crop, labels, aug_id, rng):
    img = crop.copy()
    labs = [x[:] for x in labels]

    if aug_id % 4 == 0:
        img = cv2.flip(img, 1)
        labs = hflip_labels(labs)

    elif aug_id % 4 == 1:
        alpha = rng.uniform(0.85, 1.20)
        beta = rng.uniform(-18, 18)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    elif aug_id % 4 == 2:
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        noise = rng.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    else:
        gamma = rng.uniform(0.75, 1.35)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)

    return img, labs


def build_base_train(src_root, out_root, vehicle_target, light_target, seed):
    rng = random.Random(seed)

    src_label_dir = src_root / "labels" / "train"
    src_image_dir = src_root / "images" / "train"

    out_label_dir = out_root / "labels" / "train"
    out_image_dir = out_root / "images" / "train"

    items = []

    for label_path in sorted(src_label_dir.glob("*.txt")):
        image_path = find_image(label_path, src_image_dir)
        if image_path is None:
            continue

        labels = read_labels(label_path)
        counts = count_labels(labels)

        score = (
            counts[MOTORCYCLE] * 8
            + counts[PEDESTRIAN] * 8
            + counts[TRAFFIC_SIGN] * 4
            - counts[VEHICLE] * 0.8
            - counts[TRAFFIC_LIGHT] * 0.8
        )

        items.append({
            "label_path": label_path,
            "image_path": image_path,
            "labels": labels,
            "counts": counts,
            "score": score,
        })

    items.sort(key=lambda x: x["score"], reverse=True)

    total = Counter()
    selected = []

    for item in items:
        c = item["counts"]

        has_critical = (
            c[MOTORCYCLE] > 0
            or c[PEDESTRIAN] > 0
            or c[TRAFFIC_SIGN] > 0
        )

        next_vehicle = total[VEHICLE] + c[VEHICLE]
        next_light = total[TRAFFIC_LIGHT] + c[TRAFFIC_LIGHT]

        if has_critical:
            if next_vehicle <= vehicle_target and next_light <= light_target:
                selected.append(item)
                total.update(c)
            continue

        if c[VEHICLE] > 0 or c[TRAFFIC_LIGHT] > 0:
            if next_vehicle <= vehicle_target and next_light <= light_target:
                selected.append(item)
                total.update(c)

    for idx, item in enumerate(selected):
        copy_pair(
            item["label_path"],
            item["image_path"],
            out_label_dir,
            out_image_dir,
            new_stem=f"base_{idx:07d}",
        )

    print("==========================================")
    print("[BASE TRAIN]")
    print("selected images:", len(selected))
    print("base counts:", {NAMES[k]: v for k, v in total.items()})
    print("==========================================")

    return selected


def collect_target_objects(src_root):
    src_label_dir = src_root / "labels" / "train"
    src_image_dir = src_root / "images" / "train"

    objects = []

    for label_path in sorted(src_label_dir.glob("*.txt")):
        image_path = find_image(label_path, src_image_dir)
        if image_path is None:
            continue

        labels = read_labels(label_path)

        for idx, lab in enumerate(labels):
            cls = int(lab[0])

            if cls not in [MOTORCYCLE, PEDESTRIAN]:
                continue

            objects.append({
                "image_path": image_path,
                "label_path": label_path,
                "labels": labels,
                "target_label": lab,
                "target_cls": cls,
                "object_idx": idx,
            })

    return objects


def create_crop_aug(src_root, out_root, motor_target, ped_target, seed):
    rng = np.random.default_rng(seed)
    random_rng = random.Random(seed)

    out_label_dir = out_root / "labels" / "train"
    out_image_dir = out_root / "images" / "train"

    objects = collect_target_objects(src_root)

    random_rng.shuffle(objects)

    existing_counts = Counter()

    for label_path in out_label_dir.glob("*.txt"):
        labels = read_labels(label_path)
        for lab in labels:
            existing_counts[int(lab[0])] += 1

    target_map = {
        MOTORCYCLE: motor_target,
        PEDESTRIAN: ped_target,
    }

    created = 0

    for obj in objects:
        cls = obj["target_cls"]

        if existing_counts[cls] >= target_map[cls]:
            if existing_counts[MOTORCYCLE] >= motor_target and existing_counts[PEDESTRIAN] >= ped_target:
                break
            continue

        image = cv2.imread(str(obj["image_path"]))

        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        crop_box = make_crop_box(obj["target_label"], img_w, img_h, rng)

        if crop_box is None:
            continue

        cx1, cy1, cx2, cy2 = crop_box
        crop = image[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        crop_labels = labels_in_crop(obj["labels"], crop_box, img_w, img_h)

        if not any(int(l[0]) == cls for l in crop_labels):
            continue

        aug_img, aug_labels = augment_crop(crop, crop_labels, created, rng)

        stem = f"crop_aug_{created:07d}_{NAMES[cls]}"
        img_path = out_image_dir / f"{stem}.jpg"
        lab_path = out_label_dir / f"{stem}.txt"

        cv2.imwrite(str(img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_labels(lab_path, aug_labels)

        for lab in aug_labels:
            existing_counts[int(lab[0])] += 1

        created += 1

    print("==========================================")
    print("[CROP AUG]")
    print("created crop images:", created)
    print("counts after crop:", {NAMES[k]: v for k, v in existing_counts.items()})
    print("==========================================")


def count_dataset(root):
    total = Counter()

    for split in ["train", "val", "test"]:
        label_dir = root / "labels" / split

        if not label_dir.exists():
            continue

        counts = Counter()
        files = list(label_dir.glob("*.txt"))

        for f in files:
            labels = read_labels(f)
            for lab in labels:
                counts[NAMES[int(lab[0])]] += 1
                total[NAMES[int(lab[0])]] += 1

        print("SPLIT:", split)
        print("images:", len(files))
        print(dict(counts))
        print()

    print("TOTAL:", dict(total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vehicle-target", type=int, default=5000)
    parser.add_argument("--light-target", type=int, default=5000)
    parser.add_argument("--motor-target", type=int, default=7000)
    parser.add_argument("--ped-target", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()

    if out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    write_data_yaml(out_root)

    build_base_train(
        src_root=src_root,
        out_root=out_root,
        vehicle_target=args.vehicle_target,
        light_target=args.light_target,
        seed=args.seed,
    )

    create_crop_aug(
        src_root=src_root,
        out_root=out_root,
        motor_target=args.motor_target,
        ped_target=args.ped_target,
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
