#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path
from collections import Counter


NAMES = {
    0: "motorcycle",
    1: "pedestrian",
    2: "traffic_light",
    3: "traffic_sign",
    4: "vehicle",
}

CRITICAL_CLASSES = {0, 1, 3}  # motorcycle, pedestrian, traffic_sign
TRAFFIC_LIGHT = 2
VEHICLE = 4


def read_classes(label_path):
    classes = set()

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:
            classes.add(int(float(parts[0])))
        except Exception:
            pass

    return classes


def find_image(label_path, image_dir):
    stem = label_path.stem

    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


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


def copy_pair(label_path, image_path, out_label_dir, out_image_dir):
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(label_path, out_label_dir / label_path.name)
    shutil.copy2(image_path, out_image_dir / image_path.name)


def copy_split_all(src_root, out_root, split):
    src_label_dir = src_root / "labels" / split
    src_image_dir = src_root / "images" / split

    if not src_label_dir.exists():
        return

    out_label_dir = out_root / "labels" / split
    out_image_dir = out_root / "images" / split

    copied = 0

    for label_path in src_label_dir.glob("*.txt"):
        image_path = find_image(label_path, src_image_dir)

        if image_path is None:
            continue

        copy_pair(label_path, image_path, out_label_dir, out_image_dir)
        copied += 1

    print(f"[COPY {split}] {copied} image")


def balance_train(src_root, out_root, vehicle_limit, light_limit, seed):
    random.seed(seed)

    src_label_dir = src_root / "labels" / "train"
    src_image_dir = src_root / "images" / "train"

    out_label_dir = out_root / "labels" / "train"
    out_image_dir = out_root / "images" / "train"

    critical = []
    vehicle_only = []
    light_only = []
    other = []

    for label_path in sorted(src_label_dir.glob("*.txt")):
        image_path = find_image(label_path, src_image_dir)

        if image_path is None:
            continue

        classes = read_classes(label_path)

        item = (label_path, image_path, classes)

        if classes & CRITICAL_CLASSES:
            critical.append(item)
        elif VEHICLE in classes:
            vehicle_only.append(item)
        elif TRAFFIC_LIGHT in classes:
            light_only.append(item)
        else:
            other.append(item)

    random.shuffle(vehicle_only)
    random.shuffle(light_only)

    selected = []
    selected.extend(critical)
    selected.extend(vehicle_only[:vehicle_limit])
    selected.extend(light_only[:light_limit])

    seen = set()
    final = []

    for item in selected:
        label_path = item[0]

        if label_path.name in seen:
            continue

        seen.add(label_path.name)
        final.append(item)

    for label_path, image_path, classes in final:
        copy_pair(label_path, image_path, out_label_dir, out_image_dir)

    print("==========================================")
    print("[TRAIN BALANCE]")
    print("critical kept:", len(critical))
    print("vehicle candidates:", len(vehicle_only))
    print("vehicle selected:", min(len(vehicle_only), vehicle_limit))
    print("traffic_light candidates:", len(light_only))
    print("traffic_light selected:", min(len(light_only), light_limit))
    print("other ignored:", len(other))
    print("final train images:", len(final))
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
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue

                cls = int(float(line.split()[0]))
                counts[NAMES[cls]] += 1
                total[NAMES[cls]] += 1

        print("SPLIT:", split)
        print("images:", len(files))
        print(dict(counts))
        print()

    print("TOTAL:", dict(total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vehicle-limit", type=int, default=2000)
    parser.add_argument("--light-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()

    if out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    write_data_yaml(out_root)

    balance_train(
        src_root=src_root,
        out_root=out_root,
        vehicle_limit=args.vehicle_limit,
        light_limit=args.light_limit,
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
