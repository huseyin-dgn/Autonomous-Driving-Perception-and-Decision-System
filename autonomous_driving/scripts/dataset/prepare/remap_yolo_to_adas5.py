#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
from collections import Counter
import yaml


TARGET_NAMES = {
    0: "motorcycle",
    1: "pedestrian",
    2: "traffic_light",
    3: "traffic_sign",
    4: "vehicle",
}

TARGET_ID = {v: k for k, v in TARGET_NAMES.items()}


def norm(x):
    return str(x).lower().strip().replace("-", "_").replace(" ", "_")


def map_name_to_target(name):
    n = norm(name)

    if n in ["motorbike", "motorcycle", "motor", "moto", "motobike"]:
        return 0

    if n in ["pedestrian", "person", "padestrian", "walker", "human", "people"]:
        return 1

    if n in [
        "traffic_light",
        "trafficlight",
        "tl",
        "tl_red",
        "tl_green",
        "tl_amber",
        "tl_yellow",
        "tl_red_amber",
        "red_light",
        "green_light",
        "yellow_light",
    ]:
        return 2

    if n in ["traffic_sign", "trafficsign", "sign", "road_sign", "stop_sign", "speed_sign"]:
        return 3

    if n in ["vehicle", "car", "cars", "truck", "bus", "van", "suv", "pickup", "automobile"]:
        return 4

    # Kritik: bike/bicycle kesinlikle motorcycle'a çevrilmiyor.
    if n in ["bike", "bicycle", "cycle", "rider"]:
        return None

    return None


def read_names(src_root):
    yamls = list(src_root.rglob("data.yaml")) + list(src_root.rglob("*.yaml")) + list(src_root.rglob("*.yml"))

    for y in yamls:
        try:
            data = yaml.safe_load(y.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        if not isinstance(data, dict) or "names" not in data:
            continue

        names = data["names"]

        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}, y

        if isinstance(names, list):
            return {i: v for i, v in enumerate(names)}, y

    raise RuntimeError(f"data.yaml / names bulunamadı: {src_root}")


def find_split_dirs(src_root, split):
    aliases = {
        "train": ["train", "training"],
        "val": ["valid", "val", "validation"],
        "test": ["test", "testing"],
    }

    image_dirs = []
    label_dirs = []

    for alias in aliases[split]:
        candidates_img = [
            src_root / alias / "images",
            src_root / "images" / alias,
            src_root / alias,
        ]

        candidates_lab = [
            src_root / alias / "labels",
            src_root / "labels" / alias,
            src_root / alias,
        ]

        for p in candidates_img:
            if p.exists():
                if list(p.rglob("*.jpg")) or list(p.rglob("*.jpeg")) or list(p.rglob("*.png")):
                    image_dirs.append(p)

        for p in candidates_lab:
            if p.exists():
                if list(p.rglob("*.txt")):
                    label_dirs.append(p)

    return image_dirs, label_dirs


def find_label(img_path, label_dirs):
    stem = img_path.stem
    candidates = []

    parts = list(img_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        new_parts = parts.copy()
        new_parts[idx] = "labels"
        candidates.append(Path(*new_parts).with_suffix(".txt"))

    for ld in label_dirs:
        candidates.append(ld / f"{stem}.txt")
        candidates.extend(ld.rglob(f"{stem}.txt"))

    for c in candidates:
        if c.exists():
            return c

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


def parse_include_targets(include_text):
    if not include_text:
        return set(TARGET_NAMES.keys())

    requested = set()

    for item in include_text.split(","):
        item = norm(item)

        if item in TARGET_ID:
            requested.add(TARGET_ID[item])
        else:
            raise RuntimeError(f"Bilinmeyen include target: {item}")

    return requested


def process(src_root, out_root, prefix, include_targets):
    names, yaml_path = read_names(src_root)

    print("==========================================")
    print("[SRC]", src_root)
    print("[YAML]", yaml_path)
    print("[NAMES]", names)
    print("[PREFIX]", prefix)
    print("[INCLUDE]", [TARGET_NAMES[i] for i in sorted(include_targets)])

    write_data_yaml(out_root)

    total_saved = 0
    total_counts = Counter()

    for split in ["train", "val", "test"]:
        image_dirs, label_dirs = find_split_dirs(src_root, split)

        if not image_dirs or not label_dirs:
            print(f"[WARN] split yok/eksik: {split}")
            continue

        out_img = out_root / "images" / split
        out_lab = out_root / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lab.mkdir(parents=True, exist_ok=True)

        images = []
        for d in image_dirs:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                images.extend(d.rglob(ext))

        images = sorted(set(images))

        existing = list(out_img.glob(f"{prefix}_{split}_*.*"))
        saved = len(existing)
        start_saved = saved
        split_counts = Counter()

        for img in images:
            lab = find_label(img, label_dirs)

            if lab is None:
                continue

            new_lines = []

            for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:
                    old_cls = int(float(parts[0]))
                except Exception:
                    continue

                old_name = names.get(old_cls)

                if old_name is None:
                    continue

                new_cls = map_name_to_target(old_name)

                if new_cls is None:
                    continue

                if new_cls not in include_targets:
                    continue

                vals = parts[1:5]
                new_lines.append(f"{new_cls} {' '.join(vals)}")
                split_counts[TARGET_NAMES[new_cls]] += 1
                total_counts[TARGET_NAMES[new_cls]] += 1

            if not new_lines:
                continue

            new_stem = f"{prefix}_{split}_{saved:07d}"
            new_img_name = new_stem + img.suffix.lower()
            new_lab_name = new_stem + ".txt"

            shutil.copy2(img, out_img / new_img_name)
            (out_lab / new_lab_name).write_text("\n".join(new_lines) + "\n", encoding="utf-8")

            saved += 1

        split_saved = saved - start_saved
        total_saved += split_saved

        print(f"[{split}] added_images={split_saved} counts={dict(split_counts)}")

    print("[DONE SRC]", src_root)
    print("[ADDED IMAGES]", total_saved)
    print("[ADDED COUNTS]", dict(total_counts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--include", default="")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    include_targets = parse_include_targets(args.include)

    process(src_root, out_root, args.prefix, include_targets)


if __name__ == "__main__":
    main()
