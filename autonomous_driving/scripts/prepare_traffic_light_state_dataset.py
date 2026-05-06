#!/usr/bin/env python3
from pathlib import Path
import random
import shutil

ROOT = Path("datasets/traffic_light_state_raw")
OUT = Path("datasets/traffic_light_state")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    "red": "red",
    "yellow": "yellow",
    "green": "green",
    "black": "unknown",
    "back": "unknown",
    "rear": "unknown",
    "backside": "unknown",
    "unknown": "unknown",
    "null": "unknown",
    "none": "unknown",
    "negative": "unknown",
    "off": "unknown",
}

SPLITS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}

random.seed(42)


def detect_class(path: Path):
    text = " ".join([p.lower() for p in path.parts])
    text = text.replace("-", " ").replace("_", " ").replace(".", " ")

    tokens = set(text.split())

    for key, mapped in CLASS_MAP.items():
        if key in tokens:
            return mapped

    # Dosya/klasör isimlerinde birleşik yazım varsa
    lower = str(path).lower()
    if "red" in lower:
        return "red"
    if "yellow" in lower:
        return "yellow"
    if "green" in lower:
        return "green"
    if (
        "black" in lower
        or "back" in lower
        or "rear" in lower
        or "backside" in lower
        or "unknown" in lower
        or "null" in lower
        or "negative" in lower
        or "off" in lower
    ):
        return "unknown"

    return None


def main():
    if not ROOT.exists():
        raise SystemExit(f"Raw dataset yok: {ROOT}")

    if OUT.exists():
        shutil.rmtree(OUT)

    for split in SPLITS:
        for cls in ["red", "yellow", "green", "unknown"]:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    items = {
        "red": [],
        "yellow": [],
        "green": [],
        "unknown": [],
    }

    for img in ROOT.rglob("*"):
        if not img.is_file():
            continue

        if img.suffix.lower() not in IMG_EXTS:
            continue

        cls = detect_class(img)

        if cls is None:
            print(f"[SKIP] class bulunamadı: {img}")
            continue

        items[cls].append(img)

    print("========== RAW COUNTS ==========")
    for cls, files in items.items():
        print(f"{cls}: {len(files)}")

    for cls, files in items.items():
        random.shuffle(files)

        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])

        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, split_items in split_files.items():
            for i, src in enumerate(split_items):
                dst = OUT / split / cls / f"{cls}_{i:06d}{src.suffix.lower()}"
                shutil.copy2(src, dst)

    print("")
    print("========== FINAL COUNTS ==========")
    for split in ["train", "val", "test"]:
        print(f"[{split}]")
        for cls in ["red", "yellow", "green", "unknown"]:
            c = len(list((OUT / split / cls).glob("*")))
            print(f"  {cls}: {c}")

    print("")
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
