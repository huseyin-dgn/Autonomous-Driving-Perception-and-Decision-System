import json
import shutil
from pathlib import Path

TARGET_CLASSES = {
    "car": 0,
    "traffic sign": 1,
    "traffic light": 2,
    "person": 3,
}

MAX_TRAIN_IMAGES = 10000
MAX_VAL_IMAGES = 2000

PROJECT_ROOT = Path(r"autonomous_driving_project")

BDD_ROOT = PROJECT_ROOT / "bddk"
LABELS_ROOT = BDD_ROOT / "bdd100k_labels_release" / "bdd100k" / "labels"
IMAGES_ROOT = BDD_ROOT / "bdd100k" / "bdd100k" / "images" / "100k"

TRAIN_JSON = LABELS_ROOT / "bdd100k_labels_images_train.json"
VAL_JSON = LABELS_ROOT / "bdd100k_labels_images_val.json"

TRAIN_IMAGES_DIR = IMAGES_ROOT / "train"
VAL_IMAGES_DIR = IMAGES_ROOT / "val"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "prepared_data" / "bdd_yolo"
OUT_IMG_TRAIN = OUTPUT_ROOT / "images" / "train"
OUT_IMG_VAL = OUTPUT_ROOT / "images" / "val"
OUT_LBL_TRAIN = OUTPUT_ROOT / "labels" / "train"
OUT_LBL_VAL = OUTPUT_ROOT / "labels" / "val"


def ensure_dirs():
    OUT_IMG_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_IMG_VAL.mkdir(parents=True, exist_ok=True)
    OUT_LBL_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_LBL_VAL.mkdir(parents=True, exist_ok=True)


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def clamp_box(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0.0, min(float(x1), float(img_w - 1)))
    y1 = max(0.0, min(float(y1), float(img_h - 1)))
    x2 = max(0.0, min(float(x2), float(img_w - 1)))
    y2 = max(0.0, min(float(y2), float(img_h - 1)))
    return x1, y1, x2, y2


def parse_box(label):
    box2d = label.get("box2d")
    if not box2d:
        return None

    x1 = box2d.get("x1")
    y1 = box2d.get("y1")
    x2 = box2d.get("x2")
    y2 = box2d.get("y2")

    if None in (x1, y1, x2, y2):
        return None

    return x1, y1, x2, y2


def build_image_index(split_root, allowed_subdirs=None):
    index = {}

    for p in split_root.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index[p.name] = p

    if allowed_subdirs:
        for sub in allowed_subdirs:
            subdir = split_root / sub
            if not subdir.exists():
                continue

            for p in subdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    index[p.name] = p
    else:
        for p in split_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[p.name] = p

    return index


def process_split(
    json_path,
    split_root,
    out_img_dir,
    out_lbl_dir,
    split_name,
    allowed_subdirs=None,
    max_images=None,
):
    print(f"\n[{split_name}] JSON okunuyor: {json_path}")

    image_index = build_image_index(split_root, allowed_subdirs)
    print(f"[{split_name}] Indexlenen görsel sayısı: {len(image_index)}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    copied_images = 0
    total_objects = 0

    image_width = 1280
    image_height = 720

    for item in data:
        image_name = item.get("name")
        if not image_name:
            continue

        image_path = image_index.get(image_name)
        if image_path is None:
            continue

        labels = item.get("labels", [])
        yolo_lines = []

        for label in labels:
            category = label.get("category")
            if category not in TARGET_CLASSES:
                continue

            parsed = parse_box(label)
            if parsed is None:
                continue

            x1, y1, x2, y2 = parsed
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, image_width, image_height)

            if x2 <= x1 or y2 <= y1:
                continue

            x_center, y_center, width, height = xyxy_to_yolo(
                x1, y1, x2, y2, image_width, image_height
            )

            class_id = TARGET_CLASSES[category]
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            total_objects += 1

        if not yolo_lines:
            continue

        dst_img_path = out_img_dir / image_name
        dst_lbl_path = out_lbl_dir / f"{Path(image_name).stem}.txt"

        shutil.copy2(image_path, dst_img_path)

        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        copied_images += 1

        if copied_images % 1000 == 0:
            print(f"[{split_name}] {copied_images} image işlendi | objeler={total_objects}")

        if max_images and copied_images >= max_images:
            print(f"[{split_name}] LIMIT {max_images} ulaşıldı")
            break


def write_yaml():
    yaml_path = PROJECT_ROOT / "configs" / "data_bdd.yaml"

    content = f"""path: {OUTPUT_ROOT.as_posix()}

train: images/train
val: images/val

names:
  0: car
  1: traffic sign
  2: traffic light
  3: person
"""

    yaml_path.write_text(content, encoding="utf-8")
    print(f"[YAML] Yazıldı: {yaml_path}")


def main():
    ensure_dirs()

    process_split(
        TRAIN_JSON,
        TRAIN_IMAGES_DIR,
        OUT_IMG_TRAIN,
        OUT_LBL_TRAIN,
        "TRAIN",
        allowed_subdirs=["trainA", "trainB"],
        max_images=MAX_TRAIN_IMAGES,
    )

    process_split(
        VAL_JSON,
        VAL_IMAGES_DIR,
        OUT_IMG_VAL,
        OUT_LBL_VAL,
        "VAL",
        max_images=MAX_VAL_IMAGES,
    )

    write_yaml()

    print("\n DATASET HAZIR")


if __name__ == "__main__":
    main()