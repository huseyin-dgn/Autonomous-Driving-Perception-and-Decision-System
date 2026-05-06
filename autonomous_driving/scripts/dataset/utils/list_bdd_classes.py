import json
from pathlib import Path
from collections import Counter

JSON_FILES = [
    Path(r"autonomous_driving_project\bddk\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"),
    Path(r"autonomous_driving_project\bddk\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json"),
]

OUTPUT_TXT = Path(r"autonomous_driving_project\outputs\logs\bdd_classes_all.txt")


def main():
    class_counter = Counter()
    total_images = 0
    total_labels = 0

    for json_path in JSON_FILES:
        if not json_path.exists():
            print(f"Bulunamadı: {json_path}")
            continue

        print(f"Okunuyor: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            total_images += 1
            labels = item.get("labels", [])

            for lab in labels:
                category = lab.get("category")
                if category:
                    class_counter[category] += 1
                    total_labels += 1

    if not class_counter:
        print("Hiç class bulunamadı. JSON yapısı farklı olabilir.")
        return

    print("\n===== SINIFLAR =====")
    for cls_name, count in class_counter.most_common():
        print(f"{cls_name}: {count}")

    print("\n===== ÖZET =====")
    print(f"Toplam image kaydı: {total_images}")
    print(f"Toplam annotation: {total_labels}")
    print(f"Toplam benzersiz class: {len(class_counter)}")

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("===== SINIFLAR =====\n")
        for cls_name, count in class_counter.most_common():
            f.write(f"{cls_name}: {count}\n")

        f.write("\n===== ÖZET =====\n")
        f.write(f"Toplam image kaydı: {total_images}\n")
        f.write(f"Toplam annotation: {total_labels}\n")
        f.write(f"Toplam benzersiz class: {len(class_counter)}\n")

    print(f"\nSonuç dosyaya yazıldı: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()