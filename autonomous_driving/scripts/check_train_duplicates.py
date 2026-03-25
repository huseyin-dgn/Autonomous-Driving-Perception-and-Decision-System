from pathlib import Path
from collections import Counter

TRAIN_ROOT = Path(r"autonomous_driving_project\bddk\bdd100k\bdd100k\images\100k\train")

def collect_names(folder: Path):
    names = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            names.append(p.name)
    return names

def main():
    names = collect_names(TRAIN_ROOT)
    counter = Counter(names)

    total_files = len(names)
    unique_files = len(counter)
    duplicates = [(name, count) for name, count in counter.items() if count > 1]

    print(f"Toplam dosya: {total_files}")
    print(f"Benzersiz dosya adı: {unique_files}")
    print(f"Aynı isimli dosya sayısı: {len(duplicates)}")

    if duplicates:
        print("\nİlk 20 duplicate:")
        for name, count in duplicates[:20]:
            print(f"{name}: {count}")

if __name__ == "__main__":
    main()