#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def log(msg):
    print(msg, flush=True)


def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, num_classes)
    )
    return model


def count_classes(imagefolder):
    counts = [0 for _ in imagefolder.classes]
    for _, label in imagefolder.samples:
        counts[label] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/traffic_light_state")
    parser.add_argument("--out", default="outputs/models/traffic_light_state_resnet18_carla")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.RandomRotation(degrees=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)

    log(f"Classes: {class_names}")
    log(f"Train: {len(train_ds)} Val: {len(val_ds)} Test: {len(test_ds)}")

    train_counts = count_classes(train_ds)
    log(f"Train counts: {dict(zip(class_names, train_counts))}")

    weights = []
    total = sum(train_counts)
    for c in train_counts:
        weights.append(total / max(c, 1))
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    history = []

    def run_eval(loader, split_name):
        model.eval()
        correct = 0
        total_items = 0
        total_loss = 0.0

        per_cls_correct = [0 for _ in class_names]
        per_cls_total = [0 for _ in class_names]

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total_items += labels.size(0)

                for y, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    per_cls_total[y] += 1
                    if y == p:
                        per_cls_correct[y] += 1

        acc = correct / max(total_items, 1)
        avg_loss = total_loss / max(total_items, 1)

        cls_acc = {}
        for i, name in enumerate(class_names):
            cls_acc[name] = per_cls_correct[i] / max(per_cls_total[i], 1)

        log(f"[{split_name}] loss={avg_loss:.4f} acc={acc:.4f} cls_acc={cls_acc}")

        return avg_loss, acc, cls_acc

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)

            train_loss += loss.item() * labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        log("")
        log(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} time={time.time()-start:.1f}s")

        val_loss, val_acc, val_cls_acc = run_eval(val_loader, "VAL")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_cls_acc": val_cls_acc,
        }
        history.append(row)

        ckpt = {
            "model_name": "resnet18",
            "model_state": model.state_dict(),
            "class_names": class_names,
            "img_size": args.img_size,
            "epoch": epoch,
            "val_acc": val_acc,
            "val_cls_acc": val_cls_acc,
        }

        torch.save(ckpt, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_path)
            log(f"[SAVE] best -> {best_path} val_acc={best_val_acc:.4f}")

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    log("")
    log("========== TEST BEST ==========")
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    run_eval(test_loader, "TEST")

    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": class_names,
                "img_size": args.img_size,
                "best_val_acc": best_val_acc,
                "best_path": str(best_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log("")
    log(f"BEST MODEL: {best_path}")


if __name__ == "__main__":
    main()
