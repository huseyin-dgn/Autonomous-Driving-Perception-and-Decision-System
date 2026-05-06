#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/outputs/models/bdd_yolo_v14/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="debug_outputs/carla_highres/annotated.png")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise RuntimeError(f"Görüntü okunamadı: {image_path}")

    results = model(frame, conf=args.conf, verbose=False)[0]

    names = model.names
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        label = names.get(cls_id, str(cls_id))
        detections.append((label, conf, x1, y1, x2, y2))

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

        text = f"{label} {conf:.2f}"
        cv2.putText(
            frame,
            text,
            (int(x1), max(25, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), frame)

    print(f"Annotated kaydedildi: {out_path}")
    print("Detections:")
    for label, conf, x1, y1, x2, y2 in detections:
        print(f"{label:15s} conf={conf:.3f} bbox=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")


if __name__ == "__main__":
    main()
