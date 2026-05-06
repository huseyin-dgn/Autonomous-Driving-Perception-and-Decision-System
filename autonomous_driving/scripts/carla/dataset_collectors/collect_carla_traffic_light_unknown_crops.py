#!/usr/bin/env python3
import argparse
import queue
import time
from pathlib import Path

import carla
import cv2
import numpy as np
from ultralytics import YOLO


def log(msg):
    print(msg, flush=True)


def find_rgb_camera(world):
    cams = list(world.get_actors().filter("sensor.camera.rgb"))

    if not cams:
        return None

    def score(cam):
        role = cam.attributes.get("role_name", "").lower()
        s = 0
        if "front" in role:
            s += 100
        if "rgb" in role:
            s += 20
        if "adas" in role:
            s += 20
        return s

    cams.sort(key=score, reverse=True)
    return cams[0]


def carla_image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]
    return bgr.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--model",
        default="/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/outputs/models/adas5_targeted_aug_finetune_from_old_img1024_b8_ep50/weights/best.pt",
    )
    parser.add_argument(
        "--out",
        default="datasets/traffic_light_state_extra_unknown/unknown",
    )
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--min-area", type=int, default=150)
    parser.add_argument("--sleep", type=float, default=0.03)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()

    cam = find_rgb_camera(world)
    if cam is None:
        raise RuntimeError("RGB camera bulunamadı. Önce publisher açık olmalı.")

    log(f"[CAMERA] id={cam.id}, role={cam.attributes.get('role_name', '')}")
    log(f"[MODEL] {args.model}")
    log(f"[OUT] {out_dir}")

    q = queue.Queue(maxsize=5)

    def callback(image):
        try:
            q.put_nowait(image)
        except queue.Full:
            pass

    cam.listen(callback)

    saved = 0
    seen = 0

    try:
        while saved < args.target_count:
            try:
                image = q.get(timeout=5.0)
            except queue.Empty:
                log("[WAIT] camera frame bekleniyor...")
                continue

            frame = carla_image_to_bgr(image)
            seen += 1

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=0.50,
                max_det=30,
                verbose=False,
            )

            if not results:
                continue

            names = results[0].names

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))

                if cls_name != "traffic_light":
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].item())

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(frame.shape[1] - 1, int(x2))
                y2 = min(frame.shape[0] - 1, int(y2))

                bw = x2 - x1
                bh = y2 - y1

                if bw <= 3 or bh <= 3:
                    continue

                if bw * bh < args.min_area:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Bilerek unknown olarak kaydediyoruz:
                # CARLA'da gövde/backside/sarı kasa gibi yanlış state üreten görüntüler.
                fname = out_dir / f"unknown_carla_{saved:06d}_conf{conf:.2f}_{bw}x{bh}.jpg"
                cv2.imwrite(str(fname), crop)
                saved += 1

                if saved % 25 == 0:
                    log(f"[SAVE] unknown crops: {saved}/{args.target_count}")

                if saved >= args.target_count:
                    break

            time.sleep(args.sleep)

    finally:
        try:
            cam.stop()
        except Exception:
            pass

    log(f"[DONE] seen_frames={seen}, saved={saved}")


if __name__ == "__main__":
    main()
