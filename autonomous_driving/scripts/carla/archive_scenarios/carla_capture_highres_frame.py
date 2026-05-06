#!/usr/bin/env python3

import argparse
import queue
import time
from pathlib import Path

import carla
import cv2
import numpy as np


def find_ego(world):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "ego_vehicle":
            return actor
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--out-dir", default="debug_outputs/carla_highres")
    parser.add_argument("--name", default="carla_highres_frame.png")
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()

    ego = find_ego(world)

    if ego is None:
        raise RuntimeError("ego_vehicle bulunamadı. Önce senaryo scriptini çalıştır.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.name

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("role_name", "highres_capture")
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_bp.set_attribute("sensor_tick", "0.10")

    cam_tf = carla.Transform(
        carla.Location(x=1.80, y=0.0, z=1.55),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    q = queue.Queue()
    camera.listen(q.put)

    print(f"High-res kamera oluşturuldu: {args.width}x{args.height}, fov={args.fov}")
    print("Kare bekleniyor...")

    image = None

    for _ in range(30):
        try:
            world.tick()
        except Exception:
            try:
                world.wait_for_tick(seconds=2.0)
            except Exception:
                pass

        try:
            image = q.get(timeout=2.0)
            break
        except queue.Empty:
            pass

    if image is None:
        camera.stop()
        camera.destroy()
        raise RuntimeError("Kamera frame alamadı.")

    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))

    bgr = arr[:, :, :3].copy()

    cv2.imwrite(str(out_path), bgr)

    camera.stop()
    camera.destroy()

    print(f"Kaydedildi: {out_path}")
    print(f"Boyut: {image.width}x{image.height}")


if __name__ == "__main__":
    main()
