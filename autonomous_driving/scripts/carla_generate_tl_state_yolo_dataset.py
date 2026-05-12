#!/usr/bin/env python3
import argparse
import math
import os
import random
import queue
from pathlib import Path

import cv2
import numpy as np
import carla


CLASS_NAMES = {
    0: "traffic_light_red",
    1: "traffic_light_yellow",
    2: "traffic_light_green",
    3: "traffic_light_unknown",
}

STATE_TO_CLASS = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "unknown": 3,
}

STATE_TO_CARLA = {
    "red": carla.TrafficLightState.Red,
    "yellow": carla.TrafficLightState.Yellow,
    "green": carla.TrafficLightState.Green,
    # unknown için Off kullanıyoruz. Model kararsız/kapalı ışığı ayırmayı öğrenir.
    "unknown": carla.TrafficLightState.Off,
}


def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    k = np.identity(3)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    return k


def look_at(source, target):
    dx = target.x - source.x
    dy = target.y - source.y
    dz = target.z - source.z

    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, dist_xy))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def project_bbox_to_image(bb, actor_transform, camera_transform, k, width, height):
    world_2_camera = np.array(camera_transform.get_inverse_matrix())

    pts = []
    for vertex in bb.get_world_vertices(actor_transform):
        p_world = np.array([vertex.x, vertex.y, vertex.z, 1.0])
        p_camera = world_2_camera @ p_world

        # CARLA camera coordinate dönüşümü
        depth = p_camera[0]
        if depth <= 0.1:
            return None

        x = p_camera[1]
        y = -p_camera[2]

        u = (k[0, 0] * x / depth) + k[0, 2]
        v = (k[1, 1] * y / depth) + k[1, 2]
        pts.append((u, v))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1 = max(0.0, min(width - 1.0, min(xs)))
    y1 = max(0.0, min(height - 1.0, min(ys)))
    x2 = max(0.0, min(width - 1.0, max(xs)))
    y2 = max(0.0, min(height - 1.0, max(ys)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def xyxy_to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return [
        cx / width,
        cy / height,
        bw / width,
        bh / height,
    ]


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0

    return inter / denom


def image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    # CARLA raw_data BGRA gelir. OpenCV BGR ister.
    return arr[:, :, :3].copy()


def flush_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def make_dirs(out_dir):
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_dir):
    yaml_path = out_dir / "data.yaml"
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx, name in CLASS_NAMES.items():
        lines.append(f"  {idx}: {name}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def get_light_box_center(tl, light_box):
    tf = tl.get_transform()
    center = carla.Location(
        x=light_box.location.x,
        y=light_box.location.y,
        z=light_box.location.z,
    )
    tf.transform(center)
    return center


def collect_light_targets(traffic_lights):
    targets = []

    for tl in traffic_lights:
        try:
            boxes = tl.get_light_boxes()
        except Exception:
            continue

        for bb in boxes:
            center = get_light_box_center(tl, bb)
            targets.append((tl, bb, center))

    return targets


def random_camera_transform(target, args):
    angle = random.uniform(0.0, 2.0 * math.pi)
    dist = random.uniform(args.min_distance, args.max_distance)

    cam_loc = carla.Location(
        x=target.x + math.cos(angle) * dist,
        y=target.y + math.sin(angle) * dist,
        z=max(1.4, target.z + random.uniform(-3.2, -0.8)),
    )

    rot = look_at(cam_loc, target)
    rot.yaw += random.uniform(-args.yaw_jitter, args.yaw_jitter)
    rot.pitch += random.uniform(-args.pitch_jitter, args.pitch_jitter)

    return carla.Transform(cam_loc, rot)


def set_all_traffic_lights(traffic_lights, state_name):
    state = STATE_TO_CARLA[state_name]

    for tl in traffic_lights:
        try:
            tl.set_state(state)
            tl.freeze(True)
        except Exception:
            pass


def collect_visible_labels(traffic_lights, camera, class_id, k, args):
    labels = []
    cam_tf = camera.get_transform()

    for tl in traffic_lights:
        tl_tf = tl.get_transform()

        try:
            boxes = tl.get_light_boxes()
        except Exception:
            continue

        for bb in boxes:
            xyxy = project_bbox_to_image(
                bb,
                tl_tf,
                cam_tf,
                k,
                args.width,
                args.height,
            )

            if xyxy is None:
                continue

            x1, y1, x2, y2 = xyxy
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            if bw < args.min_box_px or bh < args.min_box_px:
                continue

            if area < args.min_area_px:
                continue

            ratio = bh / max(1.0, bw)

            # Trafik ışığı kutusu genelde dikeydir. Çok yatay/saçma kutuları at.
            if ratio < 1.15 or ratio > 5.8:
                continue

            labels.append((class_id, xyxy))

    # duplicate bbox temizliği
    labels = sorted(labels, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]), reverse=True)

    filtered = []
    for cid, box in labels:
        duplicate = False
        for _, old_box in filtered:
            if bbox_iou(box, old_box) > 0.85:
                duplicate = True
                break

        if not duplicate:
            filtered.append((cid, box))

    return filtered


def choose_split(args):
    r = random.random()
    if r < args.train_ratio:
        return "train"
    if r < args.train_ratio + args.val_ratio:
        return "val"
    return "test"


def save_sample(out_dir, split, image_bgr, labels, width, height, state_name, index):
    stem = f"{state_name}_{index:06d}"

    image_path = out_dir / "images" / split / f"{stem}.jpg"
    label_path = out_dir / "labels" / split / f"{stem}.txt"

    cv2.imwrite(str(image_path), image_bgr)

    lines = []
    for class_id, xyxy in labels:
        x, y, w, h = xyxy_to_yolo(xyxy, width, height)
        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--out", default="datasets/carla_tl_state_yolo")

    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=70.0)

    parser.add_argument("--samples-per-state", type=int, default=400)
    parser.add_argument("--states", nargs="+", default=["red", "yellow", "green", "unknown"],
                        choices=["red", "yellow", "green", "unknown"])

    parser.add_argument("--min-distance", type=float, default=10.0)
    parser.add_argument("--max-distance", type=float, default=35.0)
    parser.add_argument("--yaw-jitter", type=float, default=8.0)
    parser.add_argument("--pitch-jitter", type=float, default=5.0)

    parser.add_argument("--min-box-px", type=float, default=12.0)
    parser.add_argument("--min-area-px", type=float, default=220.0)

    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=20.0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    make_dirs(out_dir)
    yaml_path = write_data_yaml(out_dir)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    original_settings = world.get_settings()

    camera = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.10
        world.apply_settings(settings)

        world.set_weather(carla.WeatherParameters.ClearNoon)

        traffic_lights = list(world.get_actors().filter("*traffic_light*"))

        if not traffic_lights:
            raise RuntimeError("CARLA dünyasında traffic_light actor bulunamadı. Town10HD/Town03 gibi ışıklı map aç.")

        targets = collect_light_targets(traffic_lights)

        if not targets:
            raise RuntimeError("Traffic light bulundu ama get_light_boxes boş döndü.")

        print(f"[INFO] traffic_lights={len(traffic_lights)} light_box_targets={len(targets)}")
        print(f"[INFO] output={out_dir.resolve()}")
        print(f"[INFO] data_yaml={yaml_path.resolve()}")

        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(args.width))
        bp.set_attribute("image_size_y", str(args.height))
        bp.set_attribute("fov", str(args.fov))
        bp.set_attribute("sensor_tick", "0.0")

        first_target = random.choice(targets)[2]
        camera_tf = random_camera_transform(first_target, args)

        camera = world.spawn_actor(bp, camera_tf)
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        k = build_projection_matrix(args.width, args.height, args.fov)

        world.tick()
        flush_queue(image_queue)

        global_index = 0

        for state_name in args.states:
            class_id = STATE_TO_CLASS[state_name]
            created = 0
            attempts = 0
            max_attempts = args.samples_per_state * 30

            print(f"[INFO] state={state_name} class_id={class_id} hedef={args.samples_per_state}")

            set_all_traffic_lights(traffic_lights, state_name)

            while created < args.samples_per_state and attempts < max_attempts:
                attempts += 1

                _, _, target_center = random.choice(targets)
                camera_tf = random_camera_transform(target_center, args)
                camera.set_transform(camera_tf)
                world.get_spectator().set_transform(camera_tf)

                flush_queue(image_queue)

                # Işık state ve kamera transformu görüntüye otursun diye birkaç tick.
                world.tick()
                world.tick()

                try:
                    image = image_queue.get(timeout=3.0)
                except queue.Empty:
                    continue

                labels = collect_visible_labels(
                    traffic_lights=traffic_lights,
                    camera=camera,
                    class_id=class_id,
                    k=k,
                    args=args,
                )

                if not labels:
                    continue

                image_bgr = image_to_bgr(image)
                split = choose_split(args)

                save_sample(
                    out_dir=out_dir,
                    split=split,
                    image_bgr=image_bgr,
                    labels=labels,
                    width=args.width,
                    height=args.height,
                    state_name=state_name,
                    index=global_index,
                )

                created += 1
                global_index += 1

                if created % 50 == 0:
                    print(f"[INFO] state={state_name} created={created}/{args.samples_per_state} attempts={attempts}")

            print(f"[INFO] state={state_name} tamamlandı created={created} attempts={attempts}")

        print("[OK] Dataset üretildi.")
        print(f"[OK] YAML: {yaml_path.resolve()}")

    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()

        for actor in world.get_actors().filter("*traffic_light*"):
            try:
                actor.freeze(False)
            except Exception:
                pass

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
