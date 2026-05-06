#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from pathlib import Path

import carla
import cv2
import numpy as np


CLASS_NAMES = ["car", "person", "traffic_light"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def connect(host, port):
    client = carla.Client(host, port)
    client.set_timeout(40.0)
    world = client.get_world()
    return client, world


def tick(world, n=5):
    for _ in range(n):
        try:
            world.tick()
        except Exception:
            try:
                world.wait_for_tick(seconds=1.0)
            except Exception:
                pass
        time.sleep(0.04)


def clear_dataset_actors(world):
    roles = {
        "dataset_ego",
        "dataset_camera",
        "dataset_vehicle",
        "dataset_walker",
    }

    removed = 0

    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")
        if role in roles:
            try:
                actor.destroy()
                removed += 1
            except Exception:
                pass

    return removed


def set_world(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    weather = carla.WeatherParameters(
        cloudiness=random.uniform(20, 55),
        precipitation=0.0,
        sun_altitude_angle=random.uniform(25, 55),
        sun_azimuth_angle=random.uniform(0, 180),
        fog_density=random.uniform(0, 1.5),
        wetness=0.0,
    )

    world.set_weather(weather)


def get_wp(world, loc):
    return world.get_map().get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def choose_spawn(world):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        wp = get_wp(world, sp.location)

        if wp is None:
            continue

        if wp.is_junction:
            continue

        base_yaw = wp.transform.rotation.yaw
        ok = True

        for d in [8, 15, 25, 35, 45]:
            nxt = wp.next(float(d))
            if not nxt:
                ok = False
                break

            nwp = nxt[0]

            if nwp.is_junction:
                ok = False
                break

            if angle_diff(base_yaw, nwp.transform.rotation.yaw) > 18.0:
                ok = False
                break

        if ok:
            return sp

    if not spawn_points:
        raise RuntimeError("Spawn point yok.")

    return spawn_points[0]


def spawn_ego_and_camera(world, width, height, fov):
    bp_lib = world.get_blueprint_library()

    ego_bp = bp_lib.find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "dataset_ego")

    if ego_bp.has_attribute("color"):
        ego_bp.set_attribute("color", "0,0,255")

    ego_tf = choose_spawn(world)
    ego_tf.location.z += 0.5

    ego = world.try_spawn_actor(ego_bp, ego_tf)

    if ego is None:
        raise RuntimeError("Ego spawn edilemedi.")

    ego.set_autopilot(False)
    ego.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True,
            steer=0.0,
        )
    )

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("role_name", "dataset_camera")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", "0.10")

    # Clean görüntü ayarları
    safe_attrs = {
        "enable_postprocess_effects": "false",
        "gamma": "2.2",
        "shutter_speed": "100",
        "iso": "200",
        "motion_blur_intensity": "0.0",
        "motion_blur_max_distortion": "0.0",
        "lens_flare_intensity": "0.0",
        "bloom_intensity": "0.0",
    }

    for k, v in safe_attrs.items():
        if cam_bp.has_attribute(k):
            cam_bp.set_attribute(k, v)

    cam_tf = carla.Transform(
        carla.Location(x=1.8, y=0.0, z=1.55),
        carla.Rotation(pitch=-4.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    return ego, camera


def choose_vehicle_bp(bp_lib):
    preferred = [
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.lincoln.mkz_2020",
        "vehicle.toyota.prius",
        "vehicle.bmw.grandtourer",
        "vehicle.mercedes.coupe",
    ]

    for name in preferred:
        xs = bp_lib.filter(name)
        if len(xs) > 0:
            return xs[0]

    xs = bp_lib.filter("vehicle.*")

    if len(xs) == 0:
        raise RuntimeError("Vehicle blueprint yok.")

    return random.choice(xs)


def relative_tf(ego, x, y, z, yaw_offset=0.0):
    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    loc = carla.Location(
        x=ego_tf.location.x + fwd.x * x + right.x * y,
        y=ego_tf.location.y + fwd.y * x + right.y * y,
        z=ego_tf.location.z + z,
    )

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=ego_tf.rotation.yaw + yaw_offset,
            roll=0.0,
        ),
    )


def spawn_vehicles(world, ego, count):
    bp_lib = world.get_blueprint_library()
    actors = []

    # yakın + orta + uzak karışık
    slots = [
        (random.uniform(6, 10), random.choice([-1, 1]) * random.uniform(1.8, 3.2)),
        (random.uniform(10, 16), random.choice([-1, 1]) * random.uniform(2.0, 3.8)),
        (random.uniform(16, 24), random.choice([-1, 1]) * random.uniform(2.2, 4.2)),
        (random.uniform(24, 36), random.choice([-1, 1]) * random.uniform(2.5, 4.5)),
        (random.uniform(36, 50), random.choice([-1, 1]) * random.uniform(2.8, 4.8)),
    ]

    random.shuffle(slots)

    for i in range(count):
        x, y = slots[i % len(slots)]

        bp = choose_vehicle_bp(bp_lib)
        bp.set_attribute("role_name", "dataset_vehicle")

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", random.choice(colors))

        yaw_offset = random.choice([0.0, 0.0, 0.0, 180.0])
        tf = relative_tf(ego, x=x, y=y, z=0.5, yaw_offset=yaw_offset)

        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            continue

        actor.set_autopilot(False)
        actor.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True,
                steer=0.0,
            )
        )

        actors.append(actor)

    return actors


def spawn_walkers(world, ego, count):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    actors = []

    if len(walker_bps) == 0:
        return actors

    slots = [
        (random.uniform(5, 9), random.choice([-1, 1]) * random.uniform(0.5, 2.0)),
        (random.uniform(9, 14), random.choice([-1, 1]) * random.uniform(0.5, 2.5)),
        (random.uniform(14, 21), random.choice([-1, 1]) * random.uniform(0.8, 3.0)),
        (random.uniform(21, 32), random.choice([-1, 1]) * random.uniform(1.0, 3.5)),
        (random.uniform(32, 45), random.choice([-1, 1]) * random.uniform(1.2, 4.0)),
    ]

    random.shuffle(slots)

    for i in range(count):
        x, y = slots[i % len(slots)]

        bp = random.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "dataset_walker")

        yaw_offset = random.choice([0.0, 180.0, 90.0, -90.0])
        tf = relative_tf(ego, x=x, y=y, z=1.0, yaw_offset=yaw_offset)

        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            continue

        actors.append(actor)

    return actors


def get_camera_intrinsic(width, height, fov):
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    k = np.identity(3)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    return k


def get_image_point(loc, k, world_to_camera):
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    point_camera = np.dot(world_to_camera, point)

    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    if point_camera[2] <= 0.1:
        return None

    point_img = np.dot(k, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return float(point_img[0]), float(point_img[1])


def actor_to_yolo_bbox(actor, camera, k, width, height):
    bb = actor.bounding_box
    verts = bb.get_world_vertices(actor.get_transform())
    world_to_camera = np.array(camera.get_transform().get_inverse_matrix())

    xs = []
    ys = []

    for v in verts:
        p = get_image_point(v, k, world_to_camera)
        if p is None:
            continue

        x, y = p
        xs.append(x)
        ys.append(y)

    if len(xs) < 2:
        return None

    x_min = max(0.0, min(xs))
    x_max = min(float(width - 1), max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(float(height - 1), max(ys))

    if x_max <= x_min or y_max <= y_min:
        return None

    bw = x_max - x_min
    bh = y_max - y_min

    if bw < 10 or bh < 10:
        return None

    if bw * bh < 140:
        return None

    x_c = (x_min + x_max) / 2.0 / width
    y_c = (y_min + y_max) / 2.0 / height
    w = bw / width
    h = bh / height

    if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0):
        return None

    return x_c, y_c, w, h


def get_traffic_light_labels(world, camera, k, width, height):
    labels = []

    for tl in world.get_actors().filter("traffic.traffic_light*"):
        bbox = actor_to_yolo_bbox(tl, camera, k, width, height)

        if bbox is None:
            continue

        x, y, w, h = bbox

        if w * width < 8 or h * height < 8:
            continue

        labels.append((CLASS_TO_ID["traffic_light"], x, y, w, h))

    return labels


def image_from_carla(carla_image):
    arr = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_image.height, carla_image.width, 4))
    bgr = arr[:, :, :3].copy()
    return bgr


def clean_image(image):
    # Dengeli exposure düzeltmesi:
    # Çok parlak CARLA görüntülerini hafif toparlar,
    # ama görüntüyü siyaha gömmez.
    image = cv2.convertScaleAbs(image, alpha=0.95, beta=-5)
    return image


def collect_one_frame(camera, timeout=5.0):
    holder = {"image": None}

    def callback(image):
        holder["image"] = image

    camera.listen(callback)

    start = time.time()

    while holder["image"] is None and time.time() - start < timeout:
        time.sleep(0.02)

    camera.stop()

    return holder["image"]


def ensure_dirs(out_dir):
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml(out_dir):
    text = f"""path: {out_dir.resolve()}
train: images/train
val: images/val

names:
  0: car
  1: person
  2: traffic_light
"""
    (out_dir / "data.yaml").write_text(text)


def save_sample(out_dir, split, idx, image, labels):
    img_name = f"carla_{idx:06d}.jpg"
    lbl_name = f"carla_{idx:06d}.txt"

    img_path = out_dir / "images" / split / img_name
    lbl_path = out_dir / "labels" / split / lbl_name

    cv2.imwrite(str(img_path), image)

    lines = []

    for cls_id, x, y, w, h in labels:
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    lbl_path.write_text("\n".join(lines))


def draw_preview(image, labels, width, height):
    preview = image.copy()

    colors = {
        0: (0, 255, 0),
        1: (255, 0, 0),
        2: (0, 255, 255),
    }

    for cls_id, x, y, w, h in labels:
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        color = colors.get(cls_id, (255, 255, 255))

        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            preview,
            CLASS_NAMES[cls_id],
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    return preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--out", default="datasets/carla_yolo")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--min-labels", type=int, default=2)
    parser.add_argument("--preview-every", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dirs(out_dir)
    write_yaml(out_dir)

    client, world = connect(args.host, args.port)

    if args.map not in world.get_map().name:
        print(f"Map yükleniyor: {args.map}")
        world = client.load_world(args.map)
        time.sleep(2.0)

    saved = 0
    attempts = 0

    print("CARLA YOLO dataset toplama başladı.")
    print(f"Hedef: {args.frames}")
    print(f"Çıkış: {out_dir}")

    while saved < args.frames:
        attempts += 1

        try:
            clear_dataset_actors(world)
            set_world(world)
            tick(world, 3)

            ego, camera = spawn_ego_and_camera(world, args.width, args.height, args.fov)

            vehicles = spawn_vehicles(world, ego, random.randint(4, 10))
            walkers = spawn_walkers(world, ego, random.randint(2, 8))

            tick(world, 6)

            carla_image = collect_one_frame(camera)

            if carla_image is None:
                print(f"[{attempts}] frame alınamadı")
                continue

            image = image_from_carla(carla_image)
            image = clean_image(image)

            k = get_camera_intrinsic(args.width, args.height, args.fov)

            labels = []

            for v in vehicles:
                bbox = actor_to_yolo_bbox(v, camera, k, args.width, args.height)
                if bbox is not None:
                    labels.append((CLASS_TO_ID["car"], *bbox))

            for w in walkers:
                bbox = actor_to_yolo_bbox(w, camera, k, args.width, args.height)
                if bbox is not None:
                    labels.append((CLASS_TO_ID["person"], *bbox))

            labels.extend(get_traffic_light_labels(world, camera, k, args.width, args.height))

            if len(labels) < args.min_labels:
                print(f"[{attempts}] label az: {len(labels)}")
                continue

            split = "val" if saved % 5 == 0 else "train"

            save_sample(out_dir, split, saved, image, labels)

            if saved % args.preview_every == 0:
                preview = draw_preview(image, labels, args.width, args.height)
                preview_dir = out_dir / "preview"
                preview_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_dir / f"preview_{saved:06d}.jpg"), preview)

            saved += 1

            if saved % 25 == 0:
                car_n = sum(1 for l in labels if l[0] == CLASS_TO_ID["car"])
                per_n = sum(1 for l in labels if l[0] == CLASS_TO_ID["person"])
                tl_n = sum(1 for l in labels if l[0] == CLASS_TO_ID["traffic_light"])
                print(
                    f"Kaydedildi {saved}/{args.frames} | labels={len(labels)} "
                    f"car={car_n} person={per_n} tl={tl_n} split={split}"
                )

        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[{attempts}] Hata: {exc}")

    clear_dataset_actors(world)
    write_yaml(out_dir)

    print("")
    print("BİTTİ")
    print(f"Kaydedilen görüntü: {saved}")
    print(f"Dataset: {out_dir}")
    print(f"YAML: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
