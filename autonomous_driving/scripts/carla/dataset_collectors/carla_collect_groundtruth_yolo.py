#!/usr/bin/env python3
import argparse
import math
import random
import time
import queue
from pathlib import Path
from collections import Counter

import carla
import cv2
import numpy as np


CLASS_NAMES = {
    0: "motorcycle",
    1: "pedestrian",
    2: "traffic_light",
    3: "traffic_sign",
    4: "vehicle",
}

ROLE_PREFIX = "adas_gt_dataset"


def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    k = np.identity(3)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    return k


def get_image_point(location, k, world_to_camera):
    point = np.array([location.x, location.y, location.z, 1.0])
    point_camera = np.dot(world_to_camera, point)

    point_camera = np.array([
        point_camera[1],
        -point_camera[2],
        point_camera[0],
    ])

    depth = point_camera[2]

    if depth <= 0.05:
        return None

    point_img = np.dot(k, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return float(point_img[0]), float(point_img[1]), float(depth)


def classify_actor(actor):
    type_id = actor.type_id.lower()

    if type_id.startswith("walker.pedestrian"):
        return 1

    if type_id.startswith("vehicle."):
        motorcycle_keywords = [
            "yamaha",
            "kawasaki",
            "harley",
            "low_rider",
            "vespa",
            "zx125",
            "ninja",
            "yzf",
        ]

        bicycle_keywords = [
            "crossbike",
            "diamondback",
            "gazelle",
            "omafiets",
            "century",
        ]

        if any(k in type_id for k in motorcycle_keywords):
            return 0

        if any(k in type_id for k in bicycle_keywords):
            return None

        return 4

    if type_id.startswith("traffic.traffic_light"):
        return 2

    if type_id.startswith("traffic."):
        return 3

    if "trafficlight" in type_id or "traffic_light" in type_id:
        return 2

    sign_keywords = [
        "streetsign",
        "trafficwarning",
        "speedlimit",
        "stop",
        "yield",
        "busstop",
    ]

    if any(k in type_id for k in sign_keywords):
        return 3

    return None


def actor_to_yolo_box(actor, camera, k, image_w, image_h):
    try:
        bbox = actor.bounding_box
        vertices = bbox.get_world_vertices(actor.get_transform())
    except Exception:
        return None

    world_to_camera = np.array(camera.get_transform().get_inverse_matrix())

    points = []

    for vertex in vertices:
        p = get_image_point(vertex, k, world_to_camera)
        if p is None:
            continue

        x, y, depth = p

        if depth <= 0.05:
            continue

        points.append((x, y))

    if len(points) < 2:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(image_w - 1), max(xs))
    y2 = min(float(image_h - 1), max(ys))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    if bw < 8 or bh < 8:
        return None

    area_ratio = (bw * bh) / float(image_w * image_h)

    if area_ratio < 0.00003:
        return None

    if bw / float(image_w) > 0.95 or bh / float(image_h) > 0.95:
        return None

    cx = ((x1 + x2) / 2.0) / float(image_w)
    cy = ((y1 + y2) / 2.0) / float(image_h)
    nw = bw / float(image_w)
    nh = bh / float(image_h)

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return cx, cy, nw, nh


def image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def set_role(bp, role):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role)


def disable_invincible(bp):
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")


def find_blueprints(world):
    bp_lib = world.get_blueprint_library()

    pedestrians = list(bp_lib.filter("walker.pedestrian.*"))

    vehicles_all = list(bp_lib.filter("vehicle.*"))

    motorcycle_keywords = [
        "yamaha",
        "kawasaki",
        "harley",
        "low_rider",
        "vespa",
        "zx125",
        "ninja",
        "yzf",
    ]

    bicycle_keywords = [
        "crossbike",
        "diamondback",
        "gazelle",
        "omafiets",
        "century",
    ]

    motorcycles = [
        bp for bp in vehicles_all
        if any(k in bp.id.lower() for k in motorcycle_keywords)
    ]

    vehicles = [
        bp for bp in vehicles_all
        if not any(k in bp.id.lower() for k in motorcycle_keywords)
        and not any(k in bp.id.lower() for k in bicycle_keywords)
    ]

    static_signs = []
    for pattern in [
        "static.prop.streetsign*",
        "static.prop.trafficwarning*",
        "static.prop.busstop*",
    ]:
        static_signs.extend(list(bp_lib.filter(pattern)))

    static_lights = []
    for pattern in [
        "static.prop.trafficlight*",
        "static.prop.traffic_light*",
    ]:
        static_lights.extend(list(bp_lib.filter(pattern)))

    return {
        "pedestrians": pedestrians,
        "motorcycles": motorcycles,
        "vehicles": vehicles,
        "static_signs": static_signs,
        "static_lights": static_lights,
    }


def print_blueprints(world):
    groups = find_blueprints(world)

    print("==========================================")
    print("AVAILABLE BLUEPRINTS")

    for name, items in groups.items():
        print(f"\n{name}: {len(items)}")
        for bp in items[:80]:
            print("  -", bp.id)

    print("==========================================")


def cleanup_old_dataset_actors(world):
    removed = 0

    for actor in world.get_actors():
        role = actor.attributes.get("role_name", "")
        if role.startswith(ROLE_PREFIX):
            try:
                actor.destroy()
                removed += 1
            except Exception:
                pass

    print(f"[OK] Eski dataset role aktörleri silindi: {removed}")


def get_forward_right(transform):
    yaw = math.radians(transform.rotation.yaw)

    forward = carla.Vector3D(
        x=math.cos(yaw),
        y=math.sin(yaw),
        z=0.0,
    )

    right = carla.Vector3D(
        x=math.cos(yaw + math.pi / 2.0),
        y=math.sin(yaw + math.pi / 2.0),
        z=0.0,
    )

    return forward, right


def relative_location(base, forward, right, fwd, lat, z):
    return carla.Location(
        x=base.x + forward.x * fwd + right.x * lat,
        y=base.y + forward.y * fwd + right.y * lat,
        z=base.z + z,
    )


def spawn_ego(world):
    bp_lib = world.get_blueprint_library()

    ego_bp = None
    for bp_id in [
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.lincoln.mkz_2020",
        "vehicle.toyota.prius",
    ]:
        matches = list(bp_lib.filter(bp_id))
        if matches:
            ego_bp = matches[0]
            break

    if ego_bp is None:
        ego_bp = list(bp_lib.filter("vehicle.*"))[0]

    set_role(ego_bp, ROLE_PREFIX + "_ego")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Map spawn point bulunamadı")

    random.shuffle(spawn_points)

    ego = None
    for sp in spawn_points:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego is not None:
            break

    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi")

    try:
        ego.set_simulate_physics(False)
    except Exception:
        pass

    print(f"[OK] Ego spawned: id={ego.id} type={ego.type_id}")
    return ego


def create_camera(world, ego, width, height, fov):
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", "0.0")
    set_role(cam_bp, ROLE_PREFIX + "_camera")

    cam_tf = carla.Transform(
        carla.Location(x=1.60, y=0.0, z=1.70),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(
        cam_bp,
        cam_tf,
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )

    print(f"[OK] Camera spawned: id={camera.id}")
    return camera


def try_spawn(world, bp, tf):
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        return None

    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass

    return actor


def choose(items):
    if not items:
        return None
    return random.choice(items)


def spawn_balanced_layout(world, ego, groups, args):
    spawned = []

    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw
    forward, right = get_forward_right(ego_tf)

    pedestrian_positions = [
        (8, -1.5), (9, 0.0), (10, 1.5),
        (13, -2.2), (14, 0.8), (15, 2.4),
        (18, -1.0), (20, 1.0), (24, 0.0),
    ]

    motorcycle_positions = [
        (10, -3.0), (12, 3.0),
        (18, -3.5), (20, 3.5),
        (26, -2.5), (28, 2.5),
    ]

    vehicle_positions = [
        (16, -4.2), (18, 4.2),
        (24, -4.5), (26, 4.5),
        (34, -2.8), (36, 2.8),
    ]

    sign_positions = [
        (9, -5.0), (12, 5.0),
        (18, -5.2), (21, 5.2),
    ]

    random.shuffle(pedestrian_positions)
    random.shuffle(motorcycle_positions)
    random.shuffle(vehicle_positions)
    random.shuffle(sign_positions)

    def spawn_at(bp, fwd, lat, z, yaw_offset, role_suffix):
        if bp is None:
            return None

        set_role(bp, ROLE_PREFIX + "_" + role_suffix)
        disable_invincible(bp)

        loc = relative_location(ego_loc, forward, right, fwd, lat, z)
        rot = carla.Rotation(
            pitch=0.0,
            yaw=ego_yaw + yaw_offset + random.uniform(-15.0, 15.0),
            roll=0.0,
        )

        tf = carla.Transform(loc, rot)
        actor = try_spawn(world, bp, tf)

        if actor is not None:
            spawned.append(actor)

        return actor

    for i in range(args.pedestrians):
        bp = choose(groups["pedestrians"])
        fwd, lat = pedestrian_positions[i % len(pedestrian_positions)]
        spawn_at(bp, fwd, lat, 0.7, 180.0, "pedestrian")

    for i in range(args.motorcycles):
        bp = choose(groups["motorcycles"])
        fwd, lat = motorcycle_positions[i % len(motorcycle_positions)]
        spawn_at(bp, fwd, lat, 0.6, 180.0, "motorcycle")

    for i in range(args.vehicles):
        bp = choose(groups["vehicles"])
        fwd, lat = vehicle_positions[i % len(vehicle_positions)]
        spawn_at(bp, fwd, lat, 0.6, 180.0, "vehicle")

    for i in range(args.static_signs):
        bp = choose(groups["static_signs"])
        fwd, lat = sign_positions[i % len(sign_positions)]
        spawn_at(bp, fwd, lat, 1.0, 180.0, "traffic_sign")

    for i in range(args.static_lights):
        bp = choose(groups["static_lights"])
        fwd, lat = sign_positions[i % len(sign_positions)]
        spawn_at(bp, fwd + 3.0, lat, 1.0, 180.0, "traffic_light")

    counts = Counter()
    for a in spawned:
        cls = classify_actor(a)
        if cls is not None:
            counts[CLASS_NAMES[cls]] += 1

    print(f"[LAYOUT] spawned={len(spawned)} counts={dict(counts)}")

    return spawned


def destroy_actors(actors):
    for actor in actors:
        try:
            actor.destroy()
        except Exception:
            pass


def write_data_yaml(out_dir):
    root = Path(out_dir).resolve()

    text = f"""path: {root}

train: images/train
val: images/val
test: images/test

names:
  0: motorcycle
  1: pedestrian
  2: traffic_light
  3: traffic_sign
  4: vehicle
"""

    with open(root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(text)


def make_dirs(out_dir, split):
    root = Path(out_dir)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    debug_dir = root / "debug" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    return image_dir, label_dir, debug_dir


def collect_labels(world, camera, ego, image_w, image_h, k):
    labels = []

    skip_ids = {camera.id, ego.id}

    for actor in world.get_actors():
        if actor.id in skip_ids:
            continue

        cls_id = classify_actor(actor)

        if cls_id is None:
            continue

        box = actor_to_yolo_box(actor, camera, k, image_w, image_h)

        if box is None:
            continue

        cx, cy, w, h = box

        labels.append({
            "cls_id": cls_id,
            "name": CLASS_NAMES[cls_id],
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "actor_id": actor.id,
            "type_id": actor.type_id,
        })

    return labels


def draw_debug(image, labels, image_w, image_h):
    debug = image.copy()

    for lab in labels:
        cx = lab["cx"]
        cy = lab["cy"]
        w = lab["w"]
        h = lab["h"]

        x1 = int((cx - w / 2.0) * image_w)
        y1 = int((cy - h / 2.0) * image_h)
        x2 = int((cx + w / 2.0) * image_w)
        y2 = int((cy + h / 2.0) * image_h)

        x1 = max(0, min(image_w - 1, x1))
        y1 = max(0, min(image_h - 1, y1))
        x2 = max(0, min(image_w - 1, x2))
        y2 = max(0, min(image_h - 1, y2))

        text = f"{lab['name']} id={lab['actor_id']}"

        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            debug,
            text,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return debug


def save_sample(image, labels, image_dir, label_dir, debug_dir, split, index, image_w, image_h, debug_every):
    stem = f"carla_gt_{split}_{index:06d}"

    image_path = image_dir / f"{stem}.jpg"
    label_path = label_dir / f"{stem}.txt"

    cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with open(label_path, "w", encoding="utf-8") as f:
        for lab in labels:
            f.write(
                f"{lab['cls_id']} "
                f"{lab['cx']:.6f} "
                f"{lab['cy']:.6f} "
                f"{lab['w']:.6f} "
                f"{lab['h']:.6f}\n"
            )

    if debug_every > 0 and index % debug_every == 0:
        debug = draw_debug(image, labels, image_w, image_h)
        debug_path = debug_dir / f"{stem}_debug.jpg"
        cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="")
    parser.add_argument("--out", default="/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/datasets/carla_gt_yolo_v1")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])

    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--layout-every", type=int, default=25)
    parser.add_argument("--debug-every", type=int, default=25)

    parser.add_argument("--width", type=int, default=1240)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)

    parser.add_argument("--pedestrians", type=int, default=8)
    parser.add_argument("--motorcycles", type=int, default=4)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--static-signs", type=int, default=3)
    parser.add_argument("--static-lights", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-bps", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    if args.map:
        print(f"[INFO] Loading map: {args.map}")
        world = client.load_world(args.map)
        time.sleep(2.0)
    else:
        world = client.get_world()

    print("[OK] CARLA bağlantısı var")
    print("[INFO] Map:", world.get_map().name)
    print("[INFO] Actors:", len(world.get_actors()))

    if args.list_bps:
        print_blueprints(world)
        return

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(root)

    image_dir, label_dir, debug_dir = make_dirs(root, args.split)

    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    cleanup_old_dataset_actors(world)

    groups = find_blueprints(world)

    print("==========================================")
    print("[BLUEPRINT COUNTS]")
    print("pedestrians:", len(groups["pedestrians"]))
    print("motorcycles:", len(groups["motorcycles"]))
    print("vehicles:", len(groups["vehicles"]))
    print("static_signs:", len(groups["static_signs"]))
    print("static_lights:", len(groups["static_lights"]))
    print("==========================================")

    if len(groups["pedestrians"]) == 0:
        raise RuntimeError("Pedestrian blueprint yok")
    if len(groups["motorcycles"]) == 0:
        print("[WARN] Motorcycle blueprint bulunamadı. Motorcycle class az/boş kalabilir.")

    ego = None
    camera = None
    spawned_layout = []

    image_queue = queue.Queue()

    total_counts = Counter()
    saved = 0

    try:
        ego = spawn_ego(world)
        camera = create_camera(world, ego, args.width, args.height, args.fov)

        camera.listen(image_queue.put)

        k = build_projection_matrix(args.width, args.height, args.fov)

        print("==========================================")
        print("[START] Ground-truth YOLO dataset toplama başladı")
        print("[OUT]", root.resolve())
        print("[SPLIT]", args.split)
        print("[FRAMES]", args.frames)
        print("Model kullanılmıyor. Label CARLA actor type_id üzerinden üretiliyor.")
        print("==========================================")

        while saved < args.frames:
            if saved % args.layout_every == 0:
                destroy_actors(spawned_layout)
                spawned_layout = spawn_balanced_layout(world, ego, groups, args)

            while not image_queue.empty():
                try:
                    image_queue.get_nowait()
                except Exception:
                    break

            world.tick()

            try:
                image = image_queue.get(timeout=5.0)
            except queue.Empty:
                print("[WARN] Camera image gelmedi")
                continue

            bgr = image_to_bgr(image)

            labels = collect_labels(
                world=world,
                camera=camera,
                ego=ego,
                image_w=args.width,
                image_h=args.height,
                k=k,
            )

            if len(labels) == 0:
                continue

            save_sample(
                image=bgr,
                labels=labels,
                image_dir=image_dir,
                label_dir=label_dir,
                debug_dir=debug_dir,
                split=args.split,
                index=saved,
                image_w=args.width,
                image_h=args.height,
                debug_every=args.debug_every,
            )

            for lab in labels:
                total_counts[lab["name"]] += 1

            saved += 1

            if saved % 10 == 0:
                print(f"[SAVE] {saved}/{args.frames} labels={len(labels)} counts={dict(total_counts)}")

    except KeyboardInterrupt:
        print("[STOP] Kullanıcı durdurdu")

    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

        destroy_actors(spawned_layout)

        if camera is not None:
            try:
                camera.destroy()
            except Exception:
                pass

        if ego is not None:
            try:
                ego.destroy()
            except Exception:
                pass

        world.apply_settings(original_settings)

        print("==========================================")
        print("[DONE] Dataset toplama bitti")
        print("[SAVED]", saved)
        print("[COUNTS]", dict(total_counts))
        print("[DATA YAML]", root / "data.yaml")
        print("==========================================")


if __name__ == "__main__":
    main()
