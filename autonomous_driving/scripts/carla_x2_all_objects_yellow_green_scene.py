#!/usr/bin/env python3
import argparse
import math
import random
import time
from typing import List, Optional, Tuple

import carla
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


ROLE_PREFIX = "adas_x2_scene_"


def set_bp_attr(bp, name: str, value: str):
    if bp.has_attribute(name):
        bp.set_attribute(name, value)


def pick_bp(bp_lib, filters: List[str]):
    for f in filters:
        bps = list(bp_lib.filter(f))
        if bps:
            return random.choice(bps)
    return None


def rel_transform(base_tf: carla.Transform, forward_m: float, right_m: float, z_m: float = 0.3, yaw_offset: float = 0.0):
    yaw = math.radians(base_tf.rotation.yaw)

    fwd = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
    right = carla.Vector3D(-math.sin(yaw), math.cos(yaw), 0.0)

    loc = base_tf.location
    new_loc = carla.Location(
        x=loc.x + fwd.x * forward_m + right.x * right_m,
        y=loc.y + fwd.y * forward_m + right.y * right_m,
        z=loc.z + z_m,
    )

    return carla.Transform(
        new_loc,
        carla.Rotation(
            pitch=0.0,
            yaw=base_tf.rotation.yaw + yaw_offset,
            roll=0.0,
        ),
    )


def destroy_old_test_actors(client: carla.Client, world: carla.World):
    old_ids = []
    for a in world.get_actors():
        role = a.attributes.get("role_name", "")
        if role.startswith(ROLE_PREFIX):
            old_ids.append(a.id)

    if old_ids:
        print(f"[CLEAN] Eski test aktörleri siliniyor: {len(old_ids)}")
        client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in old_ids])
        time.sleep(0.5)


def try_spawn(world: carla.World, bp, tf: carla.Transform, name: str):
    actor = None

    # Çarpışma yüzünden spawn olmazsa biraz yukarı alıp tekrar dene.
    for dz in [0.0, 0.25, 0.50, 0.80, 1.10]:
        tf2 = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + dz),
            tf.rotation,
        )
        actor = world.try_spawn_actor(bp, tf2)
        if actor is not None:
            print(f"[SPAWN] {name}: id={actor.id}, type={actor.type_id}, loc={tf2.location}")
            return actor

    print(f"[WARN] Spawn başarısız: {name}")
    return None


def freeze_actor(actor):
    if actor is None:
        return
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass


def draw_light_rgb(img: np.ndarray, x: int, y: int, state: str, scale: float = 1.0):
    """
    RGB görüntü üstüne kontrollü trafik ışığı çizer.
    state: yellow veya green
    """
    state = state.lower().strip()
    if state not in ("yellow", "green"):
        state = "yellow"

    bw = int(72 * scale)
    bh = int(178 * scale)
    radius = int(22 * scale)
    gap = int(52 * scale)

    # Renkler RGB formatında.
    black = (8, 8, 8)
    dark = (20, 22, 22)
    gray = (90, 90, 90)
    white = (240, 240, 240)
    yellow = (255, 240, 0)
    green = (0, 255, 35)

    active_color = yellow if state == "yellow" else green
    outline = active_color

    x1, y1 = x, y
    x2, y2 = x + bw, y + bh

    # Gövde
    cv2.rectangle(img, (x1, y1), (x2, y2), black, thickness=-1)
    cv2.rectangle(img, (x1, y1), (x2, y2), outline, thickness=max(3, int(4 * scale)))

    cx = x + bw // 2
    cy_top = y + int(42 * scale)
    cy_mid = cy_top + gap
    cy_bot = cy_mid + gap

    # Pasif lensler
    for cy in [cy_top, cy_mid, cy_bot]:
        cv2.circle(img, (cx, cy), radius, dark, thickness=-1)
        cv2.circle(img, (cx, cy), radius, gray, thickness=max(2, int(3 * scale)))

    # Aktif lens: sarı = orta, yeşil = alt
    active_cy = cy_mid if state == "yellow" else cy_bot

    # Hafif parlama
    cv2.circle(img, (cx, active_cy), int(radius * 1.45), active_color, thickness=2)
    cv2.circle(img, (cx, active_cy), radius, active_color, thickness=-1)
    cv2.circle(img, (cx - int(7 * scale), active_cy - int(7 * scale)), int(radius * 0.35), white, thickness=-1)

    # Sadece state etiketi. RED çizilmiyor.
    label = state.upper()
    cv2.putText(
        img,
        label,
        (x1, max(25, y1 - int(10 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * scale,
        active_color,
        max(2, int(3 * scale)),
        cv2.LINE_AA,
    )


class CarlaRosPublisher(Node):
    def __init__(self, topic: str, state_left: str, state_right: str, light_scale: float):
        super().__init__("carla_x2_all_objects_publisher")
        self.pub = self.create_publisher(Image, topic, 10)
        self.topic = topic
        self.state_left = state_left
        self.state_right = state_right
        self.light_scale = light_scale
        self.frame_count = 0
        self.get_logger().info(f"Publishing CARLA RGB to {topic}")
        self.get_logger().info(f"Overlay traffic lights: left={state_left}, right={state_right}")

    def handle_carla_image(self, carla_image):
        w = carla_image.width
        h = carla_image.height

        bgra = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        bgra = bgra.reshape((h, w, 4))

        # CARLA BGRA -> RGB
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

        # Kontrollü 2 trafik ışığı.
        # Büyük çiziliyor ki YOLO rahat yakalasın.
        left_x = int(w * 0.42)
        right_x = int(w * 0.60)
        light_y = int(h * 0.16)

        draw_light_rgb(rgb, left_x, light_y, self.state_left, self.light_scale)
        draw_light_rgb(rgb, right_x, light_y, self.state_right, self.light_scale)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_front_rgb"
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = rgb.tobytes()

        self.pub.publish(msg)

        self.frame_count += 1
        if self.frame_count % 60 == 0:
            self.get_logger().info(
                f"published={self.frame_count}, lights={self.state_left},{self.state_right}, size={w}x{h}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=15)

    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=75.0)
    parser.add_argument("--fps", type=float, default=20.0)

    parser.add_argument("--state-left", choices=["yellow", "green"], default="yellow")
    parser.add_argument("--state-right", choices=["yellow", "green"], default="green")
    parser.add_argument("--light-scale", type=float, default=1.18)

    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    print("[INFO] Connected to CARLA")
    print("[INFO] Map:", world.get_map().name)

    if not args.no_clean:
        destroy_old_test_actors(client, world)

    spawned = []

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Map spawn point yok.")

    ego_tf = spawn_points[args.spawn_index % len(spawn_points)]

    # Ego araç
    ego_bp = pick_bp(bp_lib, ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.lincoln.mkz_2020", "vehicle.*"])
    if ego_bp is None:
        raise RuntimeError("Araç blueprint bulunamadı.")

    set_bp_attr(ego_bp, "role_name", ROLE_PREFIX + "ego")
    set_bp_attr(ego_bp, "color", "0,0,255")

    ego = try_spawn(world, ego_bp, ego_tf, "ego_vehicle")
    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi. --spawn-index değiştir.")

    spawned.append(ego)
    freeze_actor(ego)

    # Kamera
    cam_bp = bp_lib.find("sensor.camera.rgb")
    set_bp_attr(cam_bp, "role_name", ROLE_PREFIX + "front_camera")
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_bp.set_attribute("sensor_tick", str(1.0 / args.fps))

    cam_tf = carla.Transform(
        carla.Location(x=1.8, y=0.0, z=1.65),
        carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    spawned.append(camera)
    print(f"[SPAWN] camera: id={camera.id}, topic={args.topic}, {args.width}x{args.height}")

    # 2 araç
    vehicle_filters = [
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
        "vehicle.lincoln.mkz_2020",
        "vehicle.mercedes.coupe",
        "vehicle.*",
    ]

    vehicle_positions = [
        (18.0, -1.8, 180.0, "vehicle_1"),
        (25.0, 2.8, 180.0, "vehicle_2"),
    ]

    for idx, (fx, ry, yaw_off, name) in enumerate(vehicle_positions):
        bp = pick_bp(bp_lib, vehicle_filters)
        if bp is None:
            print("[WARN] Vehicle blueprint yok")
            continue

        set_bp_attr(bp, "role_name", ROLE_PREFIX + name)
        if bp.has_attribute("color"):
            colors = ["255,0,0", "0,0,255", "0,255,0", "20,20,20"]
            bp.set_attribute("color", colors[idx % len(colors)])

        actor = try_spawn(world, bp, rel_transform(ego_tf, fx, ry, 0.40, yaw_off), name)
        if actor:
            spawned.append(actor)
            freeze_actor(actor)

    # 2 motosiklet
    motorcycle_filters = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
    ]

    motorcycle_positions = [
        (10.5, -4.0, 170.0, "motorcycle_1"),
        (14.0, -6.2, 170.0, "motorcycle_2"),
    ]

    for idx, (fx, ry, yaw_off, name) in enumerate(motorcycle_positions):
        bp = pick_bp(bp_lib, motorcycle_filters)
        if bp is None:
            print("[WARN] Motorcycle blueprint yok. CARLA blueprint listende motor yok olabilir.")
            continue

        set_bp_attr(bp, "role_name", ROLE_PREFIX + name)
        if bp.has_attribute("color"):
            bp.set_attribute("color", "0,255,0" if idx == 0 else "0,0,255")

        actor = try_spawn(world, bp, rel_transform(ego_tf, fx, ry, 0.35, yaw_off), name)
        if actor:
            spawned.append(actor)
            freeze_actor(actor)

    # 2 insan — kameraya yakın ve ayrı koyuldu, büyük görünsün diye.
    walker_filters = [
        "walker.pedestrian.0001",
        "walker.pedestrian.0002",
        "walker.pedestrian.0003",
        "walker.pedestrian.0004",
        "walker.pedestrian.0005",
        "walker.pedestrian.*",
    ]

    pedestrian_positions = [
        (8.5, 4.2, 180.0, "person_1"),
        (11.5, 6.1, 180.0, "person_2"),
    ]

    for idx, (fx, ry, yaw_off, name) in enumerate(pedestrian_positions):
        bp = pick_bp(bp_lib, walker_filters)
        if bp is None:
            print("[WARN] Walker blueprint yok")
            continue

        set_bp_attr(bp, "role_name", ROLE_PREFIX + name)
        set_bp_attr(bp, "is_invincible", "true")

        actor = try_spawn(world, bp, rel_transform(ego_tf, fx, ry, 0.85, yaw_off), name)
        if actor:
            spawned.append(actor)
            freeze_actor(actor)

    # Spectator kamerayı ego kameraya yakın ayarla
    time.sleep(0.5)
    try:
        spec = world.get_spectator()
        spec.set_transform(camera.get_transform())
    except Exception:
        pass

    rclpy.init()
    node = CarlaRosPublisher(
        topic=args.topic,
        state_left=args.state_left,
        state_right=args.state_right,
        light_scale=args.light_scale,
    )

    camera.listen(lambda image: node.handle_carla_image(image))

    print("")
    print("==============================================")
    print("ADAS X2 CARLA SCENE READY")
    print("Objects:")
    print("  vehicles      : 2")
    print("  pedestrians   : 2")
    print("  motorcycles   : 2")
    print(f"  traffic lights: 2 overlay => {args.state_left}, {args.state_right}")
    print(f"ROS topic       : {args.topic}")
    print("CTRL+C ile kapat.")
    print("==============================================")
    print("")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("[CLEANUP] Kamera durduruluyor...")
        try:
            camera.stop()
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()

        print(f"[CLEANUP] Aktörler siliniyor: {len(spawned)}")
        client.apply_batch([carla.command.DestroyActor(a.id) for a in spawned if a is not None])
        time.sleep(0.5)
        print("[DONE]")


if __name__ == "__main__":
    main()
