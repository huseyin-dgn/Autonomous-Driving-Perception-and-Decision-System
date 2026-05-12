#!/usr/bin/env python3
import argparse
import math
import random
import time
from typing import List, Optional

import carla
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


ROLE_PREFIX = "adas_x2_clear_"


def set_attr(bp, key, value):
    if bp.has_attribute(key):
        bp.set_attribute(key, str(value))


def pick_first(bp_lib, filters: List[str]):
    for f in filters:
        bps = list(bp_lib.filter(f))
        if bps:
            return random.choice(bps)
    return None


def relative_transform(base_tf: carla.Transform, forward: float, right: float, z: float, yaw_offset: float):
    yaw = math.radians(base_tf.rotation.yaw)

    fwd = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
    rgt = carla.Vector3D(-math.sin(yaw), math.cos(yaw), 0.0)

    loc = carla.Location(
        x=base_tf.location.x + fwd.x * forward + rgt.x * right,
        y=base_tf.location.y + fwd.y * forward + rgt.y * right,
        z=base_tf.location.z + z,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=base_tf.rotation.yaw + yaw_offset,
        roll=0.0,
    )

    return carla.Transform(loc, rot)


def destroy_old(client, world):
    ids = []
    for a in world.get_actors():
        role = a.attributes.get("role_name", "")
        if role.startswith(ROLE_PREFIX):
            ids.append(a.id)

    if ids:
        print(f"[CLEAN] Eski adas_x2_clear aktörleri siliniyor: {len(ids)}")
        client.apply_batch([carla.command.DestroyActor(x) for x in ids])
        time.sleep(0.5)


def spawn_try(world, bp, tf, name):
    for dz in [0.0, 0.2, 0.5, 0.9, 1.3]:
        tf2 = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + dz),
            tf.rotation,
        )
        actor = world.try_spawn_actor(bp, tf2)
        if actor:
            print(f"[SPAWN] {name}: id={actor.id}, type={actor.type_id}")
            try:
                actor.set_simulate_physics(False)
            except Exception:
                pass
            return actor

    print(f"[WARN] Spawn olmadi: {name}")
    return None


def draw_clean_tl(rgb, x, y, state, scale=1.0):
    """
    Sadece trafik ışığı gövdesi çizer.
    Yazı çizmez.
    """
    state = state.lower().strip()
    if state not in ["yellow", "green"]:
        state = "yellow"

    body_w = int(52 * scale)
    body_h = int(150 * scale)
    r = int(17 * scale)
    gap = int(43 * scale)

    black = (8, 8, 8)
    dark = (24, 24, 24)
    frame = (70, 70, 70)
    ring = (105, 105, 105)

    yellow = (255, 230, 0)
    green = (0, 255, 40)
    white = (245, 245, 245)

    x1, y1 = x, y
    x2, y2 = x + body_w, y + body_h

    cv2.rectangle(rgb, (x1, y1), (x2, y2), black, -1)
    cv2.rectangle(rgb, (x1, y1), (x2, y2), frame, 3)

    cx = x1 + body_w // 2
    cy1 = y1 + int(30 * scale)
    cy2 = cy1 + gap
    cy3 = cy2 + gap

    for cy in [cy1, cy2, cy3]:
        cv2.circle(rgb, (cx, cy), r, dark, -1)
        cv2.circle(rgb, (cx, cy), r, ring, 2)

    if state == "yellow":
        active_y = cy2
        color = yellow
    else:
        active_y = cy3
        color = green

    cv2.circle(rgb, (cx, active_y), r + 5, color, 2)
    cv2.circle(rgb, (cx, active_y), r, color, -1)
    cv2.circle(rgb, (cx - 5, active_y - 5), max(3, r // 4), white, -1)


class CarlaImagePublisher(Node):
    def __init__(self, topic, left_state, right_state):
        super().__init__("carla_x2_clear_visible_publisher")
        self.pub = self.create_publisher(Image, topic, 10)
        self.topic = topic
        self.left_state = left_state
        self.right_state = right_state
        self.count = 0

        self.get_logger().info(f"Publishing: {topic}")
        self.get_logger().info(f"Overlay TL: left={left_state}, right={right_state}")

    def on_image(self, image):
        w = image.width
        h = image.height

        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((h, w, 4))
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

        # Trafik ışıkları üst bölgede, nesneleri kapatmayacak şekilde.
        draw_clean_tl(rgb, int(w * 0.40), int(h * 0.10), self.left_state, scale=1.25)
        draw_clean_tl(rgb, int(w * 0.56), int(h * 0.10), self.right_state, scale=1.25)

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

        self.count += 1
        if self.count % 60 == 0:
            self.get_logger().info(f"published={self.count}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--spawn-index", type=int, default=15)

    ap.add_argument("--topic", default="/adas/camera/front/image_raw")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=85.0)
    ap.add_argument("--fps", type=float, default=20.0)

    ap.add_argument("--state-left", choices=["yellow", "green"], default="yellow")
    ap.add_argument("--state-right", choices=["yellow", "green"], default="green")

    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    print("[INFO] CARLA baglandi")
    print("[INFO] Map:", world.get_map().name)

    destroy_old(client, world)

    spawned = []

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Spawn point yok.")

    base_tf = spawn_points[args.spawn_index % len(spawn_points)]

    # Ego sadece kamera taşıyacak.
    ego_bp = pick_first(bp_lib, [
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
        "vehicle.*",
    ])
    if ego_bp is None:
        raise RuntimeError("Ego vehicle bp yok.")

    set_attr(ego_bp, "role_name", ROLE_PREFIX + "ego")
    set_attr(ego_bp, "color", "0,0,255")

    ego = spawn_try(world, ego_bp, base_tf, "ego")
    if ego is None:
        raise RuntimeError("Ego spawn olmadi. --spawn-index degistir.")

    spawned.append(ego)

    # Kamera: geniş FOV, biraz yüksek, az aşağı bakıyor.
    cam_bp = bp_lib.find("sensor.camera.rgb")
    set_attr(cam_bp, "role_name", ROLE_PREFIX + "front_camera")
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_bp.set_attribute("sensor_tick", str(1.0 / args.fps))

    cam_tf = carla.Transform(
        carla.Location(x=1.9, y=0.0, z=1.75),
        carla.Rotation(pitch=-4.0, yaw=0.0, roll=0.0)
    )

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    spawned.append(camera)

    # Araçlar: ortada, birbirini kapatmayacak.
    vehicle_filters = [
        "vehicle.tesla.model3",
        "vehicle.dodge.charger_2020",
        "vehicle.lincoln.mkz_2020",
        "vehicle.audi.tt",
    ]

    vehicle_layout = [
        # forward, right, yaw_offset, color, name
        (20.0, -1.8, 180.0, "255,0,0", "vehicle_1"),
        (26.0,  1.8, 180.0, "0,0,255", "vehicle_2"),
    ]

    for forward, right, yaw, color, name in vehicle_layout:
        bp = pick_first(bp_lib, vehicle_filters)
        if bp is None:
            print("[WARN] vehicle bp yok")
            continue

        set_attr(bp, "role_name", ROLE_PREFIX + name)
        set_attr(bp, "color", color)

        tf = relative_transform(base_tf, forward, right, 0.35, yaw)
        actor = spawn_try(world, bp, tf, name)
        if actor:
            spawned.append(actor)

    # MOTORLAR: arka arkaya ama çapraz. Biri diğerini kapatmaz.
    motorcycle_filters = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
    ]

    motorcycle_layout = [
        # forward, right, yaw_offset, color, name
        (10.0, -4.8, 165.0, "0,255,0", "motorcycle_1"),
        (16.0, -6.6, 165.0, "0,0,255", "motorcycle_2"),
    ]

    for forward, right, yaw, color, name in motorcycle_layout:
        bp = pick_first(bp_lib, motorcycle_filters)
        if bp is None:
            print("[WARN] motorcycle bp yok")
            continue

        set_attr(bp, "role_name", ROLE_PREFIX + name)
        set_attr(bp, "color", color)

        tf = relative_transform(base_tf, forward, right, 0.35, yaw)
        actor = spawn_try(world, bp, tf, name)
        if actor:
            spawned.append(actor)

    # İNSANLAR: kameraya yakın, sağ tarafta, çapraz. İkisi de büyük görünür.
    walker_filters = [
        "walker.pedestrian.0001",
        "walker.pedestrian.0002",
        "walker.pedestrian.0003",
        "walker.pedestrian.0004",
        "walker.pedestrian.0005",
        "walker.pedestrian.0006",
        "walker.pedestrian.*",
    ]

    pedestrian_layout = [
        # forward, right, yaw_offset, name
        (9.0,  4.8, 180.0, "person_1"),
        (14.5, 6.8, 180.0, "person_2"),
    ]

    for forward, right, yaw, name in pedestrian_layout:
        bp = pick_first(bp_lib, walker_filters)
        if bp is None:
            print("[WARN] pedestrian bp yok")
            continue

        set_attr(bp, "role_name", ROLE_PREFIX + name)
        set_attr(bp, "is_invincible", "true")

        tf = relative_transform(base_tf, forward, right, 0.95, yaw)
        actor = spawn_try(world, bp, tf, name)
        if actor:
            spawned.append(actor)

    # Spectator kameraya geçsin.
    time.sleep(0.5)
    try:
        world.get_spectator().set_transform(camera.get_transform())
    except Exception:
        pass

    rclpy.init()
    node = CarlaImagePublisher(args.topic, args.state_left, args.state_right)

    camera.listen(lambda img: node.on_image(img))

    print("")
    print("===================================================")
    print("ADAS X2 CLEAR VISIBLE SCENE READY")
    print("2 vehicles")
    print("2 motorcycles: capraz arka arkaya, gorunur")
    print("2 pedestrians: capraz arka arkaya, buyuk/gorunur")
    print(f"2 overlay traffic lights: {args.state_left}, {args.state_right}")
    print(f"topic: {args.topic}")
    print("CTRL+C ile kapat.")
    print("===================================================")
    print("")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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
