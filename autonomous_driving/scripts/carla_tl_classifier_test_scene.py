#!/usr/bin/env python3
import argparse
import math
import time
import threading
import signal
import sys

import carla
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


ROLE_PREFIX = "adas_tl_classifier_test"


def get_blueprint(bp_lib, candidates, fallback_filter=None):
    for name in candidates:
        found = bp_lib.filter(name)
        if found:
            return found[0]

    if fallback_filter:
        found = bp_lib.filter(fallback_filter)
        if found:
            return found[0]

    return None


def set_role(bp, role_name):
    if bp is not None and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)


def rel_transform(base_tf, forward_m, right_m, z_offset=0.2, yaw_offset=0.0):
    fwd = base_tf.get_forward_vector()
    right = base_tf.get_right_vector()

    loc = carla.Location(
        x=base_tf.location.x + fwd.x * forward_m + right.x * right_m,
        y=base_tf.location.y + fwd.y * forward_m + right.y * right_m,
        z=base_tf.location.z + z_offset,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=base_tf.rotation.yaw + yaw_offset,
        roll=0.0,
    )

    return carla.Transform(loc, rot)


def destroy_old_test_actors(world):
    actors = world.get_actors()
    destroy_list = []

    for actor in actors:
        try:
            role = actor.attributes.get("role_name", "")
        except Exception:
            role = ""

        if role.startswith(ROLE_PREFIX):
            destroy_list.append(actor)

    for actor in destroy_list:
        try:
            actor.destroy()
        except Exception:
            pass

    if destroy_list:
        print(f"[CLEAN] Eski test actorleri silindi: {len(destroy_list)}")


def pick_traffic_light(world):
    lights = list(world.get_actors().filter("*traffic_light*"))

    if not lights:
        return None, None

    for tl in lights:
        try:
            stop_wps = tl.get_stop_waypoints()
        except Exception:
            stop_wps = []

        if not stop_wps:
            continue

        for wp in stop_wps:
            if wp is None:
                continue

            try:
                if wp.lane_type != carla.LaneType.Driving:
                    continue
            except Exception:
                pass

            return tl, wp

    return lights[0], None


def freeze_all_lights(world, target_light):
    lights = list(world.get_actors().filter("*traffic_light*"))

    for tl in lights:
        try:
            if tl.id == target_light.id:
                tl.set_state(carla.TrafficLightState.Red)
            else:
                tl.set_state(carla.TrafficLightState.Green)

            tl.freeze(True)
            tl.set_red_time(999.0)
            tl.set_green_time(999.0)
            tl.set_yellow_time(999.0)
        except Exception:
            pass


def spawn_vehicle(world, bp_lib, transform, role_name, candidates, fallback="vehicle.*", physics=False):
    bp = get_blueprint(bp_lib, candidates, fallback)

    if bp is None:
        print(f"[WARN] Vehicle blueprint bulunamadı: {role_name}")
        return None

    set_role(bp, role_name)

    try:
        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,0,0")
    except Exception:
        pass

    actor = world.try_spawn_actor(bp, transform)

    if actor is None:
        print(f"[WARN] Spawn olmadı: {role_name}")
        return None

    try:
        actor.set_simulate_physics(physics)
    except Exception:
        pass

    return actor


def spawn_motorcycle(world, bp_lib, transform, role_name):
    motorcycle_bps = []

    for bp in bp_lib.filter("vehicle.*"):
        try:
            wheels = bp.get_attribute("number_of_wheels").as_int()
        except Exception:
            wheels = None

        bp_id = bp.id.lower()

        if wheels == 2 or "motorcycle" in bp_id or "ninja" in bp_id or "yamaha" in bp_id or "harley" in bp_id:
            motorcycle_bps.append(bp)

    if not motorcycle_bps:
        print("[WARN] Motorcycle blueprint bulunamadı.")
        return None

    bp = motorcycle_bps[0]
    set_role(bp, role_name)

    actor = world.try_spawn_actor(bp, transform)

    if actor is None:
        print(f"[WARN] Motor spawn olmadı: {role_name}")
        return None

    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass

    return actor


def spawn_walker(world, bp_lib, transform, role_name):
    walkers = bp_lib.filter("walker.pedestrian.*")

    if not walkers:
        print("[WARN] Walker blueprint bulunamadı.")
        return None

    bp = walkers[0]
    set_role(bp, role_name)

    actor = world.try_spawn_actor(bp, transform)

    if actor is None:
        print("[WARN] İnsan spawn olmadı.")
        return None

    return actor


def spawn_sign(world, bp_lib, transform):
    candidates = [
        "static.prop.streetsign01",
        "static.prop.streetsign04",
        "static.prop.streetsign",
        "static.prop.trafficwarning",
    ]

    bp = get_blueprint(bp_lib, candidates)

    if bp is None:
        print("[WARN] Trafik levhası blueprint bulunamadı.")
        return None

    actor = world.try_spawn_actor(bp, transform)

    if actor is None:
        print("[WARN] Trafik levhası spawn olmadı.")
        return None

    print(f"[SPAWN] Trafik levhası: {bp.id}")
    return actor


class CameraPublisher(Node):
    def __init__(self, topic, show):
        super().__init__("carla_tl_classifier_test_camera")
        self.pub = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()
        self.topic = topic
        self.show = show
        self.frame_count = 0
        self.last_log = time.time()
        self.get_logger().info(f"Camera publisher başladı: {topic}")

    def publish_carla_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        bgr = array[:, :, :3].copy()

        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_front_camera"
        self.pub.publish(msg)

        self.frame_count += 1

        now = time.time()
        if now - self.last_log > 2.0:
            self.get_logger().info(f"Published frames: {self.frame_count}")
            self.last_log = now

        if self.show:
            cv2.imshow("CARLA TEST CAMERA", bgr)
            cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()

    current_map = world.get_map().name.split("/")[-1]
    if args.town and current_map != args.town:
        print(f"[WORLD] {current_map} -> {args.town} yükleniyor...")
        world = client.load_world(args.town)
        time.sleep(3.0)

    bp_lib = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            sun_altitude_angle=55.0,
            sun_azimuth_angle=20.0,
        )
    )

    destroy_old_test_actors(world)

    traffic_light, stop_wp = pick_traffic_light(world)

    if traffic_light is None:
        print("[ERROR] Bu town içinde trafik ışığı bulunamadı. Town03 yerine Town01 deneyebilirsin.")
        return

    freeze_all_lights(world, traffic_light)

    if stop_wp is not None:
        base_tf = stop_wp.transform
        print(f"[TL] Trafik ışığı bulundu. stop_wp={base_tf.location}")
    else:
        tl_tf = traffic_light.get_transform()
        base_tf = carla.Transform(
            carla.Location(tl_tf.location.x - 25.0, tl_tf.location.y, tl_tf.location.z),
            carla.Rotation(pitch=0.0, yaw=tl_tf.rotation.yaw, roll=0.0),
        )
        print("[WARN] stop_waypoint bulunamadı. Fallback transform kullanılıyor.")

    ego_bp = get_blueprint(
        bp_lib,
        ["vehicle.tesla.model3", "vehicle.lincoln.mkz_2020", "vehicle.audi.tt"],
        "vehicle.*",
    )

    if ego_bp is None:
        print("[ERROR] Ego vehicle blueprint bulunamadı.")
        return

    set_role(ego_bp, f"{ROLE_PREFIX}_ego")

    ego = None
    for dist in [18.0, 22.0, 26.0, 30.0]:
        ego_tf = rel_transform(base_tf, -dist, 0.0, z_offset=0.35, yaw_offset=0.0)
        ego = world.try_spawn_actor(ego_bp, ego_tf)

        if ego is not None:
            print(f"[SPAWN] Ego araç mesafe={dist}m")
            break

    if ego is None:
        print("[ERROR] Ego araç spawn olmadı. Town01 deneyebilirsin.")
        return

    try:
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    except Exception:
        pass

    spawned = [ego]

    ego_tf = ego.get_transform()

    front_car_tf = rel_transform(ego_tf, 18.0, 3.2, z_offset=0.25, yaw_offset=0.0)
    front_car = spawn_vehicle(
        world,
        bp_lib,
        front_car_tf,
        f"{ROLE_PREFIX}_front_car",
        ["vehicle.audi.tt", "vehicle.tesla.model3", "vehicle.lincoln.mkz_2020"],
        physics=False,
    )
    if front_car:
        spawned.append(front_car)

    pedestrian_tf = rel_transform(ego_tf, 16.0, -4.8, z_offset=0.25, yaw_offset=180.0)
    pedestrian = spawn_walker(world, bp_lib, pedestrian_tf, f"{ROLE_PREFIX}_pedestrian")
    if pedestrian:
        spawned.append(pedestrian)

    motor_positions = [
        (11.5, -3.1, 0.0),
        (20.0, -5.8, 0.0),
        (24.0, 5.2, 0.0),
    ]

    for i, (fw, right, yaw_off) in enumerate(motor_positions, start=1):
        motor_tf = rel_transform(ego_tf, fw, right, z_offset=0.25, yaw_offset=yaw_off)
        motor = spawn_motorcycle(world, bp_lib, motor_tf, f"{ROLE_PREFIX}_motor_{i}")
        if motor:
            spawned.append(motor)

    sign_tf = rel_transform(ego_tf, 14.0, 5.8, z_offset=0.25, yaw_offset=180.0)
    sign = spawn_sign(world, bp_lib, sign_tf)
    if sign:
        spawned.append(sign)

    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "960")
    camera_bp.set_attribute("image_size_y", "540")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.05")

    camera_tf = carla.Transform(
        carla.Location(x=1.6, y=0.0, z=1.7),
        carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(camera_bp, camera_tf, attach_to=ego)
    spawned.append(camera)

    rclpy.init()
    node = CameraPublisher(args.topic, args.show)

    camera.listen(lambda img: node.publish_carla_image(img))

    running = True

    def stop_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("")
    print("========== TEST SAHNESİ HAZIR ==========")
    print(f"Town              : {args.town}")
    print(f"ROS image topic   : {args.topic}")
    print("Objeler           : 1 car, 1 person, 3 motorcycle, 1 sign, 1 red traffic light")
    print("Çıkmak için       : CTRL+C")
    print("=========================================")
    print("")

    start = time.time()

    try:
        while running:
            if args.duration > 0 and time.time() - start > args.duration:
                break
            time.sleep(0.1)
    finally:
        print("[CLEAN] Actorler siliniyor...")
        try:
            camera.stop()
        except Exception:
            pass

        for actor in reversed(spawned):
            try:
                actor.destroy()
            except Exception:
                pass

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()

        if args.show:
            cv2.destroyAllWindows()

        print("[DONE]")


if __name__ == "__main__":
    main()
