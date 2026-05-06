#!/usr/bin/env python3
import os
import sys
import glob
import time
import math
import random
import argparse
import threading
from typing import List, Optional


def add_carla_python_api():
    try:
        import carla  # noqa
        return
    except ImportError:
        pass

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidate_roots = [
        os.environ.get("CARLA_ROOT"),
        os.path.expanduser("~/CARLA_DISK"),
        os.path.expanduser("~/CARLA"),
        "/opt/carla",
    ]

    for root in candidate_roots:
        if not root:
            continue

        egg_pattern = os.path.join(
            root,
            "PythonAPI",
            "carla",
            "dist",
            f"carla-*{py_ver}-linux-x86_64.egg",
        )

        eggs = glob.glob(egg_pattern)
        if eggs:
            sys.path.append(eggs[0])
            return


add_carla_python_api()

import carla  # noqa: E402

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except ImportError:
    rclpy = None
    Node = object
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def get_speed_kmh(actor):
    velocity = actor.get_velocity()
    speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return speed_mps * 3.6


def get_forward_vector(transform):
    yaw = math.radians(transform.rotation.yaw)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def get_right_vector(transform):
    yaw = math.radians(transform.rotation.yaw + 90.0)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def make_transform_from_waypoint(waypoint, z_offset=0.35):
    tf = carla.Transform(
        carla.Location(
            x=waypoint.transform.location.x,
            y=waypoint.transform.location.y,
            z=waypoint.transform.location.z + z_offset,
        ),
        waypoint.transform.rotation,
    )
    return tf


def offset_transform(transform, forward=0.0, lateral=0.0, z_offset=0.0, yaw_offset=0.0):
    fv = get_forward_vector(transform)
    rv = get_right_vector(transform)

    loc = carla.Location(
        x=transform.location.x + fv.x * forward + rv.x * lateral,
        y=transform.location.y + fv.y * forward + rv.y * lateral,
        z=transform.location.z + z_offset,
    )

    rot = carla.Rotation(
        pitch=transform.rotation.pitch,
        yaw=transform.rotation.yaw + yaw_offset,
        roll=transform.rotation.roll,
    )

    return carla.Transform(loc, rot)


def advance_waypoint(start_wp, distance):
    current = start_wp
    remaining = distance

    while remaining > 0:
        step = min(5.0, remaining)
        next_points = current.next(step)
        if not next_points:
            return current

        next_wp = min(
            next_points,
            key=lambda wp: abs(normalize_angle_deg(wp.transform.rotation.yaw - current.transform.rotation.yaw)),
        )

        current = next_wp
        remaining -= step

    return current


def pick_straight_spawn(world):
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    best_spawn = None
    best_score = -9999

    for sp in spawn_points:
        wp = carla_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is None:
            continue

        if wp.is_junction:
            continue

        current = wp
        score = 0
        valid = True

        for _ in range(8):
            next_points = current.next(8.0)
            if not next_points:
                valid = False
                break

            nxt = min(
                next_points,
                key=lambda w: abs(normalize_angle_deg(w.transform.rotation.yaw - current.transform.rotation.yaw)),
            )

            yaw_diff = abs(normalize_angle_deg(nxt.transform.rotation.yaw - wp.transform.rotation.yaw))

            if yaw_diff > 12.0:
                valid = False
                break

            if not nxt.is_junction:
                score += 2
            else:
                score -= 1

            current = nxt

        if valid and score > best_score:
            best_score = score
            best_spawn = sp

    if best_spawn is None:
        best_spawn = random.choice(spawn_points)

    best_spawn.location.z += 0.45
    return best_spawn


def pick_blueprint(world, filters):
    bp_lib = world.get_blueprint_library()

    for pattern in filters:
        candidates = bp_lib.filter(pattern)
        if candidates:
            return random.choice(candidates)

    return None


def configure_vehicle_bp(bp, role_name):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)

    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))

    if bp.has_attribute("driver_id"):
        drivers = bp.get_attribute("driver_id").recommended_values
        if drivers:
            bp.set_attribute("driver_id", random.choice(drivers))

    return bp


class CarlaImagePublisher(Node):
    def __init__(self, topic_name, preview=False):
        super().__init__("carla_city_flow_image_publisher")
        self.publisher = self.create_publisher(Image, topic_name, 10)
        self.topic_name = topic_name
        self.preview = preview
        self.frame_count = 0
        self.last_log_time = time.time()

        self.get_logger().info(f"CARLA camera ROS2 topic: {topic_name}")

    def publish_carla_image(self, carla_image):
        if np is None:
            return

        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))

        rgb = array[:, :, :3][:, :, ::-1].copy()

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_front_camera"
        msg.height = carla_image.height
        msg.width = carla_image.width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = carla_image.width * 3
        msg.data = rgb.tobytes()

        self.publisher.publish(msg)

        self.frame_count += 1
        now = time.time()

        if now - self.last_log_time > 3.0:
            self.get_logger().info(
                f"Published {self.frame_count} frames to {self.topic_name}"
            )
            self.last_log_time = now

        if self.preview and cv2 is not None:
            bgr = rgb[:, :, ::-1]
            cv2.imshow("CARLA FRONT CAMERA - ADAS INPUT", bgr)
            cv2.waitKey(1)


def cleanup_world(world):
    actors = world.get_actors()

    destroy_list = []

    for actor in actors:
        tid = actor.type_id

        if tid.startswith("sensor."):
            destroy_list.append(actor)
        elif tid.startswith("vehicle."):
            destroy_list.append(actor)
        elif tid.startswith("walker."):
            destroy_list.append(actor)
        elif tid.startswith("controller.ai.walker"):
            destroy_list.append(actor)
        elif tid.startswith("static.prop.trafficcone"):
            destroy_list.append(actor)
        elif tid.startswith("static.prop.streetsign"):
            destroy_list.append(actor)
        elif tid.startswith("static.prop.trafficwarning"):
            destroy_list.append(actor)
        elif tid.startswith("static.prop.busstop"):
            destroy_list.append(actor)

    for actor in destroy_list:
        try:
            if actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass

    if destroy_list:
        print(f"[CLEANUP] Destroyed {len(destroy_list)} previous scene actors.")


def spawn_ego_vehicle(world, ego_transform):
    bp = pick_blueprint(
        world,
        [
            "vehicle.tesla.model3",
            "vehicle.lincoln.mkz_2020",
            "vehicle.audi.tt",
            "vehicle.*",
        ],
    )

    if bp is None:
        raise RuntimeError("Vehicle blueprint bulunamadı.")

    configure_vehicle_bp(bp, "adas_ego_vehicle")

    ego = world.try_spawn_actor(bp, ego_transform)

    if ego is None:
        ego_transform.location.z += 1.0
        ego = world.try_spawn_actor(bp, ego_transform)

    if ego is None:
        raise RuntimeError("Ego araç spawn edilemedi. Spawn noktası dolu olabilir.")

    ego.set_autopilot(False)
    print(f"[EGO] Spawned: {ego.type_id}")
    return ego


def spawn_front_camera(world, ego_vehicle, width, height, fov):
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find("sensor.camera.rgb")

    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", str(fov))
    camera_bp.set_attribute("sensor_tick", "0.033")

    camera_tf = carla.Transform(
        carla.Location(x=1.85, y=0.0, z=1.55),
        carla.Rotation(pitch=-3.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_tf,
        attach_to=ego_vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    print("[CAMERA] Front RGB camera attached to ego vehicle.")
    return camera


def spawn_traffic_vehicles(world, traffic_manager, tm_port, ego_wp):
    spawned = []

    lead_distances = [24, 48, 78]
    vehicle_filters = [
        "vehicle.audi.*",
        "vehicle.bmw.*",
        "vehicle.mercedes.*",
        "vehicle.lincoln.*",
        "vehicle.tesla.*",
        "vehicle.nissan.*",
        "vehicle.toyota.*",
        "vehicle.*",
    ]

    for i, dist in enumerate(lead_distances):
        wp = advance_waypoint(ego_wp, dist)
        tf = make_transform_from_waypoint(wp, z_offset=0.35)

        bp = pick_blueprint(world, vehicle_filters)
        if bp is None:
            continue

        configure_vehicle_bp(bp, f"adas_front_vehicle_{i}")

        actor = world.try_spawn_actor(bp, tf)
        if actor:
            actor.set_autopilot(True, tm_port)
            traffic_manager.distance_to_leading_vehicle(actor, 7.0)
            traffic_manager.vehicle_percentage_speed_difference(actor, random.uniform(10, 35))
            spawned.append(actor)
            print(f"[TRAFFIC] Front vehicle spawned at {dist}m: {actor.type_id}")

    parked_specs = [
        (18, 4.4),
        (32, -4.4),
        (52, 4.4),
        (70, -4.4),
    ]

    for i, (dist, lateral) in enumerate(parked_specs):
        wp = advance_waypoint(ego_wp, dist)
        base_tf = make_transform_from_waypoint(wp, z_offset=0.35)
        tf = offset_transform(base_tf, lateral=lateral, yaw_offset=random.choice([0.0, 180.0]))

        bp = pick_blueprint(world, vehicle_filters)
        if bp is None:
            continue

        configure_vehicle_bp(bp, f"adas_parked_vehicle_{i}")

        actor = world.try_spawn_actor(bp, tf)
        if actor:
            actor.set_autopilot(False)
            spawned.append(actor)
            print(f"[TRAFFIC] Parked vehicle spawned: {actor.type_id}")

    random_spawn_points = world.get_map().get_spawn_points()
    random.shuffle(random_spawn_points)

    ego_loc = ego_wp.transform.location
    count = 0

    for sp in random_spawn_points:
        if count >= 8:
            break

        d = sp.location.distance(ego_loc)

        if d < 35 or d > 180:
            continue

        bp = pick_blueprint(world, vehicle_filters)
        if bp is None:
            continue

        configure_vehicle_bp(bp, f"adas_city_vehicle_{count}")

        sp.location.z += 0.35
        actor = world.try_spawn_actor(bp, sp)

        if actor:
            actor.set_autopilot(True, tm_port)
            traffic_manager.distance_to_leading_vehicle(actor, 7.0)
            traffic_manager.vehicle_percentage_speed_difference(actor, random.uniform(0, 45))
            spawned.append(actor)
            count += 1
            print(f"[TRAFFIC] City flow vehicle spawned: {actor.type_id}")

    return spawned


def spawn_pedestrians(world, ego_wp):
    spawned = []
    bp_lib = world.get_blueprint_library()

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")

    if not walker_bps:
        print("[PEDESTRIAN] Walker blueprint bulunamadı.")
        return spawned

    pedestrian_specs = [
        (22, 5.8),
        (34, -5.8),
        (54, 5.8),
        (74, -5.8),
        (92, 5.8),
    ]

    for i, (dist, lateral) in enumerate(pedestrian_specs):
        wp = advance_waypoint(ego_wp, dist)
        base_tf = make_transform_from_waypoint(wp, z_offset=0.35)
        tf = offset_transform(
            base_tf,
            lateral=lateral,
            yaw_offset=random.choice([-90.0, 90.0, 180.0, 0.0]),
        )

        walker_bp = random.choice(walker_bps)

        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(walker_bp, tf)

        if walker:
            spawned.append(walker)
            print(f"[PEDESTRIAN] Walker spawned: {walker.type_id}")

            controller = world.try_spawn_actor(
                controller_bp,
                carla.Transform(),
                attach_to=walker,
            )

            if controller:
                spawned.append(controller)
                controller.start()

                target_tf = offset_transform(
                    base_tf,
                    forward=random.uniform(10.0, 22.0),
                    lateral=lateral + random.uniform(-1.5, 1.5),
                    z_offset=0.35,
                )

                controller.go_to_location(target_tf.location)
                controller.set_max_speed(random.uniform(0.8, 1.4))

    return spawned


def spawn_city_props(world, ego_wp):
    spawned = []

    prop_specs = [
        ("static.prop.trafficcone01", 14, 3.3, 0.0),
        ("static.prop.trafficcone02", 18, 3.3, 0.0),
        ("static.prop.trafficwarning", 28, 3.7, 0.0),
        ("static.prop.streetsign01", 38, -4.8, 90.0),
        ("static.prop.streetsign04", 58, 4.8, -90.0),
        ("static.prop.busstop", 76, -5.6, 90.0),
        ("static.prop.busstoplb", 88, 5.6, -90.0),
    ]

    for type_id, dist, lateral, yaw_offset in prop_specs:
        bp = pick_blueprint(world, [type_id])
        if bp is None:
            print(f"[PROP] Missing blueprint: {type_id}")
            continue

        wp = advance_waypoint(ego_wp, dist)
        base_tf = make_transform_from_waypoint(wp, z_offset=0.15)
        tf = offset_transform(base_tf, lateral=lateral, yaw_offset=yaw_offset)

        actor = world.try_spawn_actor(bp, tf)

        if actor:
            spawned.append(actor)
            print(f"[PROP] Spawned: {actor.type_id}")
        else:
            print(f"[PROP] Could not spawn: {type_id}")

    return spawned


def set_spectator_follow(world, ego_vehicle):
    spectator = world.get_spectator()
    ego_tf = ego_vehicle.get_transform()

    back_tf = offset_transform(
        ego_tf,
        forward=-8.0,
        lateral=0.0,
        z_offset=4.2,
        yaw_offset=0.0,
    )

    back_tf.rotation.pitch = -18.0
    back_tf.rotation.yaw = ego_tf.rotation.yaw

    spectator.set_transform(back_tf)


def apply_ego_lane_control(ego_vehicle, carla_map, target_speed_kmh):
    tf = ego_vehicle.get_transform()
    loc = ego_vehicle.get_location()

    wp = carla_map.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if wp is None:
        ego_vehicle.apply_control(
            carla.VehicleControl(throttle=0.25, steer=0.0, brake=0.0)
        )
        return

    next_points = wp.next(10.0)

    if not next_points:
        ego_vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3)
        )
        return

    next_wp = min(
        next_points,
        key=lambda w: abs(normalize_angle_deg(w.transform.rotation.yaw - tf.rotation.yaw)),
    )

    target = next_wp.transform.location

    dx = target.x - tf.location.x
    dy = target.y - tf.location.y

    target_yaw = math.degrees(math.atan2(dy, dx))
    yaw_error = normalize_angle_deg(target_yaw - tf.rotation.yaw)

    steer = clamp(yaw_error / 45.0, -0.35, 0.35)

    current_speed = get_speed_kmh(ego_vehicle)

    if current_speed < target_speed_kmh - 3.0:
        throttle = 0.42
        brake = 0.0
    elif current_speed > target_speed_kmh + 5.0:
        throttle = 0.0
        brake = 0.18
    else:
        throttle = 0.20
        brake = 0.0

    ego_vehicle.apply_control(
        carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
        )
    )


def configure_world(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            sun_altitude_angle=45.0,
            sun_azimuth_angle=35.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--no-load-town", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--ego-speed", type=float, default=28.0)
    parser.add_argument("--tm-port", type=int, default=8000)

    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=85.0)

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--no-clean", action="store_true")

    args = parser.parse_args()

    if rclpy is None:
        raise RuntimeError(
            "ROS2 Python modülleri bulunamadı. Önce şunu çalıştır:\n"
            "source /opt/ros/humble/setup.bash"
        )

    if np is None:
        raise RuntimeError(
            "numpy bulunamadı. Kur:\n"
            "pip3 install numpy"
        )

    rclpy.init()

    node = CarlaImagePublisher(args.topic, preview=args.preview)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    if args.no_load_town:
        world = client.get_world()
        print(f"[WORLD] Current world: {world.get_map().name}")
    else:
        print(f"[WORLD] Loading {args.town} ...")
        world = client.load_world(args.town)
        time.sleep(2.0)

    configure_world(world)

    if not args.no_clean:
        cleanup_world(world)
        time.sleep(1.0)

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(7.0)
    traffic_manager.set_synchronous_mode(False)
    traffic_manager.global_percentage_speed_difference(20.0)

    carla_map = world.get_map()
    ego_transform = pick_straight_spawn(world)
    ego_vehicle = spawn_ego_vehicle(world, ego_transform)

    ego_wp = carla_map.get_waypoint(
        ego_vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    actors: List[carla.Actor] = []
    actors.append(ego_vehicle)

    camera = spawn_front_camera(
        world,
        ego_vehicle,
        width=args.width,
        height=args.height,
        fov=args.fov,
    )

    actors.append(camera)

    callback_lock = threading.Lock()

    def camera_callback(image):
        with callback_lock:
            node.publish_carla_image(image)

    camera.listen(camera_callback)

    actors.extend(spawn_traffic_vehicles(world, traffic_manager, args.tm_port, ego_wp))
    actors.extend(spawn_pedestrians(world, ego_wp))
    actors.extend(spawn_city_props(world, ego_wp))

    print("")
    print("===================================================")
    print(" CARLA CITY FLOW ADAS SCENE STARTED")
    print("===================================================")
    print(f" ROS2 camera topic : {args.topic}")
    print(f" Ego speed         : {args.ego_speed} km/h")
    print(f" Town              : {world.get_map().name}")
    print("")
    print("Kontrol için:")
    print(f"  ros2 topic hz {args.topic}")
    print("  ros2 topic echo /adas/decision")
    print("  rqt_image_view /adas/perception/annotated_image")
    print("")
    print("Çıkmak için CTRL+C")
    print("===================================================")
    print("")

    start_time = time.time()

    try:
        while rclpy.ok():
            apply_ego_lane_control(ego_vehicle, carla_map, args.ego_speed)
            set_spectator_follow(world, ego_vehicle)

            rclpy.spin_once(node, timeout_sec=0.001)

            if args.duration > 0.0:
                if time.time() - start_time >= args.duration:
                    break

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[EXIT] CTRL+C alındı.")

    finally:
        print("[CLEANUP] Stopping camera and destroying actors...")

        try:
            camera.stop()
        except RuntimeError:
            pass

        for actor in reversed(actors):
            try:
                if actor is not None and actor.is_alive:
                    if actor.type_id.startswith("controller.ai.walker"):
                        actor.stop()
                    actor.destroy()
            except RuntimeError:
                pass

        if args.preview and cv2 is not None:
            cv2.destroyAllWindows()

        node.destroy_node()
        rclpy.shutdown()

        print("[DONE] Scene closed cleanly.")


if __name__ == "__main__":
    main()
