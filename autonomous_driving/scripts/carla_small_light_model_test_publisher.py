#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
import traceback

import numpy as np

try:
    import carla
except Exception:
    carla_paths = [
        os.path.expanduser("~/CARLA_DISK/PythonAPI/carla"),
        os.path.expanduser("~/CARLA_DISK/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg"),
        os.path.expanduser("~/CARLA_DISK/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg"),
    ]
    for p in carla_paths:
        if os.path.exists(p):
            sys.path.append(p)
    import carla

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


ROLE_PREFIX = "adas_tltest"


def get_blueprint(bp_lib, candidates):
    for pattern in candidates:
        found = bp_lib.filter(pattern)
        if found:
            return found[0]
    return None


def forward_vec(yaw_deg):
    yaw = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def right_vec(yaw_deg):
    yaw = math.radians(yaw_deg + 90.0)
    return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


def shifted_transform(base_tf, forward=0.0, right=0.0, z=0.25, yaw_offset=0.0):
    f = forward_vec(base_tf.rotation.yaw)
    r = right_vec(base_tf.rotation.yaw)

    loc = carla.Location(
        x=base_tf.location.x + f.x * forward + r.x * right,
        y=base_tf.location.y + f.y * forward + r.y * right,
        z=base_tf.location.z + z,
    )

    rot = carla.Rotation(
        pitch=0.0,
        yaw=base_tf.rotation.yaw + yaw_offset,
        roll=0.0,
    )

    return carla.Transform(loc, rot)


def set_role(bp, role_name):
    if bp is not None and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)


def try_spawn(world, bp, tf, name, physics=False):
    if bp is None:
        print(f"[WARN] Blueprint yok: {name}")
        return None

    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        print(f"[WARN] Spawn başarısız: {name}")
        return None

    try:
        actor.set_simulate_physics(bool(physics))
    except Exception:
        pass

    print(f"[SPAWN] {name}: id={actor.id} type={actor.type_id}")
    return actor


def destroy_dynamic_actors(world):
    actors = world.get_actors()
    to_destroy = []

    for a in actors:
        tid = a.type_id
        role = a.attributes.get("role_name", "")

        if tid.startswith("sensor."):
            to_destroy.append(a)
        elif tid.startswith("vehicle."):
            to_destroy.append(a)
        elif tid.startswith("walker."):
            to_destroy.append(a)
        elif role.startswith(ROLE_PREFIX):
            to_destroy.append(a)

    for a in to_destroy:
        try:
            a.destroy()
            print(f"[DESTROY] id={a.id} type={a.type_id}")
        except Exception:
            pass


def get_stop_waypoint(traffic_light, carla_map):
    try:
        stop_wps = traffic_light.get_stop_waypoints()
        if stop_wps:
            return stop_wps[0]
    except Exception:
        pass

    try:
        return carla_map.get_waypoint(
            traffic_light.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        return None


def set_light_state(traffic_light, state_name):
    if traffic_light is None:
        return

    state_name = str(state_name).lower().strip()

    if state_name == "green":
        state = carla.TrafficLightState.Green
    elif state_name == "yellow":
        state = carla.TrafficLightState.Yellow
    else:
        state = carla.TrafficLightState.Red

    try:
        traffic_light.set_state(state)
        traffic_light.set_red_time(999.0)
        traffic_light.set_yellow_time(999.0)
        traffic_light.set_green_time(999.0)
    except Exception:
        pass

    try:
        traffic_light.freeze(True)
    except Exception:
        pass


class CarlaSmallScenePublisher(Node):
    def __init__(self, args):
        super().__init__("carla_small_light_model_test_publisher")

        self.args = args
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, args.topic, 10)

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(30.0)

        self.world = self.client.get_world()

        if args.town:
            current_map = self.world.get_map().name
            if args.town not in current_map:
                print(f"[LOAD] {args.town} yükleniyor...")
                self.world = self.client.load_world(args.town)
                time.sleep(3.0)

        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        try:
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        except Exception:
            pass

        destroy_dynamic_actors(self.world)

        self.actors = []
        self.selected_light = None
        self.camera = None

        self.spawn_scene()
        self.create_timer(0.5, self.keep_scene_alive)

        self.get_logger().info(f"Publisher başladı: {args.topic}")
        self.get_logger().info("Sahne: 1 araç + 1 insan + 3 motor + trafik ışığı + trafik levhası")

    def choose_traffic_light_and_base(self):
        traffic_lights = list(self.world.get_actors().filter("traffic.traffic_light*"))
        print(f"[INFO] Traffic light count: {len(traffic_lights)}")

        for tl in traffic_lights:
            stop_wp = get_stop_waypoint(tl, self.carla_map)
            if stop_wp is None:
                continue

            prev = stop_wp.previous(28.0)
            if not prev:
                continue

            ego_wp = prev[0]
            ego_tf = ego_wp.transform
            ego_tf.location.z += 0.30

            return tl, ego_tf

        spawn_points = self.carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Map üzerinde spawn point yok.")

        return None, spawn_points[0]

    def spawn_scene(self):
        bp_lib = self.world.get_blueprint_library()

        ego_bp = get_blueprint(bp_lib, [
            "vehicle.tesla.model3",
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2017",
            "vehicle.*",
        ])
        front_car_bp = get_blueprint(bp_lib, [
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2017",
            "vehicle.tesla.model3",
            "vehicle.*",
        ])
        motor_bp_1 = get_blueprint(bp_lib, [
            "vehicle.yamaha.yzf",
            "vehicle.kawasaki.ninja",
            "vehicle.harley-davidson.low_rider",
            "vehicle.*",
        ])
        motor_bp_2 = get_blueprint(bp_lib, [
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
            "vehicle.harley-davidson.low_rider",
            "vehicle.*",
        ])
        motor_bp_3 = get_blueprint(bp_lib, [
            "vehicle.harley-davidson.low_rider",
            "vehicle.yamaha.yzf",
            "vehicle.kawasaki.ninja",
            "vehicle.*",
        ])
        walker_bp = get_blueprint(bp_lib, [
            "walker.pedestrian.0001",
            "walker.pedestrian.0002",
            "walker.pedestrian.*",
        ])
        sign_bp = get_blueprint(bp_lib, [
            "static.prop.trafficwarning",
            "static.prop.streetsign04",
            "static.prop.streetsign01",
            "static.prop.streetsign",
        ])

        set_role(ego_bp, f"{ROLE_PREFIX}_ego")
        set_role(front_car_bp, f"{ROLE_PREFIX}_front_vehicle")
        set_role(motor_bp_1, f"{ROLE_PREFIX}_motor_left")
        set_role(motor_bp_2, f"{ROLE_PREFIX}_motor_right")
        set_role(motor_bp_3, f"{ROLE_PREFIX}_motor_far_left")
        set_role(walker_bp, f"{ROLE_PREFIX}_person_right")
        set_role(sign_bp, f"{ROLE_PREFIX}_traffic_sign")

        self.selected_light, ego_tf = self.choose_traffic_light_and_base()
        set_light_state(self.selected_light, self.args.light)

        ego = try_spawn(self.world, ego_bp, ego_tf, "ego", physics=False)
        if ego is None:
            raise RuntimeError("Ego araç spawn edilemedi.")

        self.actors.append(ego)

        front_car_tf = shifted_transform(ego_tf, forward=18.0, right=-1.8, z=0.25, yaw_offset=0.0)
        motor_left_tf = shifted_transform(ego_tf, forward=12.0, right=-4.2, z=0.25, yaw_offset=0.0)
        motor_right_tf = shifted_transform(ego_tf, forward=16.5, right=4.1, z=0.25, yaw_offset=0.0)
        motor_far_left_tf = shifted_transform(ego_tf, forward=25.0, right=-5.2, z=0.25, yaw_offset=0.0)
        person_tf = shifted_transform(ego_tf, forward=20.0, right=5.8, z=0.45, yaw_offset=180.0)
        sign_tf = shifted_transform(ego_tf, forward=27.0, right=6.3, z=0.15, yaw_offset=180.0)

        for actor in [
            try_spawn(self.world, front_car_bp, front_car_tf, "front_vehicle", physics=False),
            try_spawn(self.world, motor_bp_1, motor_left_tf, "motor_left", physics=False),
            try_spawn(self.world, motor_bp_2, motor_right_tf, "motor_right", physics=False),
            try_spawn(self.world, motor_bp_3, motor_far_left_tf, "motor_far_left", physics=False),
            try_spawn(self.world, walker_bp, person_tf, "person_right", physics=False),
            try_spawn(self.world, sign_bp, sign_tf, "traffic_sign_right", physics=False),
        ]:
            if actor is not None:
                self.actors.append(actor)

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.args.width))
        cam_bp.set_attribute("image_size_y", str(self.args.height))
        cam_bp.set_attribute("fov", str(self.args.fov))
        cam_bp.set_attribute("sensor_tick", str(self.args.sensor_tick))
        if cam_bp.has_attribute("role_name"):
            cam_bp.set_attribute("role_name", f"{ROLE_PREFIX}_front_camera")

        cam_tf = carla.Transform(
            carla.Location(x=1.70, y=0.0, z=1.65),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )

        self.camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
        self.actors.append(self.camera)
        self.camera.listen(self.on_camera_image)

        spectator = self.world.get_spectator()
        spectator_tf = shifted_transform(ego_tf, forward=-7.0, right=0.0, z=5.5, yaw_offset=0.0)
        spectator_tf.rotation.pitch = -18.0
        spectator.set_transform(spectator_tf)

        if self.selected_light is not None:
            print(f"[LIGHT] selected id={self.selected_light.id} state={self.args.light}")
        else:
            print("[WARN] Gerçek traffic light bulunamadı; sahne yine yayın yapacak.")

    def keep_scene_alive(self):
        set_light_state(self.selected_light, self.args.light)

    def on_camera_image(self, image):
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            bgr = arr[:, :, :3].copy()

            msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "carla_front_camera"
            self.pub.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"camera publish hata: {exc}")

    def destroy(self):
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass

        for actor in reversed(self.actors):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass

        try:
            if self.selected_light is not None:
                self.selected_light.freeze(False)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--light", default="red", choices=["red", "yellow", "green"])
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--sensor-tick", type=float, default=0.05)
    args = parser.parse_args()

    rclpy.init()
    node = None

    try:
        node = CarlaSmallScenePublisher(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
