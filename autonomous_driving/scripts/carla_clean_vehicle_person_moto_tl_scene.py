#!/usr/bin/env python3
import argparse
import math
import queue
import random
import time

import carla
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def destroy_dynamic(world):
    count = 0
    for a in world.get_actors():
        if (
            a.type_id.startswith("vehicle.")
            or a.type_id.startswith("walker.")
            or a.type_id.startswith("sensor.")
            or a.type_id.startswith("controller.ai.walker")
        ):
            try:
                a.destroy()
                count += 1
            except Exception:
                pass
    print(f"[SCENE] destroyed dynamic actors: {count}")


def set_weather(world):
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=65.0,
            sun_azimuth_angle=20.0,
            fog_density=0.0,
            wetness=0.0,
        )
    )


def forward_right_from_yaw(yaw_deg):
    yaw = math.radians(yaw_deg)
    forward = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
    right = carla.Vector3D(math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0), 0.0)
    return forward, right


def location_from_camera(cam_tf, forward_dist, right_dist, z_add=0.2):
    fwd, right = forward_right_from_yaw(cam_tf.rotation.yaw)
    return carla.Location(
        x=cam_tf.location.x + fwd.x * forward_dist + right.x * right_dist,
        y=cam_tf.location.y + fwd.y * forward_dist + right.y * right_dist,
        z=cam_tf.location.z + z_add,
    )


def ground_transform(world, loc, yaw):
    try:
        wp = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        wp = None

    if wp is not None:
        out_loc = carla.Location(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z + 0.25)
    else:
        out_loc = carla.Location(loc.x, loc.y, loc.z + 0.25)

    return carla.Transform(out_loc, carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))


def find_vehicle_bp(bp_lib):
    preferred = [
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.lincoln.mkz_2020",
        "vehicle.dodge.charger_2020",
    ]

    for tid in preferred:
        try:
            bp = bp_lib.find(tid)
            if bp is not None:
                return bp
        except Exception:
            pass

    cars = list(bp_lib.filter("vehicle.*"))
    cars = [bp for bp in cars if not any(x in bp.id.lower() for x in ["bike", "crossbike", "gazelle", "diamondback"])]
    return random.choice(cars)


def find_motorcycle_bp(bp_lib):
    preferred = [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ]

    for tid in preferred:
        try:
            bp = bp_lib.find(tid)
            if bp is not None:
                print(f"[SCENE] selected motorcycle blueprint: {bp.id}")
                return bp
        except Exception:
            pass

    candidates = []
    blocked = ["crossbike", "diamondback", "gazelle", "omafiets", "bicycle", "century"]

    for bp in bp_lib.filter("vehicle.*"):
        tid = bp.id.lower()

        if any(b in tid for b in blocked):
            continue

        if any(k in tid for k in ["kawasaki", "yamaha", "harley", "vespa", "ninja", "yzf", "low_rider", "zx125"]):
            candidates.append(bp)

    if not candidates:
        raise RuntimeError("Gerçek motorcycle blueprint bulunamadı.")

    candidates.sort(key=lambda b: b.id)
    print(f"[SCENE] selected motorcycle blueprint: {candidates[0].id}")
    return candidates[0]


def prepare_bp(bp, role_name, color=None):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)

    if color and bp.has_attribute("color"):
        values = bp.get_attribute("color").recommended_values
        if color in values:
            bp.set_attribute("color", color)
        elif values:
            bp.set_attribute("color", values[0])

    return bp


def try_spawn(world, bp, tf, name):
    offsets = [
        (0.0, 0.0),
        (0.8, 0.0),
        (-0.8, 0.0),
        (0.0, 0.8),
        (0.0, -0.8),
        (1.2, 0.5),
        (-1.2, -0.5),
    ]

    yaw = math.radians(tf.rotation.yaw)
    fwd = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
    right = carla.Vector3D(math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0), 0.0)

    for fo, ro in offsets:
        loc = carla.Location(
            x=tf.location.x + fwd.x * fo + right.x * ro,
            y=tf.location.y + fwd.y * fo + right.y * ro,
            z=tf.location.z,
        )

        ntf = carla.Transform(loc, tf.rotation)
        actor = world.try_spawn_actor(bp, ntf)

        if actor is not None:
            print(f"[SCENE] spawned {name}: id={actor.id}, type={actor.type_id}, loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f})")
            return actor

    raise RuntimeError(f"{name} spawn edilemedi.")


def draw_test_traffic_light(img, x, y, state):
    # Kritik düzeltme:
    # Gövde/çerçeve artık sarı değil.
    # Sadece aktif lens renkli. Böylece classifier sarı gövdeye takılmaz.

    panel_w = 86
    panel_h = 190

    # Dış gövde: nötr gri
    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (45, 45, 45), -1)
    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (220, 220, 220), 2)

    # İç panel: siyah
    cv2.rectangle(img, (x + 10, y + 10), (x + panel_w - 10, y + panel_h - 10), (5, 5, 5), -1)

    centers = {
        "red": (x + panel_w // 2, y + 42),
        "yellow": (x + panel_w // 2, y + 95),
        "green": (x + panel_w // 2, y + 148),
    }

    # Sönük lensler
    for c in centers.values():
        cv2.circle(img, c, 19, (10, 10, 10), -1)
        cv2.circle(img, c, 19, (75, 75, 75), 2)

    colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
    }

    state = state.lower().strip()
    if state not in colors:
        state = "yellow"

    color = colors[state]
    active = centers[state]

    # Aktif lens: büyük ve net
    cv2.circle(img, active, 22, color, -1)
    cv2.circle(img, active, 31, color, 3)

    # Hafif glow
    overlay = img.copy()
    cv2.circle(overlay, active, 42, color, -1)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # Pole
    pole_x = x + panel_w // 2
    cv2.rectangle(img, (pole_x - 5, y + panel_h), (pole_x + 5, y + panel_h + 115), (55, 55, 55), -1)

    # Görsel doğrulama etiketi. Model kararına güvenmek için değil, bizim gözümüz için.
    cv2.putText(
        img,
        state.upper(),
        (x - 5, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def force_native_traffic_lights(world, state_name, quiet=False):
    state_name = str(state_name).lower().strip()

    state_map = {
        "red": carla.TrafficLightState.Red,
        "yellow": carla.TrafficLightState.Yellow,
        "green": carla.TrafficLightState.Green,
    }

    if state_name not in state_map:
        state_name = "yellow"

    target_state = state_map[state_name]
    lights = list(world.get_actors().filter("traffic.traffic_light*"))

    if not quiet:
        print(f"[SCENE] native CARLA traffic lights found: {len(lights)}")
        print(f"[SCENE] forcing native traffic lights to: {state_name.upper()}")

    try:
        world.freeze_all_traffic_lights(False)
    except Exception:
        pass

    for tl in lights:
        try:
            tl.set_state(target_state)
            tl.set_red_time(9999.0)
            tl.set_yellow_time(9999.0)
            tl.set_green_time(9999.0)
            tl.freeze(True)
        except Exception as e:
            if not quiet:
                print(f"[SCENE] traffic light force error id={getattr(tl, 'id', '?')}: {e}")

    try:
        world.freeze_all_traffic_lights(True)
    except Exception:
        pass

    return len(lights)


class CleanADASScene(Node):
    def __init__(self, args):
        super().__init__("carla_clean_adas_scene")

        self.args = args
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, args.topic, 10)
        self.frame_count = 0
        self.actors = []
        self.camera = None

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(30.0)

        if args.reload_map:
            self.world = self.client.load_world(args.map)
            time.sleep(3.0)
        else:
            self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        set_weather(self.world)

        if args.destroy_dynamic:
            destroy_dynamic(self.world)

        force_native_traffic_lights(self.world, self.args.tl_state, quiet=False)

        self.setup_scene()

    def setup_scene(self):
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            raise RuntimeError("Map spawn point yok.")

        idx = max(0, min(self.args.spawn_index, len(spawn_points) - 1))
        base_tf = spawn_points[idx]

        cam_loc = carla.Location(
            x=base_tf.location.x,
            y=base_tf.location.y,
            z=base_tf.location.z + self.args.camera_z,
        )

        cam_tf = carla.Transform(
            cam_loc,
            carla.Rotation(
                pitch=self.args.camera_pitch,
                yaw=base_tf.rotation.yaw + self.args.yaw_offset,
                roll=0.0,
            ),
        )

        self.world.get_spectator().set_transform(cam_tf)

        actor_yaw = cam_tf.rotation.yaw + 180.0

        car_bp = prepare_bp(find_vehicle_bp(bp_lib), "adas_clean_vehicle", "0,0,255")
        moto_bp = prepare_bp(find_motorcycle_bp(bp_lib), "adas_clean_motorcycle", "0,0,255")

        walker_candidates = list(bp_lib.filter("walker.pedestrian.*"))
        if not walker_candidates:
            raise RuntimeError("walker.pedestrian.* blueprint bulunamadı.")

        walker_bp = random.choice(walker_candidates)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        car_loc = location_from_camera(cam_tf, self.args.car_depth, self.args.car_right, 0.0)
        moto_loc = location_from_camera(cam_tf, self.args.moto_depth, self.args.moto_right, 0.0)
        ped_loc = location_from_camera(cam_tf, self.args.ped_depth, self.args.ped_right, 0.0)

        car_tf = ground_transform(self.world, car_loc, actor_yaw)
        moto_tf = ground_transform(self.world, moto_loc, actor_yaw)
        ped_tf = ground_transform(self.world, ped_loc, actor_yaw)

        car = try_spawn(self.world, car_bp, car_tf, "vehicle")
        moto = try_spawn(self.world, moto_bp, moto_tf, "motorcycle")
        ped = try_spawn(self.world, walker_bp, ped_tf, "person")

        self.actors.extend([car, moto, ped])

        for a in self.actors:
            try:
                if a.type_id.startswith("vehicle."):
                    a.set_autopilot(False)
                    a.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    a.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            except Exception:
                pass

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.args.width))
        cam_bp.set_attribute("image_size_y", str(self.args.height))
        cam_bp.set_attribute("fov", str(self.args.fov))
        cam_bp.set_attribute("sensor_tick", str(1.0 / self.args.fps))

        self.camera = self.world.spawn_actor(cam_bp, cam_tf)
        self.actors.append(self.camera)

        self.camera.listen(self.on_image)

        print("===================================================")
        print("[SCENE] CLEAN ADAS SCENE READY")
        print(f"[SCENE] spawn_index : {self.args.spawn_index}")
        print(f"[SCENE] camera fov  : {self.args.fov}")
        print(f"[SCENE] tl_state    : {self.args.tl_state}")
        print(f"[SCENE] topic       : {self.args.topic}")
        print("[SCENE] actors      : vehicle + person + REAL motorcycle")
        print("===================================================")

    def on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        frame = arr[:, :, :3].copy()

        # Sahte 2D trafik ışığı çizilmiyor.
        # Trafik ışığı olarak sadece CARLA'nın gerçek traffic_light aktörleri kullanılacak.
        if self.frame_count % max(1, int(self.args.fps)) == 0:
            force_native_traffic_lights(self.world, self.args.tl_state, quiet=True)

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_clean_adas_scene"
        self.pub.publish(msg)

        self.frame_count += 1
        if self.frame_count % int(self.args.fps * 2) == 0:
            self.get_logger().info(f"published frames={self.frame_count}")

    def destroy(self):
        for a in reversed(self.actors):
            try:
                a.destroy()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--reload-map", action="store_true")
    parser.add_argument("--destroy-dynamic", action="store_true")

    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=80.0)
    parser.add_argument("--fps", type=float, default=20.0)

    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--yaw-offset", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=2.0)
    parser.add_argument("--camera-pitch", type=float, default=-4.0)

    parser.add_argument("--car-depth", type=float, default=14.0)
    parser.add_argument("--car-right", type=float, default=0.0)

    parser.add_argument("--moto-depth", type=float, default=9.0)
    parser.add_argument("--moto-right", type=float, default=-3.0)

    parser.add_argument("--ped-depth", type=float, default=7.5)
    parser.add_argument("--ped-right", type=float, default=3.0)

    parser.add_argument("--tl-state", choices=["red", "yellow", "green"], default="yellow")
    parser.add_argument("--tl-x", type=int, default=760)
    parser.add_argument("--tl-y", type=int, default=165)

    args = parser.parse_args()

    rclpy.init()
    node = CleanADASScene(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
