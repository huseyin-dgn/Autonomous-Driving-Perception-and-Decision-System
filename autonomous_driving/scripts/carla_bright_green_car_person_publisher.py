#!/usr/bin/env python3
import argparse
import random
import time
import numpy as np
import carla

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def destroy_old_scene(world):
    for actor in world.get_actors():
        try:
            if (
                actor.type_id.startswith("vehicle.")
                or actor.type_id.startswith("walker.")
                or actor.type_id.startswith("controller.ai.walker")
                or actor.type_id.startswith("sensor.camera")
            ):
                actor.destroy()
        except Exception:
            pass


def set_bright_weather(world):
    weather = carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=45.0,
        sun_altitude_angle=80.0,
        fog_density=0.0,
        fog_distance=100000.0,
        fog_falloff=0.0,
        wetness=0.0,
        scattering_intensity=0.0,
        mie_scattering_scale=0.0,
        rayleigh_scattering_scale=0.0,
    )
    world.set_weather(weather)


class BrightGreenCarPersonPublisher(Node):
    def __init__(self, args):
        super().__init__("carla_bright_green_car_person_publisher")

        self.args = args
        self.bridge = CvBridge()
        self.latest = None

        self.pub = self.create_publisher(Image, "/adas/camera/front/image_raw", 10)

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()

        self.actors = []

        destroy_old_scene(self.world)
        set_bright_weather(self.world)
        self.setup_scene()

        self.timer = self.create_timer(0.05, self.publish_image)

    def setup_scene(self):
        tls = list(self.world.get_actors().filter("traffic.traffic_light*"))

        if not tls:
            raise RuntimeError("Bu CARLA haritasında traffic light actor bulunamadı.")

        target_tl = tls[self.args.tl_index % len(tls)]
        target_tl.set_state(carla.TrafficLightState.Green)
        target_tl.freeze(True)

        stop_wps = target_tl.get_stop_waypoints()

        if not stop_wps:
            raise RuntimeError("Bu traffic light için stop waypoint yok. Başka --tl-index dene.")

        stop_wp = stop_wps[0]

        ego_prev = stop_wp.previous(self.args.ego_distance)
        if not ego_prev:
            raise RuntimeError("Ego spawn waypoint bulunamadı. Başka --tl-index dene.")

        ego_wp = ego_prev[0]
        ego_tf = ego_wp.transform
        ego_tf.location.z += 0.35

        ego_bp = self.bp_lib.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("role_name", "ego_vehicle")

        ego = self.world.try_spawn_actor(ego_bp, ego_tf)
        if ego is None:
            ego_tf.location.z += 1.0
            ego = self.world.spawn_actor(ego_bp, ego_tf)

        ego.set_autopilot(False)
        self.actors.append(ego)

        # Kamera
        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.args.width))
        cam_bp.set_attribute("image_size_y", str(self.args.height))
        cam_bp.set_attribute("fov", str(self.args.fov))

        # Daha aydınlık / daha az karanlık görünüm için exposure ayarları
        for attr, val in [
            ("exposure_mode", "manual"),
            ("shutter_speed", "200"),
            ("iso", "100"),
            ("gamma", "2.2"),
            ("fstop", "2.8"),
        ]:
            if cam_bp.has_attribute(attr):
                cam_bp.set_attribute(attr, val)

        cam_tf = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.7),
            carla.Rotation(pitch=self.args.pitch, yaw=0.0, roll=0.0)
        )

        camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
        camera.listen(self.camera_callback)
        self.actors.append(camera)

        # Araç: ego'nun ön-sağ tarafına koy
        vehicle_spawn_wp = stop_wp.previous(self.args.vehicle_distance)
        if vehicle_spawn_wp:
            car_wp = vehicle_spawn_wp[0]
            car_tf = car_wp.transform

            right_vec = car_tf.get_right_vector()
            car_tf.location.x += right_vec.x * self.args.vehicle_side_offset
            car_tf.location.y += right_vec.y * self.args.vehicle_side_offset
            car_tf.location.z += 0.35

            car_bp_candidates = [
                "vehicle.audi.tt",
                "vehicle.lincoln.mkz_2020",
                "vehicle.dodge.charger_2020",
                "vehicle.tesla.model3",
            ]

            car_bp = None
            for name in car_bp_candidates:
                found = self.bp_lib.filter(name)
                if found:
                    car_bp = found[0]
                    break

            if car_bp is None:
                car_bp = self.bp_lib.filter("vehicle.*")[0]

            car = self.world.try_spawn_actor(car_bp, car_tf)
            if car is not None:
                car.set_autopilot(False)
                self.actors.append(car)

        # İnsan: sağ kaldırım tarafına koy
        ped_wp = stop_wp.previous(self.args.person_distance)
        if ped_wp:
            p_wp = ped_wp[0]
            p_tf = p_wp.transform

            right_vec = p_tf.get_right_vector()
            p_tf.location.x += right_vec.x * self.args.person_side_offset
            p_tf.location.y += right_vec.y * self.args.person_side_offset
            p_tf.location.z += 0.30
            p_tf.rotation.yaw += 180.0

            walker_bps = self.bp_lib.filter("walker.pedestrian.*")
            walker_bp = random.choice(walker_bps)

            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            walker = self.world.try_spawn_actor(walker_bp, p_tf)
            if walker is not None:
                self.actors.append(walker)

        self.get_logger().info("BRIGHT GREEN LIGHT + CAR + PERSON SCENE READY")
        self.get_logger().info(f"Traffic light id    : {target_tl.id}")
        self.get_logger().info("Traffic light state : GREEN")
        self.get_logger().info("Weather             : BRIGHT CLEAR NOON")
        self.get_logger().info("Publishing          : /adas/camera/front/image_raw")

    def camera_callback(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = arr[:, :, :3].copy()
        self.latest = bgr

    def publish_image(self):
        if self.latest is None:
            return

        msg = self.bridge.cv2_to_imgmsg(self.latest, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "front_camera"
        self.pub.publish(msg)

    def destroy_node(self):
        for actor in reversed(self.actors):
            try:
                if actor is not None:
                    if actor.type_id.startswith("sensor.camera"):
                        actor.stop()
                    actor.destroy()
            except Exception:
                pass

        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)

    parser.add_argument("--tl-index", type=int, default=0)

    parser.add_argument("--ego-distance", type=float, default=12.0)
    parser.add_argument("--vehicle-distance", type=float, default=7.5)
    parser.add_argument("--vehicle-side-offset", type=float, default=3.2)

    parser.add_argument("--person-distance", type=float, default=8.5)
    parser.add_argument("--person-side-offset", type=float, default=5.8)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument("--pitch", type=float, default=-4.0)

    args = parser.parse_args()

    rclpy.init()
    node = BrightGreenCarPersonPublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
