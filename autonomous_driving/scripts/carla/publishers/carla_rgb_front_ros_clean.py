#!/usr/bin/env python3
import argparse
import time
import sys

import numpy as np

try:
    import carla
except Exception as exc:
    print(f"[ERROR] carla import edilemedi: {exc}")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def set_attr(bp, key, value):
    try:
        if bp.has_attribute(key):
            bp.set_attribute(key, str(value))
    except Exception:
        pass


def find_ego(world):
    vehicles = world.get_actors().filter("vehicle.*")

    for v in vehicles:
        try:
            if v.attributes.get("role_name", "") == "ego":
                return v
        except Exception:
            pass

    if len(vehicles) > 0:
        return vehicles[0]

    return None


def destroy_old_clean_cameras(world):
    count = 0

    for sensor in world.get_actors().filter("sensor.camera.rgb"):
        try:
            if sensor.attributes.get("role_name", "") == "adas_front_rgb_clean":
                sensor.destroy()
                count += 1
        except Exception:
            pass

    if count:
        print(f"[ADAS] Destroyed old clean cameras: {count}")


class CarlaRGBPublisher(Node):
    def __init__(self, args):
        super().__init__("carla_rgb_front_ros_clean")

        self.args = args
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, args.topic, 10)

        self.frame_count = 0
        self.last_report = time.time()

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()

        destroy_old_clean_cameras(self.world)

        self.ego = find_ego(self.world)
        if self.ego is None:
            raise RuntimeError("Ego vehicle bulunamadı. Önce scene scriptini çalıştır.")

        self.camera = self.spawn_camera()

        self.camera.listen(self.on_image)

        self.get_logger().info("CARLA RGB front clean publisher başladı")
        self.get_logger().info(f"Topic: {args.topic}")
        self.get_logger().info(f"Resolution: {args.width}x{args.height}")
        self.get_logger().info(f"FOV: {args.fov}")
        self.get_logger().info(f"Ego: {self.ego.id}")

    def spawn_camera(self):
        bps = self.world.get_blueprint_library()
        bp = bps.find("sensor.camera.rgb")

        set_attr(bp, "role_name", "adas_front_rgb_clean")
        set_attr(bp, "image_size_x", self.args.width)
        set_attr(bp, "image_size_y", self.args.height)
        set_attr(bp, "fov", self.args.fov)
        set_attr(bp, "sensor_tick", self.args.sensor_tick)

        set_attr(bp, "gamma", "2.2")
        set_attr(bp, "enable_postprocess_effects", "true")
        set_attr(bp, "exposure_mode", "manual")
        set_attr(bp, "exposure_compensation", "0.0")
        set_attr(bp, "shutter_speed", "120.0")
        set_attr(bp, "iso", "100.0")
        set_attr(bp, "fstop", "5.6")
        set_attr(bp, "bloom_intensity", "0.0")
        set_attr(bp, "lens_flare_intensity", "0.0")
        set_attr(bp, "motion_blur_intensity", "0.0")

        transform = carla.Transform(
            carla.Location(x=1.60, y=0.0, z=1.65),
            carla.Rotation(pitch=-3.0, yaw=0.0, roll=0.0),
        )

        camera = self.world.spawn_actor(
            bp,
            transform,
            attach_to=self.ego,
        )

        return camera

    def on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))

        bgr = arr[:, :, :3].copy()

        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_front_rgb"

        self.pub.publish(msg)

        self.frame_count += 1
        now = time.time()

        if now - self.last_report >= 2.0:
            fps = self.frame_count / max(0.001, now - self.last_report)
            self.get_logger().info(f"Yayın aktif: {fps:.1f} FPS")
            self.frame_count = 0
            self.last_report = now

    def destroy(self):
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--sensor-tick", type=float, default=0.05)
    args = parser.parse_args()

    rclpy.init()

    node = None

    try:
        node = CarlaRGBPublisher(args)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        print(f"[ERROR] {exc}")

    finally:
        if node is not None:
            node.destroy()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
