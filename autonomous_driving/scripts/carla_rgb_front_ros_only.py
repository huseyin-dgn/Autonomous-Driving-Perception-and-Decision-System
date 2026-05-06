#!/usr/bin/env python3
import argparse
import time
import threading

import carla
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CarlaRGBFrontROSOnly(Node):
    def __init__(self, args):
        super().__init__("carla_rgb_front_ros_only")

        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.timeout)
        self.world = self.client.get_world()

        self.publisher = self.create_publisher(
            Image,
            "/adas/camera/front/image_raw",
            10,
        )

        self.camera = None
        self.frame_count = 0
        self.last_log_time = time.time()
        self.last_frame_count = 0
        self.lock = threading.Lock()

        self.ego = self.find_ego_vehicle()

        if self.ego is None:
            raise RuntimeError(
                "Ego araç bulunamadı. Önce CARLA sahne/spawn scriptini çalıştır."
            )

        self.destroy_old_rgb_front_cameras()
        self.spawn_rgb_front_camera()

        self.timer = self.create_timer(2.0, self.log_fps)

        self.get_logger().info("carla_rgb_front_ros_only başladı")
        self.get_logger().info(f"Topic: /adas/camera/front/image_raw")
        self.get_logger().info(f"Camera: {args.width}x{args.height}, fov={args.fov}")
        self.get_logger().info(f"Ego: id={self.ego.id}, type={self.ego.type_id}")

    def find_ego_vehicle(self):
        vehicles = self.world.get_actors().filter("vehicle.*")

        for actor in vehicles:
            role_name = actor.attributes.get("role_name", "")
            if role_name == "ego":
                return actor

        if len(vehicles) > 0:
            return vehicles[0]

        return None

    def destroy_old_rgb_front_cameras(self):
        destroyed = 0

        for actor in self.world.get_actors().filter("sensor.camera.rgb"):
            role_name = actor.attributes.get("role_name", "")
            if role_name == "rgb_front":
                try:
                    actor.stop()
                except Exception:
                    pass

                try:
                    actor.destroy()
                    destroyed += 1
                except Exception:
                    pass

        if destroyed > 0:
            self.get_logger().info(f"Eski rgb_front kamera temizlendi: {destroyed}")

    def spawn_rgb_front_camera(self):
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")

        cam_bp.set_attribute("role_name", "rgb_front")
        cam_bp.set_attribute("image_size_x", str(self.args.width))
        cam_bp.set_attribute("image_size_y", str(self.args.height))
        cam_bp.set_attribute("fov", str(self.args.fov))
        cam_bp.set_attribute("sensor_tick", str(self.args.sensor_tick))

        transform = carla.Transform(
            carla.Location(
                x=self.args.camera_x,
                y=self.args.camera_y,
                z=self.args.camera_z,
            ),
            carla.Rotation(
                pitch=self.args.camera_pitch,
                yaw=self.args.camera_yaw,
                roll=self.args.camera_roll,
            ),
        )

        self.camera = self.world.spawn_actor(
            cam_bp,
            transform,
            attach_to=self.ego,
        )

        self.camera.listen(self.on_image)

        self.get_logger().info(
            f"rgb_front kamera spawn edildi: id={self.camera.id}, type={self.camera.type_id}"
        )

    def on_image(self, carla_image):
        try:
            array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            array = array.reshape((carla_image.height, carla_image.width, 4))

            bgr = array[:, :, :3]
            bgr = np.ascontiguousarray(bgr)

            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "carla_rgb_front"

            msg.height = int(carla_image.height)
            msg.width = int(carla_image.width)
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = int(carla_image.width * 3)
            msg.data = bgr.tobytes()

            self.publisher.publish(msg)

            with self.lock:
                self.frame_count += 1

        except Exception as exc:
            self.get_logger().error(f"CARLA image publish hata: {exc}")

    def log_fps(self):
        now = time.time()

        with self.lock:
            frames = self.frame_count

        dt = max(1e-6, now - self.last_log_time)
        fps = (frames - self.last_frame_count) / dt

        self.last_log_time = now
        self.last_frame_count = frames

        self.get_logger().info(f"Yayın aktif: {fps:.1f} FPS")

    def cleanup(self):
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass

            try:
                self.camera.destroy()
                self.get_logger().info("rgb_front kamera temizlendi")
            except Exception:
                pass

            self.camera = None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)

    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--sensor-tick", type=float, default=0.05)

    parser.add_argument("--camera-x", type=float, default=1.8)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=1.45)
    parser.add_argument("--camera-pitch", type=float, default=-4.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = None

    try:
        node = CarlaRGBFrontROSOnly(args)
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            try:
                node.cleanup()
            except Exception:
                pass

            try:
                node.destroy_node()
            except Exception:
                pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
