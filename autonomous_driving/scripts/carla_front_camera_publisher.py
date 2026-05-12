#!/usr/bin/env python3
import argparse
import os
import sys
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


ROLE_PREFIX = "adas_small_test"


class CarlaFrontCameraPublisher(Node):
    def __init__(self, args):
        super().__init__("carla_front_camera_publisher")

        self.args = args
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, args.topic, 10)

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()

        self.ego = self.find_ego()
        if self.ego is None:
            raise RuntimeError(
                "Ego araç bulunamadı. Önce şu komutu çalıştır:\n"
                "python3 scripts/carla_small_static_scene.py --light red"
            )

        self.camera = self.spawn_camera()

        self.get_logger().info("CARLA front camera publisher başladı.")
        self.get_logger().info(f"topic={args.topic}")
        self.get_logger().info(f"ego_id={self.ego.id} ego_type={self.ego.type_id}")

    def find_ego(self):
        vehicles = self.world.get_actors().filter("vehicle.*")

        for actor in vehicles:
            role = actor.attributes.get("role_name", "")
            if role == f"{ROLE_PREFIX}_ego":
                return actor

        for actor in vehicles:
            role = actor.attributes.get("role_name", "")
            if "ego" in role:
                return actor

        return None

    def spawn_camera(self):
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")

        cam_bp.set_attribute("image_size_x", str(self.args.width))
        cam_bp.set_attribute("image_size_y", str(self.args.height))
        cam_bp.set_attribute("fov", str(self.args.fov))
        cam_bp.set_attribute("sensor_tick", str(self.args.sensor_tick))

        if cam_bp.has_attribute("role_name"):
            cam_bp.set_attribute("role_name", f"{ROLE_PREFIX}_front_camera")

        cam_tf = carla.Transform(
            carla.Location(x=1.65, y=0.0, z=1.55),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )

        camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.ego)
        camera.listen(self.on_image)
        return camera

    def on_image(self, image):
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            bgr = arr[:, :, :3].copy()

            msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "carla_front_camera"

            self.pub.publish(msg)

        except Exception as exc:
            self.get_logger().error(f"publish hata: {exc}")

    def cleanup(self):
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
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--sensor-tick", type=float, default=0.05)
    args = parser.parse_args()

    rclpy.init()
    node = None

    try:
        node = CarlaFrontCameraPublisher(args)
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception:
        traceback.print_exc()

    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
