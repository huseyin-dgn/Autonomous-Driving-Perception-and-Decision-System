#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import carla
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def make_transform(data):
    loc = data["location"]
    rot = data["rotation"]

    return carla.Transform(
        carla.Location(
            x=float(loc["x"]),
            y=float(loc["y"]),
            z=float(loc["z"]),
        ),
        carla.Rotation(
            pitch=float(rot["pitch"]),
            yaw=float(rot["yaw"]),
            roll=float(rot["roll"]),
        ),
    )


class CarlaTLOnlyPublisher(Node):
    def __init__(self, args):
        super().__init__("carla_tl_only_publisher")

        self.args = args
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, args.topic, 10)

        scene_path = Path(args.scene)
        if not scene_path.exists():
            raise RuntimeError(
                f"Scene JSON bulunamadı: {args.scene}\n"
                f"Önce scripts/carla_tl_only_scenario.py çalışmalı."
            )

        scene = json.loads(scene_path.read_text(encoding="utf-8"))

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()

        self.camera = None
        self.frame_count = 0
        self.last_log = time.time()

        self.spawn_camera(scene)

        self.get_logger().info(f"Publishing CARLA RGB camera to {args.topic}")
        self.get_logger().info(f"Camera scene file: {args.scene}")

    def destroy_old_test_sensors(self):
        for actor in self.world.get_actors():
            if actor.type_id.startswith("sensor.camera"):
                role_name = actor.attributes.get("role_name", "")
                if role_name == "adas_tl_only_camera":
                    try:
                        actor.destroy()
                    except Exception:
                        pass

    def spawn_camera(self, scene):
        self.destroy_old_test_sensors()

        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.args.width))
        bp.set_attribute("image_size_y", str(self.args.height))
        bp.set_attribute("fov", str(self.args.fov))
        bp.set_attribute("sensor_tick", str(1.0 / self.args.fps))

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "adas_tl_only_camera")

        cam_tf = make_transform(scene["camera_transform"])

        self.camera = self.world.spawn_actor(bp, cam_tf)
        self.camera.listen(self.on_image)

        self.world.get_spectator().set_transform(cam_tf)

    def on_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        bgr = array[:, :, :3].copy()

        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_tl_only_camera"

        self.pub.publish(msg)

        self.frame_count += 1
        now = time.time()

        if now - self.last_log > 2.0:
            self.get_logger().info(f"Published frames: {self.frame_count}")
            self.last_log = now

        if self.args.preview:
            cv2.imshow("CARLA TL ONLY PUBLISHER", bgr)
            cv2.waitKey(1)

    def destroy_node(self):
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
        except Exception:
            pass

        if self.args.preview:
            cv2.destroyAllWindows()

        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--scene", default="/tmp/adas_tl_only_scene.json")
    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = CarlaTLOnlyPublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
