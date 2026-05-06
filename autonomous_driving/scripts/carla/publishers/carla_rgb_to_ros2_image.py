#!/usr/bin/env python3

import argparse
import time

import carla
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CarlaRGBToROS2(Node):
    def __init__(self, host, port, topic, camera_role, ego_role):
        super().__init__("carla_rgb_to_ros2_image")

        self.host = host
        self.port = port
        self.topic = topic
        self.camera_role = camera_role
        self.ego_role = ego_role

        self.pub = self.create_publisher(Image, self.topic, 10)

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()

        self.camera = self._find_camera()

        if self.camera is None:
            self.get_logger().warn(
                f"role_name={self.camera_role} olan kamera bulunamadı. Ego araca yeni kamera takılacak."
            )
            self.camera = self._spawn_camera()

        self.get_logger().info(
            f"CARLA kamera bulundu: id={self.camera.id}, type={self.camera.type_id}, role_name={self.camera.attributes.get('role_name')}"
        )

        self.frame_count = 0
        self.last_log_time = time.time()

        self.camera.listen(self._on_image)

        self.get_logger().info(f"CARLA RGB görüntüsü ROS2 topic'e basılıyor: {self.topic}")

    def _find_camera(self):
        actors = self.world.get_actors()

        for actor in actors.filter("sensor.camera.rgb"):
            role_name = actor.attributes.get("role_name", "")
            if role_name == self.camera_role:
                return actor

        return None

    def _find_ego(self):
        actors = self.world.get_actors()

        for actor in actors.filter("vehicle.*"):
            role_name = actor.attributes.get("role_name", "")
            if role_name == self.ego_role:
                return actor

        return None

    def _spawn_camera(self):
        ego = self._find_ego()

        if ego is None:
            raise RuntimeError(
                f"role_name={self.ego_role} olan ego vehicle bulunamadı. Önce spawn scriptini çalıştır."
            )

        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")

        cam_bp.set_attribute("role_name", self.camera_role)
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", "0.05")

        cam_tf = carla.Transform(
            carla.Location(x=1.6, y=0.0, z=1.7),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

        camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
        return camera

    def _on_image(self, carla_image):
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))

        # CARLA raw format: BGRA
        bgr = array[:, :, :3]

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_rgb_front"

        msg.height = carla_image.height
        msg.width = carla_image.width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = carla_image.width * 3
        msg.data = bgr.tobytes()

        self.pub.publish(msg)

        self.frame_count += 1
        now = time.time()

        if now - self.last_log_time >= 2.0:
            fps = self.frame_count / (now - self.last_log_time)
            self.get_logger().info(f"Yayın aktif: {fps:.1f} FPS -> {self.topic}")
            self.frame_count = 0
            self.last_log_time = now

    def destroy_node(self):
        try:
            if self.camera is not None:
                self.camera.stop()
        except Exception:
            pass

        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--camera-role", default="rgb_front")
    parser.add_argument("--ego-role", default="ego_vehicle")
    args = parser.parse_args()

    rclpy.init()

    node = CarlaRGBToROS2(
        host=args.host,
        port=args.port,
        topic=args.topic,
        camera_role=args.camera_role,
        ego_role=args.ego_role,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
