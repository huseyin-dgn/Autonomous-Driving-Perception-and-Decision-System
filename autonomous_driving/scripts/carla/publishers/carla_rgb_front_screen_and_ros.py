#!/usr/bin/env python3

import time
import threading
import argparse

import carla
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CarlaCameraScreenAndROS(Node):
    def __init__(self, host, port, topic):
        super().__init__("carla_rgb_front_screen_and_ros")

        self.pub = self.create_publisher(Image, topic, 10)
        self.topic = topic

        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()

        self.camera = self.find_camera()

        if self.camera is None:
            raise RuntimeError("rgb_front kamera bulunamadı. Önce spawn scriptini çalıştır.")

        self.latest_frame = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_time = time.time()

        self.camera.listen(self.on_image)

        self.get_logger().info(f"CARLA rgb_front kamera dinleniyor: id={self.camera.id}")
        self.get_logger().info(f"ROS topic yayını: {self.topic}")
        self.get_logger().info("OpenCV ekranı: CARLA RGB FRONT")

    def find_camera(self):
        for actor in self.world.get_actors().filter("sensor.camera.rgb"):
            role = actor.attributes.get("role_name", "")
            parent = getattr(actor, "parent", None)

            if role == "rgb_front" and parent is not None:
                return actor

        return None

    def on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))

        bgr = arr[:, :, :3].copy()

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_rgb_front"
        msg.height = image.height
        msg.width = image.width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = image.width * 3
        msg.data = bgr.tobytes()

        self.pub.publish(msg)

        with self.lock:
            self.latest_frame = bgr

        self.frame_count += 1
        now = time.time()

        if now - self.last_time >= 2.0:
            fps = self.frame_count / (now - self.last_time)
            self.get_logger().info(f"Kamera aktif: {fps:.1f} FPS -> {self.topic}")
            self.frame_count = 0
            self.last_time = now

    def show_loop(self):
        while rclpy.ok():
            frame = None

            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is not None:
                cv2.imshow("CARLA RGB FRONT", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            rclpy.spin_once(self, timeout_sec=0.001)

        cv2.destroyAllWindows()

    def destroy_node(self):
        try:
            self.camera.stop()
        except Exception:
            pass

        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    args = parser.parse_args()

    rclpy.init()

    node = CarlaCameraScreenAndROS(
        host=args.host,
        port=args.port,
        topic=args.topic,
    )

    try:
        node.show_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
