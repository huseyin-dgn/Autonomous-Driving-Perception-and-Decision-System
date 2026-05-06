#!/usr/bin/env python3

import time
import carla
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CarlaRGBFrontTrafficLightsOverlay(Node):
    def __init__(self):
        super().__init__("carla_rgb_front_traffic_lights_overlay")

        self.pub = self.create_publisher(Image, "/adas/camera/front/image_raw", 10)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()

        self.camera = self.find_camera()

        if self.camera is None:
            raise RuntimeError("rgb_front kamera bulunamadı. Önce spawn scriptini çalıştır.")

        self.frame_count = 0
        self.last_time = time.time()

        self.camera.listen(self.on_image)

        self.get_logger().info(f"CARLA kamera bulundu: id={self.camera.id}")
        self.get_logger().info("3 ışık overlay aktif: RED / YELLOW / GREEN")
        self.get_logger().info("Yayın başladı: /adas/camera/front/image_raw")

    def find_camera(self):
        for actor in self.world.get_actors().filter("sensor.camera.rgb"):
            role = actor.attributes.get("role_name", "")
            parent = getattr(actor, "parent", None)

            if role == "rgb_front" and parent is not None:
                return actor

        return None

    def draw_single_light(self, frame, x, y, state):
        # siyah trafik ışığı gövdesi
        cv2.rectangle(frame, (x - 28, y - 28), (x + 28, y + 92), (15, 15, 15), -1)
        cv2.rectangle(frame, (x - 28, y - 28), (x + 28, y + 92), (230, 230, 230), 2)

        red_pos = (x, y)
        yellow_pos = (x, y + 38)
        green_pos = (x, y + 76)

        # pasif lambalar
        cv2.circle(frame, red_pos, 13, (30, 30, 90), -1)
        cv2.circle(frame, yellow_pos, 13, (30, 90, 90), -1)
        cv2.circle(frame, green_pos, 13, (30, 90, 30), -1)

        if state == "red":
            cv2.circle(frame, red_pos, 14, (0, 0, 255), -1)
            cv2.circle(frame, red_pos, 18, (0, 0, 180), 2)

        elif state == "yellow":
            cv2.circle(frame, yellow_pos, 14, (0, 255, 255), -1)
            cv2.circle(frame, yellow_pos, 18, (0, 180, 180), 2)

        elif state == "green":
            cv2.circle(frame, green_pos, 14, (0, 255, 0), -1)
            cv2.circle(frame, green_pos, 18, (0, 180, 0), 2)

    def draw_test_lights(self, frame):
        h, w = frame.shape[:2]

        y = 45
        x1 = int(w * 0.18)
        x2 = int(w * 0.28)
        x3 = int(w * 0.38)

        self.draw_single_light(frame, x1, y, "red")
        self.draw_single_light(frame, x2, y, "yellow")
        self.draw_single_light(frame, x3, y, "green")

        return frame

    def on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))

        bgr = arr[:, :, :3].copy()
        bgr = self.draw_test_lights(bgr)

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

        self.frame_count += 1
        now = time.time()

        if now - self.last_time >= 2.0:
            fps = self.frame_count / (now - self.last_time)
            self.get_logger().info(f"Yayın aktif: {fps:.1f} FPS")
            self.frame_count = 0
            self.last_time = now


def main():
    rclpy.init()
    node = CarlaRGBFrontTrafficLightsOverlay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.camera.stop()
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
