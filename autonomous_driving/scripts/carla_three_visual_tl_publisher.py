#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ThreeTrafficLightVisualPublisher(Node):
    def __init__(self, args):
        super().__init__("three_visual_traffic_light_publisher")

        self.args = args
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, args.topic, 10)

        self.frame_id = 0
        self.timer = self.create_timer(1.0 / args.fps, self.publish_frame)

        self.get_logger().info("THREE VISUAL TRAFFIC LIGHT TEST READY")
        self.get_logger().info(f"topic={args.topic}")
        self.get_logger().info(f"image={args.width}x{args.height}")
        self.get_logger().info("left=RED, center=YELLOW, right=GREEN")

    def draw_one_light(self, img, x, y, state):
        panel_w = 70
        panel_h = 155

        # dış sarı çerçeve
        cv2.rectangle(
            img,
            (x, y),
            (x + panel_w, y + panel_h),
            (0, 190, 190),
            -1,
        )

        # iç siyah panel
        margin = 8
        cv2.rectangle(
            img,
            (x + margin, y + margin),
            (x + panel_w - margin, y + panel_h - margin),
            (20, 20, 20),
            -1,
        )

        centers = {
            "red": (x + panel_w // 2, y + 35),
            "yellow": (x + panel_w // 2, y + 78),
            "green": (x + panel_w // 2, y + 121),
        }

        # sönük lensler
        for name, center in centers.items():
            cv2.circle(img, center, 15, (5, 5, 5), -1)
            cv2.circle(img, center, 15, (60, 60, 60), 2)

        if state == "red":
            color = (20, 20, 255)
        elif state == "yellow":
            color = (0, 255, 255)
        elif state == "green":
            color = (20, 255, 20)
        else:
            color = (120, 120, 120)

        # aktif lens
        cv2.circle(img, centers[state], 16, color, -1)
        cv2.circle(img, centers[state], 22, color, 2)

        # küçük glow
        overlay = img.copy()
        cv2.circle(overlay, centers[state], 30, color, -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        # label
        cv2.putText(
            img,
            state.upper(),
            (x - 5, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    def publish_frame(self):
        w = self.args.width
        h = self.args.height

        img = np.zeros((h, w, 3), dtype=np.uint8)

        # arka plan: CARLA benzeri yol / kaldırım / bina basit sahne
        img[:] = (190, 190, 190)

        cv2.rectangle(img, (0, int(h * 0.65)), (w, h), (90, 90, 90), -1)
        cv2.line(img, (0, int(h * 0.82)), (w, int(h * 0.82)), (255, 255, 255), 3)

        for i in range(0, w, 120):
            cv2.rectangle(img, (i, int(h * 0.82) - 4), (i + 60, int(h * 0.82) + 4), (240, 240, 240), -1)

        # bina
        cv2.rectangle(img, (0, 0), (w, int(h * 0.62)), (150, 150, 150), -1)
        for x in range(80, w, 180):
            for y in range(70, int(h * 0.55), 120):
                cv2.rectangle(img, (x, y), (x + 70, y + 55), (80, 90, 105), -1)
                cv2.rectangle(img, (x, y), (x + 70, y + 55), (210, 210, 210), 2)

        # 3 trafik ışığı
        base_y = 210
        self.draw_one_light(img, 260, base_y, "red")
        self.draw_one_light(img, 600, base_y, "yellow")
        self.draw_one_light(img, 940, base_y, "green")

        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "three_visual_traffic_lights"

        self.pub.publish(msg)

        self.frame_id += 1

        if self.frame_id % int(self.args.fps * 2) == 0:
            self.get_logger().info(f"published frames={self.frame_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = ThreeTrafficLightVisualPublisher(args)

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
