#!/usr/bin/env python3
import argparse
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ThreeTrafficLightScenario(Node):
    def __init__(self, args):
        super().__init__("three_visual_traffic_light_scenario")

        self.args = args
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, args.topic, 10)

        self.frame_count = 0
        self.timer = self.create_timer(1.0 / args.fps, self.publish_frame)

        self.get_logger().info("THREE TRAFFIC LIGHT VISUAL SCENARIO READY")
        self.get_logger().info(f"topic={args.topic}")
        self.get_logger().info("LEFT=RED | CENTER=YELLOW | RIGHT=GREEN")

    def draw_traffic_light(self, img, x, y, state):
        panel_w = 72
        panel_h = 165

        # Sarı dış kasa
        cv2.rectangle(
            img,
            (x, y),
            (x + panel_w, y + panel_h),
            (0, 185, 185),
            -1,
        )

        # Siyah iç panel
        cv2.rectangle(
            img,
            (x + 8, y + 8),
            (x + panel_w - 8, y + panel_h - 8),
            (18, 18, 18),
            -1,
        )

        red_c = (x + panel_w // 2, y + 38)
        yellow_c = (x + panel_w // 2, y + 82)
        green_c = (x + panel_w // 2, y + 126)

        centers = {
            "red": red_c,
            "yellow": yellow_c,
            "green": green_c,
        }

        # Sönük lensler
        for c in centers.values():
            cv2.circle(img, c, 16, (6, 6, 6), -1)
            cv2.circle(img, c, 16, (70, 70, 70), 2)

        if state == "red":
            active_color = (20, 20, 255)
        elif state == "yellow":
            active_color = (0, 255, 255)
        elif state == "green":
            active_color = (20, 255, 20)
        else:
            active_color = (150, 150, 150)

        # Aktif lens
        cv2.circle(img, centers[state], 17, active_color, -1)
        cv2.circle(img, centers[state], 25, active_color, 2)

        # Glow efekti
        overlay = img.copy()
        cv2.circle(overlay, centers[state], 34, active_color, -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        # Direk
        pole_x = x + panel_w // 2
        cv2.rectangle(
            img,
            (pole_x - 5, y + panel_h),
            (pole_x + 5, y + panel_h + 210),
            (70, 70, 70),
            -1,
        )

        # Label
        cv2.putText(
            img,
            state.upper(),
            (x - 8, y - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            active_color,
            2,
            cv2.LINE_AA,
        )

    def make_background(self):
        w = self.args.width
        h = self.args.height

        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Gökyüzü
        img[:] = (210, 210, 210)

        # Bina
        cv2.rectangle(img, (0, 0), (w, int(h * 0.62)), (150, 150, 150), -1)

        # Bina pencereleri
        for x in range(70, w, 170):
            for y in range(70, int(h * 0.55), 115):
                cv2.rectangle(img, (x, y), (x + 70, y + 55), (65, 75, 95), -1)
                cv2.rectangle(img, (x, y), (x + 70, y + 55), (210, 210, 210), 2)

        # Yol
        road_y = int(h * 0.66)
        cv2.rectangle(img, (0, road_y), (w, h), (85, 85, 85), -1)

        # Yol çizgisi
        line_y = int(h * 0.82)
        for x in range(0, w, 150):
            cv2.rectangle(img, (x, line_y - 4), (x + 75, line_y + 4), (240, 240, 240), -1)

        # Kaldırım
        cv2.rectangle(img, (0, int(h * 0.60)), (w, road_y), (190, 190, 190), -1)

        return img

    def publish_frame(self):
        img = self.make_background()

        # Üç trafik ışığı: sol red, orta yellow, sağ green
        base_y = 215

        self.draw_traffic_light(img, 250, base_y, "red")
        self.draw_traffic_light(img, 600, base_y, "yellow")
        self.draw_traffic_light(img, 950, base_y, "green")

        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "three_traffic_light_test"

        self.pub.publish(msg)

        self.frame_count += 1

        if self.frame_count % int(self.args.fps * 2) == 0:
            self.get_logger().info(f"published frames={self.frame_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/adas/camera/front/image_raw")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = ThreeTrafficLightScenario(args)

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
