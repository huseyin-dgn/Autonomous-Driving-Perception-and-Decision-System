import queue
import cv2
import numpy as np
import carla

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CarlaGreenRGBPublisher(Node):
    def __init__(self):
        super().__init__("carla_green_rgb_publisher")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "/adas/camera/front/image_raw", 10)

        client = carla.Client("localhost", 2000)
        client.set_timeout(20.0)
        self.world = client.get_world()

        cameras = []
        for a in self.world.get_actors().filter("sensor.camera.rgb"):
            role = a.attributes.get("role_name", "")
            self.get_logger().info(f"FOUND CAMERA id={a.id} role_name={role}")
            if role == "rgb_front":
                cameras.append(a)

        if not cameras:
            raise RuntimeError("rgb_front kamera yok. Scene scripti kamerayı spawn etmemiş veya başka worldde.")

        self.camera = cameras[-1]
        self.q = queue.Queue(maxsize=2)

        self.camera.listen(self.on_image)

        self.get_logger().info(
            f"USING CAMERA id={self.camera.id} role_name={self.camera.attributes.get('role_name')}"
        )
        self.get_logger().info("Publishing -> /adas/camera/front/image_raw")

    def on_image(self, image):
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            bgr = arr[:, :, :3]

            if self.q.full():
                try:
                    self.q.get_nowait()
                except Exception:
                    pass

            self.q.put_nowait(bgr.copy())
        except Exception as e:
            self.get_logger().error(f"on_image hata: {e}")

    def spin_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            try:
                frame = self.q.get(timeout=0.05)
            except queue.Empty:
                continue

            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "carla_rgb_front"
            self.pub.publish(msg)

            cv2.imshow("CARLA RGB FRONT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    rclpy.init()
    node = CarlaGreenRGBPublisher()
    try:
        node.spin_loop()
    finally:
        try:
            node.camera.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
