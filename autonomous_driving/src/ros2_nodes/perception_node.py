import json
from typing import Any, Dict, List

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("detections_topic", "/adas/perception/detections_json")
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("conf_threshold", 0.05)
        self.declare_parameter("show_debug", True)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_threshold = self.get_parameter("conf_threshold").get_parameter_value().double_value
        self.show_debug = self.get_parameter("show_debug").get_parameter_value().bool_value

        self.bridge = CvBridge()

        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"YOLO model yüklendi: {self.model_path}")
        except Exception as exc:
            self.get_logger().warn(
                f"Model yüklenemedi ({self.model_path}). Varsayılan yolov8n.pt kullanılacak. Hata: {exc}"
            )
            self.model = YOLO("yolov8n.pt")

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.pub = self.create_publisher(String, self.detections_topic, 10)

        self.get_logger().info("perception_node başladı")
        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"detections_topic={self.detections_topic}")
        self.get_logger().info(f"conf_threshold={self.conf_threshold}")

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"CvBridge hata: {exc}")
            return

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                imgsz=960,
                verbose=False
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO inference hata: {exc}")
            return

        annotated = frame.copy()
        detections: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]
            names = result.names

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    label = names.get(cls_id, str(cls_id))

                    detections.append(
                        {
                            "class_id": cls_id,
                            "label": label,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

                    if self.show_debug:
                        cv2.rectangle(
                            annotated,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            annotated,
                            f"{label} {conf:.2f}",
                            (int(x1), max(20, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

        labels = [det["label"] for det in detections]
        self.get_logger().info(f"Detected labels: {labels}")

        payload = {
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "detections": detections,
        }

        out = String()
        out.data = json.dumps(payload)
        self.pub.publish(out)

        if self.show_debug:
            cv2.imshow("adas_perception_debug", annotated)
            cv2.waitKey(1)

    def destroy(self) -> None:
        cv2.destroyAllWindows()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()