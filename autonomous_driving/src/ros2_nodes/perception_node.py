from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.bridge = CvBridge()
        self.latest_scan = None

        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.decision_pub = self.create_publisher(String, "/driving_decision", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.model = self._load_model()

        self.get_logger().info("perception_node started.")

    def _load_model(self):
        if YOLO is None:
            self.get_logger().warn("Ultralytics not found. Running without YOLO.")
            return None

        root = Path(__file__).resolve().parents[2]
        model_path = root / "outputs" / "models" / "bdd_yolo_v14" / "weights" / "best.pt"

        if not model_path.exists():
            self.get_logger().warn(f"YOLO weights not found: {model_path}")
            return None

        try:
            model = YOLO(str(model_path))
            self.get_logger().info(f"YOLO loaded from: {model_path}")
            return model
        except Exception as e:
            self.get_logger().error(f"YOLO load failed: {e}")
            return None

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def _front_obstacle_distance(self) -> float | None:
        if self.latest_scan is None:
            return None

        ranges = list(self.latest_scan.ranges)
        if not ranges:
            return None

        center = len(ranges) // 2
        window = ranges[max(0, center - 20): min(len(ranges), center + 20)]
        valid = [r for r in window if self.latest_scan.range_min < r < self.latest_scan.range_max]

        if not valid:
            return None

        return min(valid)

    def _simple_lane_direction(self, frame):
        h, w = frame.shape[:2]
        roi = frame[h // 2:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=cv2.cv2.PI / 180 if hasattr(cv2, "cv2") else cv2.PI / 180,
            threshold=40,
            minLineLength=40,
            maxLineGap=20,
        )

        direction = "GO"

        if lines is not None:
            left_count = 0
            right_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    continue
                slope = dy / dx
                if slope < -0.3:
                    left_count += 1
                elif slope > 0.3:
                    right_count += 1

            if left_count > right_count + 3:
                direction = "LEFT"
            elif right_count > left_count + 3:
                direction = "RIGHT"

        return direction, edges

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        decision = "GO"

        front_dist = self._front_obstacle_distance()
        if front_dist is not None and front_dist < 1.5:
            decision = "STOP"

        lane_decision, lane_edges = self._simple_lane_direction(frame)
        if decision != "STOP":
            decision = lane_decision

        vis = frame.copy()

        if self.model is not None:
            try:
                results = self.model(vis, verbose=False)
                annotated = results[0].plot()
                vis = annotated

                names = results[0].names
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                        label = names.get(int(cls_id), str(int(cls_id))).lower()
                        if label in {"person", "pedestrian"} and conf > 0.4:
                            decision = "STOP"
                        if label in {"car", "truck", "bus"} and conf > 0.5 and front_dist is not None and front_dist < 3.0:
                            decision = "SLOW"
            except Exception as e:
                self.get_logger().warn(f"YOLO inference failed: {e}")

        cv2.putText(vis, f"Decision: {decision}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if front_dist is not None:
            cv2.putText(vis, f"Front dist: {front_dist:.2f} m", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        try:
            small_edges = cv2.cvtColor(lane_edges, cv2.COLOR_GRAY2BGR)
            small_edges = cv2.resize(small_edges, (320, 180))
            vis[0:180, 0:320] = small_edges
        except Exception:
            pass

        cv2.imshow("Perception Node", vis)
        cv2.waitKey(1)

        decision_msg = String()
        decision_msg.data = decision
        self.decision_pub.publish(decision_msg)

        cmd = Twist()
        if decision == "STOP":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif decision == "SLOW":
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif decision == "GO":
            cmd.linear.x = 0.8
            cmd.angular.z = 0.0
        elif decision == "LEFT":
            cmd.linear.x = 0.4
            cmd.angular.z = 0.4
        elif decision == "RIGHT":
            cmd.linear.x = 0.4
            cmd.angular.z = -0.4
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()