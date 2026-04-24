import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DecisionNode(Node):
    def __init__(self):
        super().__init__("decision_node")

        self.declare_parameter("detections_topic", "/adas/perception/detections_json")
        self.declare_parameter("decision_topic", "/adas/decision")

        self.declare_parameter("distance_k", 1200.0)
        self.declare_parameter("stop_distance", 5.0)
        self.declare_parameter("slow_distance", 12.0)

        self.declare_parameter("lane_center_tolerance_ratio", 0.24)
        self.declare_parameter("vehicle_conf_threshold", 0.70)

        self.declare_parameter("min_bbox_height_ratio", 0.13)
        self.declare_parameter("min_bbox_width_ratio", 0.15)
        self.declare_parameter("min_bbox_area_ratio", 0.025)
        self.declare_parameter("min_aspect_ratio", 1.10)
        self.declare_parameter("max_aspect_ratio", 4.80)
        self.declare_parameter("min_bottom_y_ratio", 0.42)

        self.declare_parameter("ignore_left_edge_ratio", 0.02)
        self.declare_parameter("ignore_right_edge_ratio", 0.88)
        self.declare_parameter("max_missing_front", 3)

        self.detections_topic = self.get_parameter("detections_topic").value
        self.decision_topic = self.get_parameter("decision_topic").value

        self.distance_k = float(self.get_parameter("distance_k").value)
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.slow_distance = float(self.get_parameter("slow_distance").value)

        self.lane_center_tolerance_ratio = float(self.get_parameter("lane_center_tolerance_ratio").value)
        self.vehicle_conf_threshold = float(self.get_parameter("vehicle_conf_threshold").value)

        self.min_bbox_height_ratio = float(self.get_parameter("min_bbox_height_ratio").value)
        self.min_bbox_width_ratio = float(self.get_parameter("min_bbox_width_ratio").value)
        self.min_bbox_area_ratio = float(self.get_parameter("min_bbox_area_ratio").value)
        self.min_aspect_ratio = float(self.get_parameter("min_aspect_ratio").value)
        self.max_aspect_ratio = float(self.get_parameter("max_aspect_ratio").value)
        self.min_bottom_y_ratio = float(self.get_parameter("min_bottom_y_ratio").value)

        self.ignore_left_edge_ratio = float(self.get_parameter("ignore_left_edge_ratio").value)
        self.ignore_right_edge_ratio = float(self.get_parameter("ignore_right_edge_ratio").value)
        self.max_missing_front = int(self.get_parameter("max_missing_front").value)

        self.vehicle_labels = {"car", "truck", "bus"}

        self.last_front_vehicle = None
        self.missing_front_count = 0

        self.sub = self.create_subscription(String, self.detections_topic, self.callback, 10)
        self.pub = self.create_publisher(String, self.decision_topic, 10)

        self.get_logger().info(
            f"decision_node başladı: {self.detections_topic} -> {self.decision_topic}"
        )

    def estimate_distance(self, bbox_height):
        if bbox_height <= 1:
            return None
        return self.distance_k / bbox_height

    def is_valid_vehicle(self, det, frame_width, frame_height):
        label = det.get("label", "")
        conf = float(det.get("confidence", 0.0))
        bbox = det.get("bbox", None)

        if label not in self.vehicle_labels:
            return False

        if conf < self.vehicle_conf_threshold:
            return False

        if bbox is None or len(bbox) != 4:
            return False

        x1, y1, x2, y2 = map(float, bbox)

        if x2 <= x1 or y2 <= y1:
            return False

        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = w / max(h, 1.0)

        if x1 < frame_width * self.ignore_left_edge_ratio:
            return False

        if x2 > frame_width * self.ignore_right_edge_ratio:
            return False

        if h < frame_height * self.min_bbox_height_ratio:
            return False

        if w < frame_width * self.min_bbox_width_ratio:
            return False

        if area < frame_width * frame_height * self.min_bbox_area_ratio:
            return False

        if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
            return False

        if y2 < frame_height * self.min_bottom_y_ratio:
            return False

        return True

    def select_front_vehicle(self, detections, frame_width, frame_height):
        image_center_x = frame_width / 2.0
        lane_half_width = frame_width * self.lane_center_tolerance_ratio

        candidates = []

        for det in detections:
            if not self.is_valid_vehicle(det, frame_width, frame_height):
                continue

            x1, y1, x2, y2 = map(float, det["bbox"])

            w = x2 - x1
            h = y2 - y1
            area = w * h
            cx = (x1 + x2) / 2.0
            bottom_y = y2

            center_error = abs(cx - image_center_x)

            if center_error > lane_half_width:
                continue

            distance_est = self.estimate_distance(h)

            if distance_est is None:
                continue

            area_ratio = area / float(frame_width * frame_height)
            bottom_ratio = bottom_y / float(frame_height)
            center_score = 1.0 - min(center_error / lane_half_width, 1.0)

            selected = dict(det)
            selected["center_x"] = cx
            selected["bbox_width"] = w
            selected["bbox_height"] = h
            selected["area_ratio"] = area_ratio
            selected["distance_est"] = distance_est

            score = (
                float(selected["confidence"]) * 3.0
                + area_ratio * 10.0
                + bottom_ratio * 2.0
                + center_score * 3.0
            )

            selected["front_score"] = score
            candidates.append((score, selected))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def callback(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"JSON parse hatası: {exc}")
            return

        frame_width = int(data.get("frame_width", 800))
        frame_height = int(data.get("frame_height", 800))
        detections = data.get("detections", [])

        front_vehicle = self.select_front_vehicle(detections, frame_width, frame_height)

        used_memory = False

        if front_vehicle is not None:
            self.last_front_vehicle = front_vehicle
            self.missing_front_count = 0
        else:
            self.missing_front_count += 1

            if (
                self.last_front_vehicle is not None
                and self.missing_front_count <= self.max_missing_front
            ):
                front_vehicle = self.last_front_vehicle
                used_memory = True
            else:
                self.last_front_vehicle = None

        if front_vehicle is None:
            decision = "GO"
            risk = "LOW"
            distance_est = None
            reason = "front_vehicle_not_found"
        else:
            distance_est = float(front_vehicle["distance_est"])

            if distance_est <= self.stop_distance:
                decision = "STOP"
                risk = "HIGH"
            elif distance_est <= self.slow_distance:
                decision = "SLOW"
                risk = "MEDIUM"
            else:
                decision = "GO"
                risk = "LOW"

            reason = "front_vehicle_selected_from_memory" if used_memory else "front_vehicle_selected"

        output = {
            "decision": decision,
            "risk": risk,
            "distance_est": round(distance_est, 2) if distance_est is not None else None,
            "front_vehicle": front_vehicle,
            "reason": reason
        }

        out_msg = String()
        out_msg.data = json.dumps(output)
        self.pub.publish(out_msg)

        self.get_logger().info(json.dumps(output), throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()