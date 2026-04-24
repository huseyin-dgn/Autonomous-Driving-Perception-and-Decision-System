import json
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("detections_topic", "/adas/perception/detections_json")
        self.declare_parameter("annotated_topic", "/adas/perception/annotated_image")
        self.declare_parameter("model_path", "yolov8n.pt")

        self.declare_parameter("conf_threshold", 0.20)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("max_det", 20)
        self.declare_parameter("imgsz", 640)

        self.declare_parameter("vehicle_min_conf", 0.20)
        self.declare_parameter("person_min_conf", 0.35)
        self.declare_parameter("traffic_min_conf", 0.40)

        self.declare_parameter("vehicle_min_area_ratio", 0.015)
        self.declare_parameter("vehicle_min_width_ratio", 0.08)
        self.declare_parameter("vehicle_min_height_ratio", 0.08)
        self.declare_parameter("vehicle_min_bottom_ratio", 0.30)
        self.declare_parameter("vehicle_min_aspect", 0.8)
        self.declare_parameter("vehicle_max_aspect", 4.5)

        self.declare_parameter("person_hold_frames", 12)
        self.declare_parameter("show_debug", True)

        self.image_topic = self.get_parameter("image_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.model_path = self.get_parameter("model_path").value

        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.imgsz = int(self.get_parameter("imgsz").value)

        self.vehicle_min_conf = float(self.get_parameter("vehicle_min_conf").value)
        self.person_min_conf = float(self.get_parameter("person_min_conf").value)
        self.traffic_min_conf = float(self.get_parameter("traffic_min_conf").value)

        self.vehicle_min_area_ratio = float(self.get_parameter("vehicle_min_area_ratio").value)
        self.vehicle_min_width_ratio = float(self.get_parameter("vehicle_min_width_ratio").value)
        self.vehicle_min_height_ratio = float(self.get_parameter("vehicle_min_height_ratio").value)
        self.vehicle_min_bottom_ratio = float(self.get_parameter("vehicle_min_bottom_ratio").value)
        self.vehicle_min_aspect = float(self.get_parameter("vehicle_min_aspect").value)
        self.vehicle_max_aspect = float(self.get_parameter("vehicle_max_aspect").value)

        self.person_hold_frames = int(self.get_parameter("person_hold_frames").value)
        self.show_debug = bool(self.get_parameter("show_debug").value)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.last_person_detections = []
        self.person_missing_count = 0

        self.window_name = "ADAS PERCEPTION DEBUG"

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.det_pub = self.create_publisher(
            String,
            self.detections_topic,
            10,
        )

        self.annotated_pub = self.create_publisher(
            Image,
            self.annotated_topic,
            10,
        )

        if self.show_debug:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("perception_node başladı")
        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"detections_topic={self.detections_topic}")
        self.get_logger().info(f"annotated_topic={self.annotated_topic}")
        self.get_logger().info(f"model_path={self.model_path}")
        self.get_logger().info(f"base_conf={self.conf_threshold} iou={self.iou_threshold}")

    def normalize_label(self, label):
        label = str(label).lower().strip()

        if label in ["car", "truck", "bus", "vehicle", "van", "suv"]:
            return "car"

        if label in ["person", "pedestrian"]:
            return "person"

        if label in ["traffic light", "traffic_light", "light"]:
            return "traffic_light"

        if label in ["traffic sign", "traffic_sign", "sign", "stop sign"]:
            return "traffic_sign"

        return label

    def is_vehicle(self, label):
        return label == "car"

    def is_person(self, label):
        return label == "person"

    def is_traffic(self, label):
        return label in ["traffic_light", "traffic_sign"]

    def estimate_traffic_light_state(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
        red_mask = red_mask1 + red_mask2

        yellow_mask = cv2.inRange(hsv, (18, 80, 80), (35, 255, 255))
        green_mask = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))

        red_score = cv2.countNonZero(red_mask)
        yellow_score = cv2.countNonZero(yellow_mask)
        green_score = cv2.countNonZero(green_mask)

        scores = {
            "red": red_score,
            "yellow": yellow_score,
            "green": green_score,
        }

        state = max(scores, key=scores.get)

        if scores[state] < 5:
            return "unknown"

        return state

    def pass_filter(self, det, frame_w, frame_h):
        label = det["label"]
        conf = det["confidence"]

        x1, y1, x2, y2 = det["bbox"]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        width_ratio = bw / frame_w
        height_ratio = bh / frame_h
        area_ratio = (bw * bh) / (frame_w * frame_h)
        bottom_ratio = y2 / frame_h
        aspect = bw / bh

        det["center_x"] = (x1 + x2) / 2.0
        det["bbox_width"] = bw
        det["bbox_height"] = bh
        det["area_ratio"] = area_ratio

        if self.is_vehicle(label):
            if conf < self.vehicle_min_conf:
                return False
            if area_ratio < self.vehicle_min_area_ratio:
                return False
            if width_ratio < self.vehicle_min_width_ratio:
                return False
            if height_ratio < self.vehicle_min_height_ratio:
                return False
            if bottom_ratio < self.vehicle_min_bottom_ratio:
                return False
            if aspect < self.vehicle_min_aspect:
                return False
            if aspect > self.vehicle_max_aspect:
                return False
            return True

        if self.is_person(label):
            if conf < 0.15:
                return False
            return True

        if self.is_traffic(label):
            if conf < self.traffic_min_conf:
                return False
            return True

        return False

    def apply_person_temporal_hold(self, detections):
        current_persons = [d for d in detections if d["label"] == "person"]

        strong_persons = [
            d for d in current_persons
            if d["confidence"] >= self.person_min_conf
        ]

        if len(strong_persons) > 0:
            self.last_person_detections = strong_persons
            self.person_missing_count = 0
            return detections

        if len(current_persons) > 0:
            if len(self.last_person_detections) > 0:
                merged = detections + self.last_person_detections
                self.person_missing_count = 0
                return merged

        if len(self.last_person_detections) > 0 and self.person_missing_count < self.person_hold_frames:
            held = []

            for det in self.last_person_detections:
                copied = dict(det)
                copied["held"] = True
                copied["confidence"] = max(0.01, copied["confidence"] * 0.90)
                held.append(copied)

            self.person_missing_count += 1
            return detections + held

        self.last_person_detections = []
        self.person_missing_count = 0
        return detections

    def remove_duplicate_detections(self, detections):
        vehicles = [d for d in detections if self.is_vehicle(d["label"])]
        others = [d for d in detections if not self.is_vehicle(d["label"])]

        vehicles = sorted(
            vehicles,
            key=lambda d: d["confidence"] * 2.0 + d["area_ratio"],
            reverse=True,
        )

        kept = []

        for det in vehicles:
            duplicate = False
            for old in kept:
                if self.iou(det["bbox"], old["bbox"]) > 0.55:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(det)

        return kept + others

    def iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter

        if union <= 0:
            return 0.0

        return inter / union

    def draw_detection(self, frame, det):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        conf = det["confidence"]
        held = bool(det.get("held", False))

        if label == "person":
            color = (255, 0, 0)
        elif label == "car":
            color = (0, 255, 0)
        elif label == "traffic_light":
            state = det.get("traffic_light_state", "unknown")

            if state == "red":
                color = (0, 0, 255)
            elif state == "yellow":
                color = (0, 255, 255)
            elif state == "green":
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)
        else:
            color = (0, 255, 255)

        thickness = 1 if held else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if label == "traffic_light":
            state = det.get("traffic_light_state", "unknown")
            text = f"{label} {state} {conf:.2f}"
        else:
            text = f"{label} {conf:.2f}"

        if held:
            text = f"{text} hold"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 2

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        tx1 = x1
        ty1 = max(0, y1 - th - 8)
        tx2 = min(frame.shape[1] - 1, x1 + tw + 8)
        ty2 = y1

        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 4, max(15, y1 - 6)),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    def draw_status_panel(self, frame, detections):
        vehicles = len([d for d in detections if self.is_vehicle(d["label"])])
        persons = len([d for d in detections if self.is_person(d["label"])])
        traffic_lights = len([d for d in detections if d["label"] == "traffic_light"])

        states = [
            d.get("traffic_light_state", "unknown")
            for d in detections
            if d["label"] == "traffic_light"
        ]

        state_text = "-"
        if len(states) > 0:
            state_text = ",".join(states)

        panel_h = 110
        cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_h), (85, 85, 85), -1)

        cv2.putText(
            frame,
            "ADAS PERCEPTION DEBUG",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"vehicles={vehicles} persons={persons} traffic_lights={traffic_lights} total={len(detections)}",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"traffic_light_state={state_text}",
            (15, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"model={self.model_path}",
            (15, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge hata: {exc}")
            return

        frame_h, frame_w = frame.shape[:2]
        annotated = frame.copy()

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO predict hata: {exc}")
            return

        raw_detections = []
        clean_detections = []

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    original_label = self.model.names.get(cls_id, str(cls_id))
                    label = self.normalize_label(original_label)

                    det = {
                        "class_id": cls_id,
                        "label": label,
                        "original_label": original_label,
                        "confidence": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }

                    if label == "traffic_light":
                        det["traffic_light_state"] = self.estimate_traffic_light_state(
                            frame,
                            det["bbox"],
                        )

                    raw_detections.append(det)

                    if self.pass_filter(det, frame_w, frame_h):
                        clean_detections.append(det)

        clean_detections = self.remove_duplicate_detections(clean_detections)
        clean_detections = self.apply_person_temporal_hold(clean_detections)

        for det in clean_detections:
            self.draw_detection(annotated, det)

        self.draw_status_panel(annotated, clean_detections)

        payload = {
            "stamp": time.time(),
            "image_width": frame_w,
            "image_height": frame_h,
            "detections": clean_detections,
        }

        msg_out = String()
        msg_out.data = json.dumps(payload)
        self.det_pub.publish(msg_out)

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
        except Exception as exc:
            self.get_logger().error(f"annotated image publish hata: {exc}")

        if self.show_debug:
            cv2.imshow(self.window_name, annotated)
            cv2.waitKey(1)

        clean_log = []
        for d in clean_detections:
            if d["label"] == "traffic_light":
                clean_log.append(
                    (
                        d["label"],
                        d.get("traffic_light_state", "unknown"),
                        round(d["confidence"], 2),
                    )
                )
            else:
                clean_log.append((d["label"], round(d["confidence"], 2)))

        self.get_logger().info(f"raw={len(raw_detections)} clean={clean_log}")


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_debug:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()