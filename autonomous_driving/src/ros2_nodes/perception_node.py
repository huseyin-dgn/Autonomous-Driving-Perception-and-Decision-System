import json
import time

import cv2
import numpy as np
import rclpy
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image as PILImage
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
        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("model_path", "yolov8n.pt")

        self.declare_parameter(
            "sign_classifier_path",
            "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/sign_classifier/outputs/sign_classifier_resnet18_best.pt",
        )
        self.declare_parameter(
            "sign_class_names_path",
            "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/sign_classifier/outputs/class_names.json",
        )
        self.declare_parameter("sign_classifier_enabled", True)
        self.declare_parameter("sign_classifier_min_conf", 0.25)
        self.declare_parameter("sign_crop_padding_ratio", 0.20)

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
        self.decision_topic = self.get_parameter("decision_topic").value
        self.model_path = self.get_parameter("model_path").value

        self.sign_classifier_path = self.get_parameter("sign_classifier_path").value
        self.sign_class_names_path = self.get_parameter("sign_class_names_path").value
        self.sign_classifier_enabled = bool(self.get_parameter("sign_classifier_enabled").value)
        self.sign_classifier_min_conf = float(self.get_parameter("sign_classifier_min_conf").value)
        self.sign_crop_padding_ratio = float(self.get_parameter("sign_crop_padding_ratio").value)

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

        self.latest_decision = {
            "decision": "-",
            "risk": "-",
            "target_speed": "-",
            "reason": "-",
            "traffic_light_state": "-",
            "speed_limit_active": "-",
            "distance_est": None,
            "active_sign": None,
            "person_risk": "-",
        }

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sign_model = None
        self.sign_class_names = []

        self.sign_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if self.sign_classifier_enabled:
            self.load_sign_classifier()

        self.last_person_detections = []
        self.person_missing_count = 0

        self.window_name = "ADAS PERCEPTION + DECISION DEBUG"

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.decision_sub = self.create_subscription(
            String,
            self.decision_topic,
            self.decision_callback,
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
            cv2.resizeWindow(self.window_name, 1400, 800)

        self.get_logger().info("perception_node başladı")
        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"detections_topic={self.detections_topic}")
        self.get_logger().info(f"annotated_topic={self.annotated_topic}")
        self.get_logger().info(f"decision_topic={self.decision_topic}")
        self.get_logger().info(f"model_path={self.model_path}")
        self.get_logger().info(f"sign_classifier_enabled={self.sign_classifier_enabled}")

    def decision_callback(self, msg):
        try:
            self.latest_decision = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"decision JSON parse hata: {exc}")

    def load_sign_classifier(self):
        try:
            checkpoint = torch.load(self.sign_classifier_path, map_location=self.torch_device)

            if "class_names" in checkpoint:
                self.sign_class_names = checkpoint["class_names"]
            else:
                with open(self.sign_class_names_path, "r", encoding="utf-8") as f:
                    self.sign_class_names = json.load(f)

            num_classes = len(self.sign_class_names)

            self.sign_model = models.resnet18(weights=None)
            self.sign_model.fc = nn.Sequential(
                nn.Dropout(0.30),
                nn.Linear(self.sign_model.fc.in_features, num_classes),
            )

            self.sign_model.load_state_dict(checkpoint["model_state_dict"])
            self.sign_model.to(self.torch_device)
            self.sign_model.eval()

            self.get_logger().info("traffic sign classifier yüklendi")
            self.get_logger().info(f"sign classes={self.sign_class_names}")

        except Exception as exc:
            self.sign_classifier_enabled = False
            self.sign_model = None
            self.get_logger().error(f"traffic sign classifier yüklenemedi: {exc}")

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

    def pretty_sign_name(self, sign_type):
        names = {
            "dikkat": "Dikkat",
            "dur": "Dur",
            "duraklamak_park_yasaktir": "Duraklamak/Park Yasak",
            "girisi_olmayan_yol": "Girisi Olmayan Yol",
            "hiz_siniri_20": "Hiz Siniri 20",
            "hiz_siniri_30": "Hiz Siniri 30",
            "hiz_siniri_40": "Hiz Siniri 40",
            "hiz_siniri_50": "Hiz Siniri 50",
            "isikli_isaret_cihazi": "Isikli Isaret Cihazi",
            "okul_gecidi": "Okul Gecidi",
            "park_etmek_yasaktir": "Park Yasak",
            "saga_donulmez": "Saga Donulmez",
            "sola_donulmez": "Sola Donulmez",
            "tasit_giremez": "Tasit Giremez",
            "yaya_gecidi": "Yaya Gecidi",
            "yol_calismasi": "Yol Calismasi",
            "yol_ver": "Yol Ver",
            "unknown": "Bilinmiyor",
        }

        return names.get(sign_type, sign_type)

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

    def classify_traffic_sign(self, frame, bbox):
        if not self.sign_classifier_enabled or self.sign_model is None:
            return "unknown", 0.0

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = int(bw * self.sign_crop_padding_ratio)
        pad_y = int(bh * self.sign_crop_padding_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return "unknown", 0.0

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return "unknown", 0.0

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(crop_rgb)

            tensor = self.sign_transform(pil_img).unsqueeze(0).to(self.torch_device)

            with torch.no_grad():
                logits = self.sign_model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

            sign_type = self.sign_class_names[int(pred_idx.item())]
            sign_conf = float(conf.item())

            return sign_type, sign_conf

        except Exception as exc:
            self.get_logger().error(f"traffic sign classify hata: {exc}")
            return "unknown", 0.0

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
        elif label == "traffic_sign":
            color = (0, 255, 255)
        else:
            color = (0, 255, 255)

        thickness = 1 if held else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if label == "traffic_light":
            state = det.get("traffic_light_state", "unknown")
            text = f"{label} {state} {conf:.2f}"
        elif label == "traffic_sign":
            sign_type = det.get("sign_type", "unknown")
            sign_conf = det.get("sign_confidence", 0.0)
            pretty_name = self.pretty_sign_name(sign_type)

            if sign_type != "unknown":
                text = f"LEVHA: {pretty_name} {sign_conf:.2f}"
            else:
                text = f"LEVHA: Bilinmiyor {conf:.2f}"
        else:
            text = f"{label} {conf:.2f}"

        if held:
            text = f"{text} hold"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
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

    def draw_text(self, frame, text, x, y, color=(255, 255, 255), scale=0.52, thickness=2):
        cv2.putText(
            frame,
            str(text),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def make_debug_canvas(self, frame, detections):
        h, w = frame.shape[:2]
        panel_w = 470

        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        panel = canvas[:, w:w + panel_w]
        panel[:, :] = (45, 45, 45)

        vehicles = len([d for d in detections if self.is_vehicle(d["label"])])
        persons = len([d for d in detections if self.is_person(d["label"])])
        traffic_lights = len([d for d in detections if d["label"] == "traffic_light"])
        traffic_signs = len([d for d in detections if d["label"] == "traffic_sign"])

        states = [
            d.get("traffic_light_state", "unknown")
            for d in detections
            if d["label"] == "traffic_light"
        ]

        signs = [
            self.pretty_sign_name(d.get("sign_type", "unknown"))
            for d in detections
            if d["label"] == "traffic_sign"
        ]

        state_text = ",".join(states) if len(states) > 0 else "-"
        sign_text = ",".join(signs[:4]) if len(signs) > 0 else "-"

        decision = self.latest_decision.get("decision", "-")
        risk = self.latest_decision.get("risk", "-")
        target_speed = self.latest_decision.get("target_speed", "-")
        reason = self.latest_decision.get("reason", "-")
        distance_est = self.latest_decision.get("distance_est", "-")
        traffic_light_state = self.latest_decision.get("traffic_light_state", "-")
        person_risk = self.latest_decision.get("person_risk", "-")

        active_sign = self.latest_decision.get("active_sign")
        active_sign_name = "-"
        active_sign_type = "-"

        if active_sign:
            active_sign_name = active_sign.get("sign_name", "-")
            active_sign_type = active_sign.get("sign_type", "-")

        if decision == "STOP":
            decision_color = (0, 0, 255)
        elif decision == "SLOW":
            decision_color = (0, 255, 255)
        elif decision == "GO":
            decision_color = (0, 255, 0)
        else:
            decision_color = (255, 255, 255)

        px = w + 20

        self.draw_text(canvas, "ADAS DEBUG PANEL", px, 35, (255, 255, 255), 0.70, 2)

        self.draw_text(canvas, "DETECTION", px, 80, (200, 200, 200), 0.58, 2)
        self.draw_text(canvas, f"Vehicles      : {vehicles}", px, 112)
        self.draw_text(canvas, f"Persons       : {persons}", px, 140)
        self.draw_text(canvas, f"TrafficLight  : {traffic_lights}", px, 168)
        self.draw_text(canvas, f"TrafficSign   : {traffic_signs}", px, 196)
        self.draw_text(canvas, f"Light States  : {state_text}", px, 224)
        self.draw_text(canvas, f"Signs         : {sign_text}", px, 252, (255, 255, 255), 0.45, 1)

        cv2.line(canvas, (w + 15, 280), (w + panel_w - 15, 280), (100, 100, 100), 1)

        self.draw_text(canvas, "DECISION", px, 320, (200, 200, 200), 0.58, 2)
        self.draw_text(canvas, f"Decision      : {decision}", px, 360, decision_color, 0.75, 2)
        self.draw_text(canvas, f"Risk          : {risk}", px, 395, decision_color, 0.60, 2)
        self.draw_text(canvas, f"Target Speed  : {target_speed}", px, 425)
        self.draw_text(canvas, f"Front Dist    : {distance_est}", px, 455)
        self.draw_text(canvas, f"Rule          : {reason}", px, 485, (255, 255, 255), 0.45, 1)

        cv2.line(canvas, (w + 15, 515), (w + panel_w - 15, 515), (100, 100, 100), 1)

        self.draw_text(canvas, "ACTIVE INPUTS", px, 555, (200, 200, 200), 0.58, 2)
        self.draw_text(canvas, f"Light         : {traffic_light_state}", px, 590)
        self.draw_text(canvas, f"Sign          : {active_sign_name}", px, 620, (255, 255, 255), 0.48, 1)
        self.draw_text(canvas, f"Sign Type     : {active_sign_type}", px, 648, (255, 255, 255), 0.45, 1)
        self.draw_text(canvas, f"Person Risk   : {person_risk}", px, 676)

        return canvas

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

                    if label == "traffic_sign":
                        sign_type, sign_confidence = self.classify_traffic_sign(
                            frame,
                            det["bbox"],
                        )
                        det["sign_type"] = sign_type
                        det["sign_confidence"] = sign_confidence
                        det["sign_name"] = self.pretty_sign_name(sign_type)

                    raw_detections.append(det)

                    if self.pass_filter(det, frame_w, frame_h):
                        clean_detections.append(det)

        clean_detections = self.remove_duplicate_detections(clean_detections)
        clean_detections = self.apply_person_temporal_hold(clean_detections)

        for det in clean_detections:
            self.draw_detection(annotated, det)

        debug_canvas = self.make_debug_canvas(annotated, clean_detections)

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
            annotated_msg = self.bridge.cv2_to_imgmsg(debug_canvas, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
        except Exception as exc:
            self.get_logger().error(f"annotated image publish hata: {exc}")

        if self.show_debug:
            cv2.imshow(self.window_name, debug_canvas)
            cv2.waitKey(1)


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