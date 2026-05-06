import json
import time

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

        self.declare_parameter("default_go_speed", 1.5)
        self.declare_parameter("slow_speed", 0.8)
        self.declare_parameter("stop_speed", 0.0)

        self.declare_parameter("person_conf_threshold", 0.08)
        self.declare_parameter("traffic_light_conf_threshold", 0.40)
        self.declare_parameter("traffic_sign_conf_threshold", 0.20)
        self.declare_parameter("sign_classifier_conf_threshold", 0.25)

        self.declare_parameter("near_person_bottom_ratio", 0.30)
        self.declare_parameter("person_stop_hold_seconds", 3.0)

        self.detections_topic = self.get_parameter("detections_topic").value
        self.decision_topic = self.get_parameter("decision_topic").value

        self.distance_k = float(self.get_parameter("distance_k").value)
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.slow_distance = float(self.get_parameter("slow_distance").value)

        self.lane_center_tolerance_ratio = float(
            self.get_parameter("lane_center_tolerance_ratio").value
        )
        self.vehicle_conf_threshold = float(
            self.get_parameter("vehicle_conf_threshold").value
        )

        self.min_bbox_height_ratio = float(
            self.get_parameter("min_bbox_height_ratio").value
        )
        self.min_bbox_width_ratio = float(
            self.get_parameter("min_bbox_width_ratio").value
        )
        self.min_bbox_area_ratio = float(
            self.get_parameter("min_bbox_area_ratio").value
        )
        self.min_aspect_ratio = float(self.get_parameter("min_aspect_ratio").value)
        self.max_aspect_ratio = float(self.get_parameter("max_aspect_ratio").value)
        self.min_bottom_y_ratio = float(self.get_parameter("min_bottom_y_ratio").value)

        self.ignore_left_edge_ratio = float(
            self.get_parameter("ignore_left_edge_ratio").value
        )
        self.ignore_right_edge_ratio = float(
            self.get_parameter("ignore_right_edge_ratio").value
        )
        self.max_missing_front = int(self.get_parameter("max_missing_front").value)

        self.default_go_speed = float(self.get_parameter("default_go_speed").value)
        self.slow_speed = float(self.get_parameter("slow_speed").value)
        self.stop_speed = float(self.get_parameter("stop_speed").value)

        self.person_conf_threshold = float(
            self.get_parameter("person_conf_threshold").value
        )
        self.traffic_light_conf_threshold = float(
            self.get_parameter("traffic_light_conf_threshold").value
        )
        self.traffic_sign_conf_threshold = float(
            self.get_parameter("traffic_sign_conf_threshold").value
        )
        self.sign_classifier_conf_threshold = float(
            self.get_parameter("sign_classifier_conf_threshold").value
        )

        self.near_person_bottom_ratio = float(
            self.get_parameter("near_person_bottom_ratio").value
        )
        self.person_stop_hold_seconds = float(
            self.get_parameter("person_stop_hold_seconds").value
        )

        self.vehicle_labels = {"vehicle", "car", "truck", "bus", "van", "suv", "motorcycle"}

        self.stop_signs = {
            "dur",
            "girisi_olmayan_yol",
            "tasit_giremez",
        }

        self.slow_signs = {
            "yaya_gecidi",
            "okul_gecidi",
            "yol_calismasi",
            "dikkat",
            "yol_ver",
        }

        self.warning_signs = {
            "saga_donulmez",
            "sola_donulmez",
            "park_etmek_yasaktir",
            "duraklamak_park_yasaktir",
            "isikli_isaret_cihazi",
        }

        self.speed_limit_map = {
            "hiz_siniri_20": 0.5,
            "hiz_siniri_30": 0.8,
            "hiz_siniri_40": 1.0,
            "hiz_siniri_50": 1.2,
        }

        self.last_front_vehicle = None
        self.missing_front_count = 0

        self.last_speed_limit_sign = None
        self.current_speed_limit = None

        self.last_near_person_time = 0.0
        self.last_near_person = None

        self.sub = self.create_subscription(
            String,
            self.detections_topic,
            self.callback,
            10,
        )

        self.pub = self.create_publisher(
            String,
            self.decision_topic,
            10,
        )

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

    def get_best_traffic_light_state(self, detections):
        candidates = []

        for det in detections:
            if det.get("label") != "traffic_light":
                continue

            conf = float(det.get("confidence", 0.0))

            if conf < self.traffic_light_conf_threshold:
                continue

            state = det.get("traffic_light_state", "unknown")

            if state not in ["red", "yellow", "green"]:
                continue

            priority = {
                "red": 3,
                "yellow": 2,
                "green": 1,
            }.get(state, 0)

            candidates.append((priority, conf, state, det))

        if not candidates:
            return "unknown", None

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2], candidates[0][3]

    def get_best_sign(self, detections):
        candidates = []

        for det in detections:
            if det.get("label") != "traffic_sign":
                continue

            yolo_conf = float(det.get("confidence", 0.0))
            sign_type = det.get("sign_type", "unknown")
            sign_conf = float(det.get("sign_confidence", 0.0))

            if yolo_conf < self.traffic_sign_conf_threshold:
                continue

            if sign_type == "unknown":
                continue

            if sign_conf < self.sign_classifier_conf_threshold:
                continue

            score = sign_conf * 2.0 + yolo_conf
            candidates.append((score, det))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def get_person_risk(self, detections, frame_height):
        risky_persons = []

        for det in detections:
            if det.get("label") != "person":
                continue

            conf = float(det.get("confidence", 0.0))

            if conf < self.person_conf_threshold:
                continue

            bbox = det.get("bbox")

            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(float, bbox)
            bottom_ratio = y2 / float(frame_height)

            copied = dict(det)
            copied["bottom_ratio"] = bottom_ratio
            risky_persons.append(copied)

        if not risky_persons:
            return "none", None

        risky_persons.sort(key=lambda d: d["bottom_ratio"], reverse=True)
        nearest = risky_persons[0]

        if nearest["bottom_ratio"] >= self.near_person_bottom_ratio:
            return "near", nearest

        return "far", nearest

    def apply_person_stop_hold(self, person_risk, active_person):
        now = time.time()
        person_hold_active = False

        if person_risk == "near" and active_person is not None:
            self.last_near_person_time = now
            self.last_near_person = active_person
            return person_risk, active_person, person_hold_active

        recently_seen = (
            self.last_near_person is not None
            and (now - self.last_near_person_time) <= self.person_stop_hold_seconds
        )

        if recently_seen:
            person_hold_active = True

            held_person = dict(self.last_near_person)
            held_person["held"] = True
            held_person["hold_seconds_left"] = round(
                self.person_stop_hold_seconds - (now - self.last_near_person_time),
                2,
            )

            return "near", held_person, person_hold_active

        return person_risk, active_person, person_hold_active

    def apply_front_vehicle_memory(self, front_vehicle):
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

        return front_vehicle, used_memory

    def evaluate_vehicle_rule(self, front_vehicle, used_memory):
        if front_vehicle is None:
            return {
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.default_go_speed,
                "distance_est": None,
                "reason": "front_vehicle_not_found",
            }

        distance_est = float(front_vehicle["distance_est"])

        if distance_est <= self.stop_distance:
            return {
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "distance_est": distance_est,
                "reason": "front_vehicle_too_close_memory"
                if used_memory
                else "front_vehicle_too_close",
            }

        if distance_est <= self.slow_distance:
            return {
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "distance_est": distance_est,
                "reason": "front_vehicle_near_memory"
                if used_memory
                else "front_vehicle_near",
            }

        return {
            "decision": "GO",
            "risk": "LOW",
            "target_speed": self.default_go_speed,
            "distance_est": distance_est,
            "reason": "front_vehicle_safe",
        }

    def build_rule_decision(
        self,
        vehicle_rule,
        traffic_light_state,
        traffic_light_det,
        active_sign,
        person_risk,
        active_person,
        person_hold_active,
    ):
        active_sign_type = None
        active_sign_name = None
        sign_confidence = None

        if active_sign is not None:
            active_sign_type = active_sign.get("sign_type", "unknown")
            active_sign_name = active_sign.get("sign_name", active_sign_type)
            sign_confidence = float(active_sign.get("sign_confidence", 0.0))

        if active_sign_type in self.speed_limit_map:
            self.current_speed_limit = self.speed_limit_map[active_sign_type]
            self.last_speed_limit_sign = active_sign_type

        if traffic_light_state == "red":
            return {
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "reason": "red_light_detected",
            }

        if person_risk == "near":
            return {
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "reason": "near_person_hold"
                if person_hold_active
                else "near_person_detected",
            }

        if active_sign_type in self.stop_signs:
            return {
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "reason": f"stop_sign_detected:{active_sign_type}",
            }

        if vehicle_rule["decision"] == "STOP":
            return vehicle_rule

        if traffic_light_state == "yellow":
            return {
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "reason": "yellow_light_detected",
            }

        if active_sign_type in self.slow_signs:
            return {
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "reason": f"slow_sign_detected:{active_sign_type}",
            }

        if person_risk == "far":
            return {
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "reason": "person_detected",
            }

        if vehicle_rule["decision"] == "SLOW":
            return vehicle_rule

        if self.current_speed_limit is not None:
            return {
                "decision": "GO",
                "risk": "LOW",
                "target_speed": min(self.default_go_speed, self.current_speed_limit),
                "reason": f"speed_limit_active:{self.last_speed_limit_sign}",
            }

        if active_sign_type in self.warning_signs:
            return {
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.default_go_speed,
                "reason": f"warning_sign_detected:{active_sign_type}",
            }

        if traffic_light_state == "green":
            return {
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.default_go_speed,
                "reason": "green_light_detected",
            }

        return vehicle_rule

    def callback(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"JSON parse hatası: {exc}")
            return

        frame_width = int(data.get("image_width", data.get("frame_width", 800)))
        frame_height = int(data.get("image_height", data.get("frame_height", 800)))
        detections = data.get("detections", [])

        front_vehicle = self.select_front_vehicle(
            detections,
            frame_width,
            frame_height,
        )

        front_vehicle, used_memory = self.apply_front_vehicle_memory(front_vehicle)
        vehicle_rule = self.evaluate_vehicle_rule(front_vehicle, used_memory)

        traffic_light_state, traffic_light_det = self.get_best_traffic_light_state(
            detections
        )

        active_sign = self.get_best_sign(detections)

        raw_person_risk, raw_active_person = self.get_person_risk(
            detections,
            frame_height,
        )

        person_risk, active_person, person_hold_active = self.apply_person_stop_hold(
            raw_person_risk,
            raw_active_person,
        )

        final_rule = self.build_rule_decision(
            vehicle_rule=vehicle_rule,
            traffic_light_state=traffic_light_state,
            traffic_light_det=traffic_light_det,
            active_sign=active_sign,
            person_risk=person_risk,
            active_person=active_person,
            person_hold_active=person_hold_active,
        )

        distance_est = vehicle_rule.get("distance_est", None)

        output = {
            "decision": final_rule["decision"],
            "risk": final_rule["risk"],
            "target_speed": round(float(final_rule["target_speed"]), 2),
            "distance_est": round(float(distance_est), 2)
            if distance_est is not None
            else None,
            "front_vehicle": front_vehicle,
            "traffic_light_state": traffic_light_state,
            "traffic_light": traffic_light_det,
            "active_sign": active_sign,
            "person_risk": person_risk,
            "raw_person_risk": raw_person_risk,
            "person_hold_active": person_hold_active,
            "active_person": active_person,
            "speed_limit_active": self.last_speed_limit_sign,
            "reason": final_rule["reason"],
        }

        out_msg = String()
        out_msg.data = json.dumps(output)
        self.pub.publish(out_msg)

        active_sign = output.get("active_sign")
        traffic_light = output.get("traffic_light")
        active_person = output.get("active_person")

        sign_name = active_sign.get("sign_name") if active_sign else None
        sign_type = active_sign.get("sign_type") if active_sign else None
        sign_conf = active_sign.get("sign_confidence") if active_sign else None

        light_conf = traffic_light.get("confidence") if traffic_light else None
        person_conf = active_person.get("confidence") if active_person else None
        person_held = active_person.get("held") if active_person else None

        decision_log = (
            "\n"
            "================ ADAS DECISION LOG ================\n"
            f"DECISION        : {output.get('decision')}\n"
            f"RISK            : {output.get('risk')}\n"
            f"TARGET SPEED    : {output.get('target_speed')}\n"
            f"REASON          : {output.get('reason')}\n"
            "---------------------------------------------------\n"
            f"FRONT DISTANCE  : {output.get('distance_est')}\n"
            f"TRAFFIC LIGHT   : {output.get('traffic_light_state')} | conf={light_conf}\n"
            f"TRAFFIC SIGN    : {sign_name} | type={sign_type} | conf={sign_conf}\n"
            f"PERSON RISK     : {output.get('person_risk')} | conf={person_conf} | hold={person_held}\n"
            f"SPEED LIMIT     : {output.get('speed_limit_active')}\n"
            "===================================================\n"
        )

        self.get_logger().info(decision_log, throttle_duration_sec=0.5)


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