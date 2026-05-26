import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TrafficLightDecisionGateNode(Node):
    def __init__(self):
        super().__init__("traffic_light_decision_gate_node")

        self.declare_parameter("enabled", True)

        self.declare_parameter("detections_topic", "/adas/perception/detections_json")
        self.declare_parameter("base_decision_topic", "/adas/decision")
        self.declare_parameter("safe_decision_topic", "/adas/decision_safe")
        self.declare_parameter("debug_topic", "/adas/traffic_light/gate_debug")

        self.declare_parameter("tl_fresh_timeout_s", 0.80)
        self.declare_parameter("base_decision_timeout_s", 1.50)

        self.declare_parameter("min_det_conf", 0.40)
        self.declare_parameter("min_state_conf", 0.60)

        self.declare_parameter("red_stop_hold_s", 0.90)
        self.declare_parameter("yellow_slow_speed", 0.60)
        self.declare_parameter("offlane_person_filter_enabled", True)
        self.declare_parameter("person_image_width", 960.0)
        self.declare_parameter("person_stop_min_center_x_ratio", 0.30)
        self.declare_parameter("person_stop_max_center_x_ratio", 0.70)
        self.declare_parameter("person_stop_min_bottom_ratio", 0.45)

        self.enabled = bool(self.get_parameter("enabled").value)

        self.detections_topic = self.get_parameter("detections_topic").value
        self.base_decision_topic = self.get_parameter("base_decision_topic").value
        self.safe_decision_topic = self.get_parameter("safe_decision_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value

        self.tl_fresh_timeout_s = float(self.get_parameter("tl_fresh_timeout_s").value)
        self.base_decision_timeout_s = float(self.get_parameter("base_decision_timeout_s").value)

        self.min_det_conf = float(self.get_parameter("min_det_conf").value)
        self.min_state_conf = float(self.get_parameter("min_state_conf").value)

        self.red_stop_hold_s = float(self.get_parameter("red_stop_hold_s").value)
        self.yellow_slow_speed = float(self.get_parameter("yellow_slow_speed").value)

        self.offlane_person_filter_enabled = bool(self.get_parameter("offlane_person_filter_enabled").value)
        self.person_image_width = float(self.get_parameter("person_image_width").value)
        self.person_stop_min_center_x_ratio = float(self.get_parameter("person_stop_min_center_x_ratio").value)
        self.person_stop_max_center_x_ratio = float(self.get_parameter("person_stop_max_center_x_ratio").value)
        self.person_stop_min_bottom_ratio = float(self.get_parameter("person_stop_min_bottom_ratio").value)

        self.latest_base_decision = None
        self.latest_base_time = 0.0

        self.latest_tl = None
        self.latest_tl_time = 0.0
        self.red_hold_until = 0.0

        self.safe_pub = self.create_publisher(String, self.safe_decision_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.create_subscription(String, self.detections_topic, self.detections_cb, 10)
        self.create_subscription(String, self.base_decision_topic, self.base_decision_cb, 10)

        self.timer = self.create_timer(0.10, self.publish_safe_decision)

        self.get_logger().info(
            f"TrafficLightDecisionGate hazır. enabled={self.enabled}, "
            f"detections={self.detections_topic}, base={self.base_decision_topic}, "
            f"safe={self.safe_decision_topic}"
        )

    def as_float(self, value, default=None):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def norm_state(self, state):
        if state is None:
            return None
        s = str(state).strip().lower()
        aliases = {
            "r": "red",
            "red": "red",
            "kirmizi": "red",
            "kırmızı": "red",
            "y": "yellow",
            "yellow": "yellow",
            "sari": "yellow",
            "sarı": "yellow",
            "g": "green",
            "green": "green",
            "yesil": "green",
            "yeşil": "green",
            "unknown": "unknown",
            "unk": "unknown",
            "none": "unknown",
        }
        return aliases.get(s, s)

    def walk_dicts(self, obj):
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from self.walk_dicts(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from self.walk_dicts(item)

    def looks_like_traffic_light(self, d):
        label_keys = [
            "label", "mapped_label", "class_name", "name",
            "model_name", "original", "type", "category"
        ]

        for k in label_keys:
            v = d.get(k)
            if v is not None and "traffic_light" in str(v).lower():
                return True
            if v is not None and "traffic light" in str(v).lower():
                return True

        # Bazı JSON'larda class_id=2 trafik ışığıydı.
        if str(d.get("class_id", "")).strip() == "2":
            if any(k in d for k in ["state", "traffic_light_state", "state_conf", "state_source", "probs"]):
                return True

        # Top-level active_traffic_light gibi yapılarda label olmayabilir.
        if any(k in d for k in ["traffic_light_state", "light_state"]):
            return True

        return False

    def get_state_and_conf(self, d):
        state = (
            d.get("traffic_light_state")
            or d.get("light_state")
            or d.get("tl_state")
            or d.get("state")
        )

        state_conf = (
            d.get("traffic_light_state_conf")
            or d.get("state_conf")
            or d.get("tl_state_conf")
            or d.get("light_state_conf")
        )

        probs = d.get("probs")
        if isinstance(probs, dict):
            best_state = None
            best_conf = -1.0
            for k, v in probs.items():
                fv = self.as_float(v, None)
                if fv is not None and fv > best_conf:
                    best_state = k
                    best_conf = fv

            if state is None and best_state is not None:
                state = best_state
            if state_conf is None and best_conf >= 0.0:
                state_conf = best_conf

        state = self.norm_state(state)
        state_conf = self.as_float(state_conf, None)

        return state, state_conf

    def get_det_conf(self, d):
        for k in ["det_conf", "confidence", "conf", "score", "prob"]:
            if k in d:
                v = self.as_float(d.get(k), None)
                if v is not None:
                    return v
        return None

    def get_bbox(self, d):
        bbox = d.get("bbox") or d.get("box") or d.get("xyxy")
        if isinstance(bbox, dict):
            try:
                return [
                    float(bbox.get("x1")),
                    float(bbox.get("y1")),
                    float(bbox.get("x2")),
                    float(bbox.get("y2")),
                ]
            except Exception:
                return None

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            except Exception:
                return None

        return None

    def bbox_score(self, bbox, image_width=None, image_height=None):
        if not bbox:
            return 0.0

        x1, y1, x2, y2 = bbox
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)

        if bw <= 1.0 or bh <= 1.0:
            return 0.0

        # Çok küçük bbox'ları tamamen öldürmüyoruz; sadece skoru düşürüyoruz.
        area = bw * bh
        size_score = min(area / 900.0, 1.0)

        center_score = 0.0
        if image_width:
            cx = (x1 + x2) / 2.0
            dist_from_center = abs(cx - image_width / 2.0) / max(image_width / 2.0, 1.0)
            center_score = max(0.0, 1.0 - dist_from_center)

        return 0.55 * size_score + 0.45 * center_score

    def extract_best_traffic_light(self, data):
        image_width = data.get("image_width") if isinstance(data, dict) else None
        image_height = data.get("image_height") if isinstance(data, dict) else None

        candidates = []

        for d in self.walk_dicts(data):
            if not self.looks_like_traffic_light(d):
                continue

            state, state_conf = self.get_state_and_conf(d)
            det_conf = self.get_det_conf(d)
            bbox = self.get_bbox(d)

            if state not in {"red", "yellow", "green"}:
                continue

            if det_conf is None:
                det_conf = 1.0

            if state_conf is None:
                state_conf = 1.0 if state != "unknown" else 0.0

            if det_conf < self.min_det_conf:
                continue

            if state != "unknown" and state_conf < self.min_state_conf:
                continue

            bscore = self.bbox_score(bbox, image_width, image_height)

            # Red/yellow daha kritik; aynı skor durumunda onları öne al.
            state_priority = {
                "red": 0.18,
                "yellow": 0.10,
                "green": 0.05,
                "unknown": 0.0,
            }.get(state, 0.0)

            score = 0.45 * det_conf + 0.40 * state_conf + 0.15 * bscore + state_priority

            candidates.append({
                "state": state,
                "det_conf": round(float(det_conf), 4),
                "state_conf": round(float(state_conf), 4),
                "bbox": bbox,
                "score": round(float(score), 4),
                "raw_keys": sorted(list(d.keys()))[:20],
            })

        if not candidates:
            return None

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[0]

    def detections_cb(self, msg):
        try:
            data = json.loads(msg.data)
            best = self.extract_best_traffic_light(data)

            if best is not None:
                self.latest_tl = best
                self.latest_tl_time = time.time()

                if best["state"] == "red":
                    self.red_hold_until = time.time() + self.red_stop_hold_s

                if best["state"] == "green":
                    self.red_hold_until = 0.0

        except Exception as e:
            self.get_logger().warning(f"detections parse hata: {repr(e)}")

    def base_decision_cb(self, msg):
        try:
            self.latest_base_decision = json.loads(msg.data)
            self.latest_base_time = time.time()
        except Exception as e:
            self.get_logger().warning(f"base decision parse hata: {repr(e)}")

    def default_decision(self):
        return {
            "decision": "GO",
            "risk": "LOW",
            "target_speed": 2.0,
            "distance_est": None,
            "front_vehicle": None,
            "traffic_light_state": "unknown",
            "traffic_light_conf": None,
            "traffic_sign": None,
            "traffic_sign_type": None,
            "traffic_sign_conf": None,
            "person_risk": "none",
            "person_conf": None,
            "reason": "default_no_base_decision",
        }

    def apply_offlane_person_filter(self, out):
        if not self.offlane_person_filter_enabled:
            return out, "disabled", False

        decision = str(out.get("decision", "")).upper()
        reason = str(out.get("reason", ""))

        if decision != "STOP" or "person" not in reason:
            return out, "not_person_stop", False

        active = out.get("active_person")
        if not isinstance(active, dict):
            return out, "no_active_person_keep_stop", False

        center_x = self.as_float(active.get("center_x"), None)
        bottom_ratio = self.as_float(active.get("bottom_ratio"), None)

        if center_x is None:
            bbox = active.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                center_x = (self.as_float(bbox[0], 0.0) + self.as_float(bbox[2], 0.0)) / 2.0

        if center_x is None:
            return out, "no_center_x_keep_stop", False

        cx_ratio = center_x / max(self.person_image_width, 1.0)

        in_center_band = (
            self.person_stop_min_center_x_ratio
            <= cx_ratio
            <= self.person_stop_max_center_x_ratio
        )

        bottom_ok = True
        if bottom_ratio is not None:
            bottom_ok = bottom_ratio >= self.person_stop_min_bottom_ratio

        if in_center_band and bottom_ok:
            return out, f"person_in_lane_keep_stop:cx={cx_ratio:.3f},bottom={bottom_ratio}", False

        old_reason = out.get("reason")

        out["decision"] = "GO"
        out["risk"] = "LOW"
        out["target_speed"] = 2.0
        out["reason"] = f"offlane_person_filtered:old={old_reason},cx={cx_ratio:.3f},bottom={bottom_ratio}"
        out["person_risk"] = "offlane_filtered"
        out["person_filter"] = {
            "used": True,
            "center_x_ratio": round(float(cx_ratio), 3),
            "bottom_ratio": bottom_ratio,
            "center_band": [
                self.person_stop_min_center_x_ratio,
                self.person_stop_max_center_x_ratio,
            ],
        }

        return out, out["reason"], True

    def publish_safe_decision(self):
        now = time.time()

        if self.latest_base_decision is not None and now - self.latest_base_time <= self.base_decision_timeout_s:
            out = dict(self.latest_base_decision)
        else:
            out = self.default_decision()

        person_filter_reason = "not_checked"
        person_filter_used = False
        out, person_filter_reason, person_filter_used = self.apply_offlane_person_filter(out)

        gate_reason = "pass_base"
        gate_used = False
        tl = None

        if self.latest_tl is not None and now - self.latest_tl_time <= self.tl_fresh_timeout_s:
            tl = dict(self.latest_tl)

        if not self.enabled:
            gate_reason = "gate_disabled"

        elif tl is not None:
            state = tl.get("state", "unknown")
            det_conf = tl.get("det_conf")
            state_conf = tl.get("state_conf")

            out["traffic_light_state"] = state
            out["traffic_light_conf"] = state_conf
            out["traffic_light_det_conf"] = det_conf
            out["traffic_light_gate_candidate"] = tl

            base_decision = str(out.get("decision", "GO")).upper()

            if state == "red":
                out["decision"] = "STOP"
                out["risk"] = "HIGH"
                out["target_speed"] = 0.0
                out["reason"] = "traffic_light_red_gate"
                gate_reason = "red_stop"
                gate_used = True

            elif state == "yellow":
                if base_decision != "STOP":
                    out["decision"] = "SLOW"
                    out["risk"] = "MEDIUM"
                    out["target_speed"] = min(
                        float(out.get("target_speed", self.yellow_slow_speed) or self.yellow_slow_speed),
                        self.yellow_slow_speed,
                    )
                    out["reason"] = "traffic_light_yellow_gate"
                    gate_reason = "yellow_slow"
                    gate_used = True
                else:
                    gate_reason = "base_already_stop"

            elif state == "green":
                # Yeşil ışık karar sistemini serbest bırakır ama base STOP ise yaya/araç sebebiyle STOP kalabilir.
                gate_reason = "green_pass_base"

            else:
                gate_reason = "tl_unknown_pass_base"

        elif now < self.red_hold_until and self.enabled:
            out["decision"] = "STOP"
            out["risk"] = "HIGH"
            out["target_speed"] = 0.0
            out["traffic_light_state"] = "red"
            out["traffic_light_conf"] = None
            out["reason"] = "traffic_light_red_hold"
            gate_reason = "red_hold"
            gate_used = True

        else:
            if "traffic_light_state" not in out:
                out["traffic_light_state"] = "unknown"
            if "traffic_light_conf" not in out:
                out["traffic_light_conf"] = None

        out["tl_gate"] = {
            "enabled": self.enabled,
            "used": gate_used,
            "reason": gate_reason,
            "latest_tl_age_s": None if self.latest_tl is None else round(now - self.latest_tl_time, 3),
            "base_age_s": None if self.latest_base_decision is None else round(now - self.latest_base_time, 3),
            "person_filter_used": person_filter_used,
            "person_filter_reason": person_filter_reason,
        }

        msg = String()
        msg.data = json.dumps(out, ensure_ascii=False)
        self.safe_pub.publish(msg)

        dbg = String()
        dbg.data = json.dumps({
            "stamp": now,
            "gate_used": gate_used,
            "gate_reason": gate_reason,
            "person_filter_used": person_filter_used,
            "person_filter_reason": person_filter_reason,
            "selected_tl": tl,
            "safe_decision": {
                "decision": out.get("decision"),
                "risk": out.get("risk"),
                "target_speed": out.get("target_speed"),
                "reason": out.get("reason"),
                "traffic_light_state": out.get("traffic_light_state"),
                "traffic_light_conf": out.get("traffic_light_conf"),
            },
        }, ensure_ascii=False)
        self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDecisionGateNode()

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
