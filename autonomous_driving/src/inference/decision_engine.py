from collections import deque
from typing import List, Dict, Any


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_width_height(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def estimate_distance_from_bbox_height(box, k=1200.0):
    _, h = box_width_height(box)
    return k / float(h)


class TLStateSmoother:
    def __init__(self, size=5):
        self.buffer = deque(maxlen=size)

    def update(self, state: str) -> str:
        if state != "unknown":
            self.buffer.append(state)

        if not self.buffer:
            return "unknown"

        values = list(self.buffer)
        return max(set(values), key=values.count)


class DecisionSmoother:
    def __init__(self, hold_stop_frames: int = 3):
        self.prev_action = "GO"
        self.stop_hold_counter = 0
        self.hold_stop_frames = hold_stop_frames

    def update(self, action: str) -> str:
        if action == "STOP":
            self.prev_action = "STOP"
            self.stop_hold_counter = self.hold_stop_frames
            return "STOP"

        if self.stop_hold_counter > 0:
            self.stop_hold_counter -= 1
            return "STOP"

        self.prev_action = action
        return action


class DecisionEngine:
    def __init__(self, tl_conf_thr: float = 0.45, car_conf_thr: float = 0.35, distance_k: float = 1200.0):
        self.tl_conf_thr = tl_conf_thr
        self.car_conf_thr = car_conf_thr
        self.distance_k = distance_k
        self.action_smoother = DecisionSmoother()

    def select_main_traffic_light(self, detections: List[Dict[str, Any]]):
        tls = []

        for det in detections:
            if det["class_name"] != "traffic light":
                continue
            if det["conf"] < self.tl_conf_thr:
                continue

            x1, y1, x2, y2 = det["box"]
            w = x2 - x1
            h = y2 - y1

            if w < 10 or h < 15:
                continue

            tls.append(det)

        if not tls:
            return None

        return max(tls, key=lambda d: d["conf"])

    def select_critical_front_car(self, detections: List[Dict[str, Any]], frame_shape):
        frame_h, frame_w = frame_shape[:2]
        lane_x_min = frame_w * 0.35
        lane_x_max = frame_w * 0.65

        candidate_cars = []

        for det in detections:
            if det["class_name"] != "car":
                continue
            if det["conf"] < self.car_conf_thr:
                continue

            box = det["box"]
            cx, _ = box_center(box)
            bw, bh = box_width_height(box)

            if bw < 20 or bh < 20:
                continue
            if not (lane_x_min <= cx <= lane_x_max):
                continue

            candidate_cars.append(det)

        if not candidate_cars:
            return None

        return max(candidate_cars, key=lambda d: box_width_height(d["box"])[1])

    def analyze(self, detections: List[Dict[str, Any]], frame_shape) -> Dict[str, Any]:
        summary = {
            "traffic_light_state": "unknown",
            "critical_car_found": False,
            "critical_car_distance": None,
            "action": "GO",
            "reason": "default_go",
        }

        main_tl = self.select_main_traffic_light(detections)
        critical_car = self.select_critical_front_car(detections, frame_shape)

        if main_tl is not None:
            tl_state = main_tl.get("traffic_light_state", "unknown")
            summary["traffic_light_state"] = tl_state
            summary["main_tl_box"] = main_tl["box"]

        if critical_car is not None:
            dist = estimate_distance_from_bbox_height(critical_car["box"], k=self.distance_k)
            summary["critical_car_found"] = True
            summary["critical_car_distance"] = dist
            summary["critical_car_box"] = critical_car["box"]
            summary["critical_car_conf"] = critical_car["conf"]

        tl_state = summary["traffic_light_state"]
        car_dist = summary["critical_car_distance"]
        stop_distance = 14.0

        if tl_state == "red":
            summary["action"] = "STOP"
            summary["reason"] = "red_light"

        elif tl_state == "yellow":
            summary["action"] = "STOP"
            summary["reason"] = "yellow_light"

        elif tl_state == "green":
            if car_dist is not None and car_dist < stop_distance:
                summary["action"] = "STOP"
                summary["reason"] = "car_close"
            else:
                summary["action"] = "GO"
                summary["reason"] = "clear"

        else:
            if car_dist is not None and car_dist < stop_distance:
                summary["action"] = "STOP"
                summary["reason"] = "car_close_unknown_light"
            else:
                summary["action"] = "GO"
                summary["reason"] = "default"

        summary["action"] = self.action_smoother.update(summary["action"])
        return summary