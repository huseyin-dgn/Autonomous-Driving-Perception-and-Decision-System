import cv2
import numpy as np

from src.models.detector import Detector
from src.inference.decision_engine import TLStateSmoother
from src.utils.paths import MODEL_PATH
from src.utils.visualizer import draw_box, draw_overlay, clamp_box
from src.utils.logger import SimpleLogger

from src.lane.lane_detector import LaneDetector
from src.lane.steering import compute_steering

def crop_from_box(frame, box):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def count_mask_pixels(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return int(np.count_nonzero(mask))


def detect_tl_raw(crop):
    if crop is None or crop.size == 0:
        return "unknown"

    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    top = hsv[0:h // 3]
    mid = hsv[h // 3: 2*h // 3]
    bot = hsv[2*h // 3:]

    red = count_mask_pixels(top, (0,90,90), (10,255,255)) + \
          count_mask_pixels(top, (160,90,90), (180,255,255))

    yellow = count_mask_pixels(mid, (15,90,90), (40,255,255))
    green = count_mask_pixels(bot, (40,70,70), (95,255,255))

    scores = {"red": red, "yellow": yellow, "green": green}
    state = max(scores, key=scores.get)

    if scores[state] < 10:
        return "unknown"

    return state

class DecisionSmoother:
    def __init__(self, hold_stop_frames=5):
        self.prev = "GO"
        self.counter = 0
        self.hold = hold_stop_frames

    def update(self, action):
        if action == "STOP":
            self.prev = "STOP"
            self.counter = self.hold
            return "STOP"

        if self.counter > 0:
            self.counter -= 1
            return "STOP"

        self.prev = action
        return action


def center(box):
    return (box[0] + box[2]) / 2


def height(box):
    return max(1, box[3] - box[1])


def distance(box, k=1200):
    return k / height(box)


def select_front_car(dets, frame_w):
    cars = []

    for d in dets:
        if d["class_name"] != "car" or d["conf"] < 0.35:
            continue

        cx = center(d["box"])

        if not (frame_w * 0.35 < cx < frame_w * 0.65):
            continue

        cars.append(d)

    if not cars:
        return None

    return max(cars, key=lambda d: height(d["box"]))


def decision(dets, frame_shape, smoother):
    h, w = frame_shape[:2]

    stop_distance = 12.0
    slow_distance = 18.0

    summary = {
        "traffic_light_state": "unknown",
        "critical_car_distance": None,
        "action": "GO",
        "reason": "default"
    }


    tls = [d for d in dets if d["class_name"] == "traffic light" and d["conf"] > 0.45]

    if tls:
        tl = max(tls, key=lambda d: d["conf"])
        summary["traffic_light_state"] = tl.get("traffic_light_state", "unknown")
        summary["main_tl_box"] = tl["box"]


    car = select_front_car(dets, w)

    if car:
        dist = distance(car["box"])
        summary["critical_car_distance"] = dist
        summary["critical_car_box"] = car["box"]

    tl = summary["traffic_light_state"]
    dist = summary["critical_car_distance"]

    if tl == "red":
        summary["action"] = "STOP"
        summary["reason"] = "red_light"

    elif tl == "yellow":
        summary["action"] = "STOP"
        summary["reason"] = "yellow_light"

    elif tl == "green":
        if dist is not None:
            if dist < stop_distance:
                summary["action"] = "STOP"
                summary["reason"] = "car_close"
            elif dist < slow_distance:
                summary["action"] = "SLOW"
                summary["reason"] = "car_medium"
            else:
                summary["action"] = "GO"
                summary["reason"] = "clear"
        else:
            summary["action"] = "GO"

    else:  
        if dist is not None and dist < stop_distance:
            summary["action"] = "SLOW"
            summary["reason"] = "unknown_light_close"
        else:
            summary["action"] = "GO"
            summary["reason"] = "unknown_clear"

    summary["action"] = smoother.update(summary["action"])

    return summary

def run_video(video_path):

    detector = Detector(MODEL_PATH)
    tl_smoother = TLStateSmoother()
    decision_smoother = DecisionSmoother()

    lane_detector = LaneDetector()
    logger = SimpleLogger()

    cap = cv2.VideoCapture(video_path)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.predict(frame)


        for det in detections:
            if det["class_name"] == "traffic light":
                crop = crop_from_box(frame, det["box"])
                raw = detect_tl_raw(crop)
                det["traffic_light_state"] = tl_smoother.update(raw)

        summary = decision(detections, frame.shape, decision_smoother)


        left_lane, right_lane = lane_detector.detect(frame)
        steer, _ = compute_steering(frame.shape, left_lane, right_lane)

        critical_box = summary.get("critical_car_box", None)

        for d in detections:
            label = f"{d['class_name']} {d['conf']:.2f}"
            color = (0,255,0)

            if d["class_name"] == "traffic light":
                label += f" {d.get('traffic_light_state','?')}"

            if critical_box and d["box"] == critical_box:
                color = (0,0,255)
                label += f" {summary['critical_car_distance']:.1f}"

            draw_box(frame, d["box"], label, color)

        # lanes
        if left_lane:
            cv2.line(frame, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255,0,0), 5)
        if right_lane:
            cv2.line(frame, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (255,0,0), 5)

        draw_overlay(frame, summary)

        cv2.putText(frame, f"STEER: {steer}", (20,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        logger.log(frame_id, summary)

        cv2.imshow("ADAS FINAL", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

VIDEO = r"two-cars-go-straight-through-a-red-traffic-light-driving-dash-cam-uk-dash-cam-ca.mp4"
if __name__ == "__main__":
    run_video(VIDEO)