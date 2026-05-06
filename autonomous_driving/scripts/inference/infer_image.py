from pathlib import Path
import cv2

from src.inference.predictor import Predictor
from src.inference.decision_engine import DecisionEngine
from src.utils.bbox_utils import draw_box, crop_from_box
from src.utils.color_utils import detect_traffic_light_color


PROJECT_ROOT = Path(r"autonomous_driving_project")

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "bdd_yolo_v1" / "weights" / "best.pt"
SOURCE_IMAGE = PROJECT_ROOT / "test.jpg"
OUTPUT_IMAGE = PROJECT_ROOT / "outputs" / "predictions" / "infer_image_output.jpg"


def main():
    frame = cv2.imread(str(SOURCE_IMAGE))
    if frame is None:
        raise RuntimeError(f"Görsel okunamadı: {SOURCE_IMAGE}")

    predictor = Predictor(str(MODEL_PATH), conf=0.25)
    decision_engine = DecisionEngine()

    detections = predictor.predict_frame(frame)

    for det in detections:
        if det["class_name"] == "traffic light":
            crop = crop_from_box(frame, det["box"])
            tl_info = detect_traffic_light_color(crop)
            det["traffic_light_state"] = tl_info["state"]

    summary = decision_engine.analyze(detections, frame.shape)

    for det in detections:
        label = f"{det['class_name']} {det['conf']:.2f}"
        if det["class_name"] == "traffic light":
            label += f" | {det.get('traffic_light_state', 'unknown')}"
        frame = draw_box(frame, det["box"], label)

    overlay_lines = [
        f"ACTION: {summary['action']}",
        f"RISK: {summary['risk']}",
        f"TL_STATE: {summary['traffic_light_state']}",
        f"NOTES: {', '.join(summary['notes']) if summary['notes'] else '-'}",
    ]

    y = 30
    for line in overlay_lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE), frame)

    cv2.imshow("Inference Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()