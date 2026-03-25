from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(r"autonomous_driving_project")
DATA_YAML = PROJECT_ROOT / "configs" / "data_bdd.yaml"

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=15,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=str(PROJECT_ROOT / "outputs" / "models"),
        name="bdd_yolo_v1",
        pretrained=True,
        verbose=True,
    )

if __name__ == "__main__":
    
    main()