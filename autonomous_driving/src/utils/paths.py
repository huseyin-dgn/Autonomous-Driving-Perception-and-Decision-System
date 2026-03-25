from pathlib import Path

PROJECT_ROOT = Path(r"autonomous_driving_project")

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "bdd_yolo_v14" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)