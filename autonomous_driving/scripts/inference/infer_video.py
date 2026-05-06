from pathlib import Path

from src.inference.video_processor import VideoProcessor


PROJECT_ROOT = Path(r"autonomous_driving_project")

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "bdd_yolo_v1" / "weights" / "best.pt"
SOURCE_VIDEO = PROJECT_ROOT / "test.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "outputs" / "predictions" / "infer_video_output.mp4"


def main():
    processor = VideoProcessor(
        model_path=str(MODEL_PATH),
        source=str(SOURCE_VIDEO),
        output_path=str(OUTPUT_VIDEO),
        conf=0.25,
        show=True,
    )
    processor.run()


if __name__ == "__main__":
    main()