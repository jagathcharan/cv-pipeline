"""
Training pipeline for object detection using Ultralytics YOLO.

Provides a function API and a CLI to launch CPU-friendly training runs.
"""
import os
import shutil
from typing import Optional

from ultralytics import YOLO
import argparse


def train(
    data_yaml_path: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 1,
    imgsz: int = 416,
    batch: int = 4,
    device: Optional[str] = "cpu",
    project_dir: str = "/app/data/outputs",
    name: str = "yolo_train",
):
    os.makedirs(project_dir, exist_ok=True)
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name=name,
        verbose=True,
        workers=2,
    )
    # Centralize best weights in /app/models for downstream evaluation
    try:
        best_src = os.path.join(project_dir, name, "weights", "best.pt")
        models_dir = "/app/models"
        os.makedirs(models_dir, exist_ok=True)
        best_dst = os.path.join(models_dir, "best.pt")
        if os.path.exists(best_src):
            shutil.copy2(best_src, best_dst)
    except Exception:
        pass
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO on traffic dataset")
    parser.add_argument("--data", dest="data_yaml_path", default="/app/data/traffic-detection-project/data.yaml")
    parser.add_argument("--model", dest="model_name", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", dest="project_dir", default="/app/data/outputs")
    parser.add_argument("--name", default="yolo_train")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_yaml_path=args.data_yaml_path,
        model_name=args.model_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project_dir=args.project_dir,
        name=args.name,
    )


