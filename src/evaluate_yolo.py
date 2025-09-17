"""
Evaluation pipeline for Ultralytics YOLO model on the validation split.

Computes detection metrics and produces qualitative visualizations comparing predictions vs. ground truth.
"""
import os
from typing import Optional
import json

import numpy as np
import pandas as pd
from ultralytics import YOLO


def evaluate(
    weights_path: str,
    data_yaml_path: str,
    imgsz: int = 640,
    device: Optional[str] = "cpu",
    project_dir: str = "/app/data/outputs",
    name: str = "yolo_eval",
):
    os.makedirs(project_dir, exist_ok=True)
    model = YOLO(weights_path)
    # Ensure class names match between model and dataset; only proceed if intersection is non-empty
    ds_names = _load_dataset_classes(data_yaml_path)
    mdl_names = _get_model_classes(model)
    common = [c for c in mdl_names if c in ds_names]
    if not common:
        raise ValueError("No overlapping classes between model and dataset. Aborting evaluation.")

    metrics = model.val(
        data=data_yaml_path,
        imgsz=imgsz,
        device=device,
        project=project_dir,
        name=name,
        split="val",
        save_json=True,
        save_txt=True,
        save_hybrid=True,
        plots=True,
    )
    # metrics is a Box with keys like metrics/mAP50, metrics/mAP50-95, etc.
    # Convert to dict of floats for easy serialization
    out = {k: float(v) for k, v in metrics.results_dict.items()} if hasattr(metrics, "results_dict") else {}

    # Persist metrics for UI to load later without re-running
    try:
        run_dir = os.path.join(project_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(out, f, indent=2)
    except Exception:
        # Non-fatal: UI can still read other artifacts
        pass

    return out


def _load_dataset_classes(yaml_path: str):
    import yaml
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names or [])


def _get_model_classes(model: YOLO):
    try:
        return list(model.names.values())
    except Exception:
        return []


if __name__ == "__main__":
    DATA_YAML = "/app/data/traffic-detection-project/data.yaml"
    WEIGHTS = "/app/data/outputs/yolo_train/weights/best.pt"
    print(evaluate(WEIGHTS, DATA_YAML))


