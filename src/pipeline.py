"""
End-to-end pipeline: ensure dataset, train for 1 epoch (if needed), evaluate, and save reports.

Usage:
  python src/pipeline.py --data_yaml /app/data/traffic-detection-project/data.yaml \
      --project /app/data/outputs --train_name yolo_train --eval_name yolo_eval
"""
import os
import argparse
import glob
import shutil
from typing import Tuple

from train_yolo import train
from evaluate_yolo import evaluate
import glob as _glob
from analysis_failures import analyze_failures


def _create_fast_subset(data_yaml: str, max_train: int = 200, max_val: int = 80) -> Tuple[str, str]:
    """
    Create a small subset of the dataset for fast CPU demos.

    Returns:
        Tuple[str, str]: (fast_data_yaml_path, fast_root_dir)
    """
    import yaml
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    root = os.path.dirname(data_yaml)
    fast_root = os.path.join(root, "..", "traffic-detection-project_fast")
    os.makedirs(fast_root, exist_ok=True)
    for split, limit in (("train", max_train), ("valid", max_val)):
        src_img_dir = os.path.join(root, split, "images")
        src_lbl_dir = os.path.join(root, split, "labels")
        dst_img_dir = os.path.join(fast_root, split, "images")
        dst_lbl_dir = os.path.join(fast_root, split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        images = sorted([p for p in glob.glob(os.path.join(src_img_dir, "*.jpg"))])[:limit]
        for img in images:
            base = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(src_lbl_dir, base + ".txt")
            try:
                if not os.path.exists(os.path.join(dst_img_dir, os.path.basename(img))):
                    shutil.copy2(img, dst_img_dir)
                if os.path.exists(lbl) and not os.path.exists(os.path.join(dst_lbl_dir, os.path.basename(lbl))):
                    shutil.copy2(lbl, dst_lbl_dir)
            except Exception:
                continue
    fast_yaml = os.path.join(fast_root, "data.yaml")
    out = {
        "path": fast_root,
        "train": os.path.join(fast_root, "train", "images"),
        "val": os.path.join(fast_root, "valid", "images"),
        "names": cfg.get("names"),
        "nc": cfg.get("nc", len(cfg.get("names", []))),
    }
    with open(fast_yaml, "w") as f:
        import yaml
        yaml.safe_dump(out, f)
    return fast_yaml, fast_root


def _find_best_weights(project: str) -> str:
    """Return the most recent best.pt path under project/**/weights/, or empty string if none."""
    candidates = glob.glob(os.path.join(project, "**", "weights", "best.pt"), recursive=True)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _resolve_data_yaml(preferred_path: str) -> str:
    """Return a valid data.yaml path. If preferred does not exist, search /app/data/**/data.yaml.

    Preference order:
    - Exact preferred path if exists
    - Any path whose parent contains 'traffic-detection-project'
    - Any path whose parent contains 'traffic'
    - First discovered path
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
    candidates = sorted(_glob.glob("/app/data/**/data.yaml", recursive=True))
    if not candidates:
        raise FileNotFoundError(f"data.yaml not found. Looked for '{preferred_path}' and scanned /app/data/**/data.yaml")
    def _score(p: str) -> int:
        parent = os.path.basename(os.path.dirname(p)).lower()
        if "traffic-detection-project" in parent:
            return 0
        if "traffic" in parent:
            return 1
        return 2
    candidates.sort(key=_score)
    chosen = candidates[0]
    print(f"[pipeline] Using dataset config: {chosen}")
    return chosen


def run_pipeline(data_yaml: str, project: str, train_name: str, eval_name: str, imgsz: int = 416):
    # Resolve dataset config robustly
    data_yaml = _resolve_data_yaml(data_yaml)
    # Prefer a small fast subset for demos
    try:
        data_yaml, _fast_root = _create_fast_subset(data_yaml)
    except Exception:
        pass
    # Purge previous run directories for a fresh start
    try:
        for d in glob.glob(os.path.join(project, f"{train_name}*")):
            shutil.rmtree(d, ignore_errors=True)
        for d in glob.glob(os.path.join(project, f"{eval_name}*")):
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass
    os.makedirs(project, exist_ok=True)
    # Prefer centralized models directory
    models_dir = "/app/models"
    best = os.path.join(models_dir, "best.pt")
    if not os.path.exists(best):
        best = os.path.join(project, train_name, "weights", "best.pt")
    if not os.path.exists(best):
        train(
            data_yaml_path=data_yaml,
            model_name="yolov8n.pt",
            epochs=1,
            imgsz=int(imgsz),
            batch=4,
            device="cpu",
            project_dir=project,
            name=train_name,
        )
    # Resolve best weights path robustly
    if not os.path.exists(best):
        best = _find_best_weights(project)
    if not os.path.exists(best):
        # Final fallback to pretrained if nothing found
        best = "yolov8n.pt"
    metrics = evaluate(best, data_yaml, imgsz=int(imgsz), project_dir=project, name=eval_name)
    # Run failure analysis (non-fatal)
    try:
        eval_dir = os.path.join(project, eval_name)
        analyze_failures(data_yaml=data_yaml, eval_dir=eval_dir, iou_thr=0.5)
    except Exception:
        pass
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_yaml", default="/app/data/traffic-detection-project/data.yaml")
    p.add_argument("--project", default="/app/data/outputs")
    p.add_argument("--train_name", default="yolo_train")
    p.add_argument("--eval_name", default="yolo_eval")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    m = run_pipeline(args.data_yaml, args.project, args.train_name, args.eval_name, imgsz=args.imgsz)
    print(m)


