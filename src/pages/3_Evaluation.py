import os
import glob
import shutil
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import cv2
from evaluate_yolo import evaluate
from dataset_utils import parse_yolo_label, load_classes_from_yaml
from train_yolo import train

def draw_boxes(image_path, boxes, class_names, color=(0, 255, 0)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    for b in boxes:
        # supports either YOLO normalized {x_center,y_center,width,height,class_id} or xyxy
        if all(k in b for k in ("x_center", "y_center", "width", "height")):
            x1 = int((b["x_center"] - b["width"]/2) * w)
            y1 = int((b["y_center"] - b["height"]/2) * h)
            x2 = int((b["x_center"] + b["width"]/2) * w)
            y2 = int((b["y_center"] + b["height"]/2) * h)
            cls_id = int(b.get("class_id", 0))
        else:
            x1, y1, x2, y2 = map(int, b[:4])
            cls_id = int(b[5]) if len(b) > 5 else 0
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls_id]
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def _find_weight_candidates(project_dir: str):
    pattern = os.path.join(project_dir, "**", "weights", "best.pt")
    return sorted(glob.glob(pattern, recursive=True))


def _latest_best_or_default(project_dir: str) -> str:
    """Return latest best.pt under project_dir or fallback to /app/models/best.pt or yolov8n.pt."""
    # Prefer centralized models path
    models_best = os.path.join("/app/models", "best.pt")
    candidates = _find_weight_candidates(project_dir)
    if os.path.exists(models_best):
        candidates.append(models_best)
    if candidates:
        try:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        except Exception:
            candidates = sorted(candidates)
        return candidates[0]
    return "yolov8n.pt"


def main():
    st.set_page_config(page_title="Evaluation | Traffic Detection", layout="wide")
    st.title("Training and Evaluation Reports")

    # Fixed configuration for automated pipeline
    # Use the fast subset if available; pipeline will create it as .../traffic-detection-project_fast
    data_yaml = "/app/data/traffic-detection-project_fast/data.yaml"
    if not os.path.exists(data_yaml):
        data_yaml = "/app/data/traffic-detection-project/data.yaml"
    project_dir = "/app/data/outputs"
    train_run = "yolo_train"
    eval_run = "yolo_eval"
    imgsz = 416

    # Minimal UI: show single evaluation report from yolo_eval if present
    st.subheader("Evaluation")
    best_weights = os.path.join(project_dir, train_run, "weights", "best.pt")

    metrics = None
    metrics_path = os.path.join(project_dir, eval_run, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    if metrics is not None:
        st.subheader("Quantitative Metrics")
        st.json(metrics)
    else:
        st.info("Evaluation report not found. It will be created on first startup build run.")

    # Optional baseline comparison against yolov8n.pt
    #st.info("Baseline comparison skipped to avoid COCO-class confusion. Using only dataset-trained weights.")

    # Load dataset classes early for downstream sections (class alignment expander)
    try:
        class_names = load_classes_from_yaml(data_yaml)
    except Exception:
        class_names = []

    # Remove manual controls; this page is read-only

    # Class alignment diagnostics for the trained model
    with st.expander("Class Alignment (Dataset vs Model)", expanded=False):
        ds_classes = class_names
        st.write("Dataset classes (from YAML):")
        st.code(", ".join(ds_classes))
        try:
            from ultralytics import YOLO as _YOLO
            mdl = _YOLO(best_weights)
            mdl_classes = list(mdl.names.values()) if hasattr(mdl, "names") else []
        except Exception:
            mdl_classes = []
        if mdl_classes:
            st.write("Model classes:")
            st.code(", ".join(mdl_classes))
        # Intersections
        common = [c for c in (mdl_classes or []) if c in ds_classes]
        if mdl_classes:
            st.info(f"Dataset ∩ Model: {len(common)} classes")

        # Display confusion matrix and normalized confusion matrix if saved by Ultralytics
        cm_path = os.path.join(project_dir, eval_run, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")
        else:
            st.info("Confusion matrix image not found; depending on version it can be inside runs/val.* folder.")

        # PR curves and F1 curves if generated
        for fname, label in [
            ("PR_curve.png", "Precision-Recall Curve"),
            ("F1_curve.png", "F1 vs Confidence"),
            ("confusion_matrix_normalized.png", "Confusion Matrix (Normalized)"),
        ]:
            p = os.path.join(project_dir, eval_run, fname)
            if os.path.exists(p):
                st.image(p, caption=label)

    st.header("Qualitative Comparison and Class-matched Analysis")
    st.write("Visualize predictions vs. ground truth on validation images and compare counts for dataset YAML classes only.")
    data_dir = "/app/data/traffic-detection-project"
    class_names = load_classes_from_yaml(os.path.join(data_dir, "data.yaml"))

    # Show class matching info
    try:
        from ultralytics import YOLO as _YOLO
        _model = _YOLO(best_weights)
        model_classes = list(_model.names.values()) if hasattr(_model, "names") else []
    except Exception:
        model_classes = []
    # Strictly focus on dataset yaml classes for all charts
    common_classes = list(class_names)
    if common_classes:
        st.info(f"Using classes (dataset YAML): {', '.join(common_classes)}")
    else:
        st.warning("No classes resolved from dataset.yaml; showing GT only.")

    # Select a few images from valid split
    img_dir = os.path.join(data_dir, "valid", "images")
    lbl_dir = os.path.join(data_dir, "valid", "labels")
    if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        picks = files[:6]
        cols = st.columns(3)
        for idx, fname in enumerate(picks):
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            gt = parse_yolo_label(lbl_path) if os.path.exists(lbl_path) else []
            # Filter GT to matching classes (by name)
            if common_classes and gt:
                name_to_id = {name: idx for idx, name in enumerate(class_names)}
                allowed_ids = {name_to_id[c] for c in common_classes if c in name_to_id}
                gt = [b for b in gt if int(b.get("class_id", -1)) in allowed_ids]
            gt_img = draw_boxes(img_path, gt, class_names, color=(0, 255, 0))
            with cols[idx % 3]:
                if gt_img:
                    st.image(gt_img, caption=f"GT: {fname}")
                else:
                    st.warning(f"Could not load {fname}")

        # Additional charts: per-class counts (GT) limited to dataset classes
        import glob
        import pandas as pd
        lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        counts = {c: 0 for c in class_names}
        for lf in lbl_files[:1000]:
            for line in open(lf, "r", encoding="utf-8", errors="ignore").read().strip().splitlines():
                if not line:
                    continue
                try:
                    cid = int(line.split()[0])
                    if 0 <= cid < len(class_names):
                        counts[class_names[cid]] += 1
                except Exception:
                    continue
        if any(v > 0 for v in counts.values()):
            st.subheader("Validation Label Class Distribution (Dataset classes)")
            dfc = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())}).sort_values("count", ascending=False)
            fig = px.bar(dfc, x="class", y="count")
            st.plotly_chart(fig, use_container_width=True)

        # Predicted per-class counts from latest eval run (Ultralytics saves labels under labels dir)
        pred_dir = os.path.join(project_dir, eval_run, "labels")
        if os.path.isdir(pred_dir):
            pred_counts = {c: 0 for c in class_names}
            p_lbls = glob.glob(os.path.join(pred_dir, "*.txt"))
            for pf in p_lbls[:2000]:
                for line in open(pf, "r", encoding="utf-8", errors="ignore").read().strip().splitlines():
                    if not line:
                        continue
                    try:
                        cid = int(line.split()[0])
                        if 0 <= cid < len(class_names):
                            pred_counts[class_names[cid]] += 1
                    except Exception:
                        continue
            if any(v > 0 for v in pred_counts.values()):
                st.subheader("Predicted Class Distribution (from eval labels)")
                dpf = pd.DataFrame({"class": list(pred_counts.keys()), "count": list(pred_counts.values())}).sort_values("count", ascending=False)
                figp = px.bar(dpf, x="class", y="count")
                st.plotly_chart(figp, use_container_width=True)

                # GT vs Pred side-by-side
                st.subheader("GT vs Pred Counts (Dataset classes)")
                dcmp = dpf.merge(dfc, on="class", how="outer", suffixes=("_pred", "_gt")).fillna(0)
                figcmp = px.bar(dcmp.melt(id_vars="class", value_vars=["count_pred", "count_gt"], var_name="type", value_name="count"), x="class", y="count", color="type", barmode="group")
                st.plotly_chart(figcmp, use_container_width=True)

        # If evaluation outputs include PR/F1 curves per-class, show them
        pr_png = os.path.join(project_dir, eval_run, "PR_curve.png")
        f1_png = os.path.join(project_dir, eval_run, "F1_curve.png")
        if os.path.exists(pr_png) or os.path.exists(f1_png):
            st.subheader("Curves")
            colc1, colc2 = st.columns(2)
            if os.path.exists(pr_png):
                with colc1:
                    st.image(pr_png, caption="Precision-Recall Curve")
            if os.path.exists(f1_png):
                with colc2:
                    st.image(f1_png, caption="F1 vs Confidence")

        # Extra evaluation charts
        st.subheader("Prediction Confidence Histogram (sample)")
        pred_dir = os.path.join(project_dir, eval_run, "labels")
        if os.path.isdir(pred_dir):
            import glob as _glob
            import pandas as _pd
            confs = []
            for pf in _glob.glob(os.path.join(pred_dir, "*.txt"))[:500]:
                for line in open(pf, "r", encoding="utf-8", errors="ignore").read().strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            confs.append(float(parts[5]))
                        except Exception:
                            pass
            if confs:
                import numpy as _np
                hist = _np.histogram(confs, bins=20, range=(0, 1))
                dfh = _pd.DataFrame({"bin": [f"{b:.2f}" for b in hist[1][:-1]], "count": hist[0]})
                st.plotly_chart(px.bar(dfh, x="bin", y="count", title="Confidence Histogram"), use_container_width=True)

        # Add small bar chart of class distribution in validation labels
        import glob
        import pandas as pd
        lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        counts = {}
        for lf in lbl_files[:500]:
            for line in open(lf, "r", encoding="utf-8", errors="ignore").read().strip().splitlines():
                if not line:
                    continue
                try:
                    cid = int(line.split()[0])
                    cname = class_names[cid] if 0 <= cid < len(class_names) else "unk"
                    counts[cname] = counts.get(cname, 0) + 1
                except Exception:
                    continue
        if counts:
            st.subheader("Validation Label Class Distribution (sample)")
            dfc = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())}).sort_values("count", ascending=False)
            fig = px.bar(dfc, x="class", y="count")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Validation images/labels not found.")

    st.header("Next steps and improvements")
    st.markdown("""
    - Consider larger models (yolov8s/m) and longer training for better mAP.
    - Augmentations (mosaic, hsv, mixup) can improve robustness.
    - Address class imbalance with sampling or loss weighting if needed.
    - Improve data integrity (missing/empty labels) based on earlier checks.
    - Analyze failure clusters to refine augmentations or model architecture.
    """)


if __name__ == "__main__":
    main()


