"""
Failure analysis utilities for YOLO evaluation results.

Reads Ultralytics predictions.json produced by validation, compares with ground
truth labels, and produces:
- CSV summary of false positives and false negatives per class
- Example images highlighting FP (red) and FN (yellow) boxes

Outputs are written under the evaluation run directory, e.g.:
  /app/data/outputs/yolo_eval/failures/

Run:
  python src/analysis_failures.py \
    --data_yaml /app/data/traffic-detection-project/data.yaml \
    --eval_dir /app/data/outputs/yolo_eval
"""
import os
import json
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from dataset_utils import load_classes_from_yaml, parse_yolo_label


def _xywhn_to_xyxy(box: Dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x_c, y_c, w, h = box["x_center"], box["y_center"], box["width"], box["height"]
    x1 = int((x_c - w / 2.0) * img_w)
    y1 = int((y_c - h / 2.0) * img_h)
    x2 = int((x_c + w / 2.0) * img_w)
    y2 = int((y_c + h / 2.0) * img_h)
    return x1, y1, x2, y2


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _greedy_match(gt_boxes: List[Tuple[int, int, int, int, int]],
                  pred_boxes: List[Tuple[int, int, int, int, int, float]],
                  iou_thr: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Greedy IoU matching by confidence: returns matched indices for GT and predictions.
    gt_boxes: [(x1,y1,x2,y2,class_id)]
    pred_boxes: [(x1,y1,x2,y2,class_id,conf)]
    """
    matched_gt = set()
    matched_pr = set()
    # sort preds by confidence desc
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i][5], reverse=True)
    for pi in order:
        px1, py1, px2, py2, pcid, pconf = pred_boxes[pi]
        best_iou = 0.0
        best_gi = -1
        for gi, (gx1, gy1, gx2, gy2, gcid) in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if pcid != gcid:
                continue
            iou = _iou_xyxy((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched_pr.add(pi)
            matched_gt.add(best_gi)
    return list(matched_gt), list(matched_pr)


def analyze_failures(data_yaml: str, eval_dir: str, iou_thr: float = 0.5) -> pd.DataFrame:
    classes = load_classes_from_yaml(data_yaml)
    pred_json = os.path.join(eval_dir, "predictions.json")
    labels_dir = os.path.join(os.path.dirname(os.path.dirname(data_yaml)), "valid", "labels")
    images_dir = os.path.join(os.path.dirname(os.path.dirname(data_yaml)), "valid", "images")

    if not os.path.exists(pred_json):
        raise FileNotFoundError(f"predictions.json not found at {pred_json}")

    with open(pred_json, "r") as f:
        preds = json.load(f)

    failures_dir = os.path.join(eval_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)

    # Map image path (basename) -> predictions
    preds_by_image = {}
    for p in preds:
        # Ultralytics stores {'image_path': str, 'prediction': {'boxes': {'data': [...]}}} in some versions
        img_path = p.get("image") or p.get("image_path") or p.get("path")
        if not img_path:
            continue
        preds_by_image[os.path.basename(img_path)] = p

    rows = []
    for img_name, pobj in preds_by_image.items():
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # ground truth
        gt = parse_yolo_label(lbl_path) if os.path.exists(lbl_path) else []
        gt_xyxy = []
        for g in gt:
            x1, y1, x2, y2 = _xywhn_to_xyxy(g, w, h)
            gt_xyxy.append((x1, y1, x2, y2, int(g.get("class_id", 0))))

        # predictions
        # Try different prediction JSON structures
        pred_boxes = []
        boxes = None
        if isinstance(pobj, dict):
            if "prediction" in pobj and isinstance(pobj["prediction"], dict):
                boxes = pobj["prediction"].get("boxes", {}).get("data")
            if boxes is None:
                boxes = pobj.get("boxes") or pobj.get("detections")
        if boxes is None:
            continue
        for b in boxes:
            # Expected format [x1,y1,x2,y2,conf,cls]
            if isinstance(b, (list, tuple)) and len(b) >= 6:
                x1, y1, x2, y2 = map(int, b[:4])
                conf = float(b[4])
                cid = int(b[5])
            elif isinstance(b, dict):
                x1, y1, x2, y2 = map(int, b.get("xyxy", b.get("bbox", [0, 0, 0, 0]))[:4])
                conf = float(b.get("confidence", b.get("conf", 0)))
                cid = int(b.get("class", b.get("cls", 0)))
            else:
                continue
            pred_boxes.append((x1, y1, x2, y2, cid, conf))

        matched_gt, matched_pr = _greedy_match(gt_xyxy, pred_boxes, iou_thr=iou_thr)
        fp_indices = [i for i in range(len(pred_boxes)) if i not in matched_pr]
        fn_indices = [i for i in range(len(gt_xyxy)) if i not in matched_gt]

        # Draw a sample image if any failures
        drew = False
        canvas = img.copy()
        # False positives in red
        for i in fp_indices[:10]:
            x1, y1, x2, y2, cid, conf = pred_boxes[i]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"FP {classes[cid]} {conf:.2f}"
            cv2.putText(canvas, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            drew = True
        # False negatives in yellow
        for i in fn_indices[:10]:
            x1, y1, x2, y2, cid = gt_xyxy[i]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"FN {classes[cid]}"
            cv2.putText(canvas, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            drew = True

        if drew:
            out_path = os.path.join(failures_dir, f"fail_{os.path.splitext(img_name)[0]}.jpg")
            cv2.imwrite(out_path, canvas)

        # Aggregate counts per class
        fp_by_class: Dict[int, int] = {}
        fn_by_class: Dict[int, int] = {}
        for i in fp_indices:
            cid = pred_boxes[i][4]
            fp_by_class[cid] = fp_by_class.get(cid, 0) + 1
        for i in fn_indices:
            cid = gt_xyxy[i][4]
            fn_by_class[cid] = fn_by_class.get(cid, 0) + 1

        for cid, cnt in fp_by_class.items():
            rows.append({"image": img_name, "class": classes[cid], "type": "FP", "count": cnt})
        for cid, cnt in fn_by_class.items():
            rows.append({"image": img_name, "class": classes[cid], "type": "FN", "count": cnt})

    df = pd.DataFrame(rows)
    if not df.empty:
        df_summary = df.groupby(["class", "type"], as_index=False)["count"].sum().sort_values(["type", "count"], ascending=[True, False])
        df_summary.to_csv(os.path.join(failures_dir, "failures_summary.csv"), index=False)
        df.to_csv(os.path.join(failures_dir, "failures_detailed.csv"), index=False)
        return df_summary
    else:
        # still create folder marker
        open(os.path.join(failures_dir, "README.txt"), "w").write("No failures detected or predictions unavailable.")
        return pd.DataFrame(columns=["class", "type", "count"])


def parse_args():
    p = argparse.ArgumentParser(description="Analyze YOLO evaluation failures")
    p.add_argument("--data_yaml", default="/app/data/traffic-detection-project/data.yaml")
    p.add_argument("--eval_dir", default="/app/data/outputs/yolo_eval")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = analyze_failures(args.data_yaml, args.eval_dir, iou_thr=args.iou)
    print(summary.head() if not summary.empty else "No failures summary produced.")


