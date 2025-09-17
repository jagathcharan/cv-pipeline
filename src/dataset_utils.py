import os
import glob
import pandas as pd
from typing import List, Dict, Tuple
import yaml

def get_image_label_files(data_dir: str) -> Dict[str, List[str]]:
    """
    Get image and label file paths.
    
    Args:
        data_dir (str): Root data directory
    
    Returns:
        Dict[str, List[str]]: {'images': [...], 'labels': [...]}
    """
    images = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    labels = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    return {"images": images, "labels": labels}


def load_classes_from_yaml(yaml_file: str) -> list:
    """
    Load class names from YOLO data.yaml file
    
    Args:
        yaml_file (str): Path to data.yaml
    
    Returns:
        List[str]: Class names in order
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return [data['names'][i] for i in range(len(data['names']))]


def parse_yolo_label(label_file: str) -> List[Dict]:
    """
    Parse YOLO label file into structured data
    
    Args:
        label_file (str): Path to .txt YOLO label
    
    Returns:
        List[Dict]: [{'class_id': int, 'x': float, 'y': float, 'w': float, 'h': float}]
    """
    annotations = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                annotations.append({
                    "class_id": int(parts[0]),
                    "x_center": float(parts[1]),
                    "y_center": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4])
                })
    return annotations


def summarize_class_distribution(labels: List[str], class_names: List[str]) -> pd.DataFrame:
    """
    Count number of instances per class
    
    Args:
        labels (List[str]): List of label file paths
        class_names (List[str]): List of class names
    
    Returns:
        pd.DataFrame: Class frequency table
    """
    class_counts = {cls: 0 for cls in class_names}
    for lbl in labels:
        annots = parse_yolo_label(lbl)
        for a in annots:
            cls_name = class_names[a["class_id"]]
            class_counts[cls_name] += 1
    df = pd.DataFrame.from_dict(class_counts, orient="index", columns=["count"])
    df.sort_values("count", ascending=False, inplace=True)
    # Ensure Arrow-safe index type for Streamlit; keep index as string labels
    df.index = df.index.map(str)
    return df


def verify_yolo_dataset(
    root_dir: str,
    class_names: List[str],
) -> Dict[str, object]:
    """
    Verify YOLO annotation files under a dataset root directory.

    Checks performed per label file:
    - Exactly 5 values per line (class_id x_center y_center width height)
    - Numeric parsing succeeds
    - class_id within [0, len(class_names)-1]
    - x_center, y_center, width, height in [0,1]
    - width > 0, height > 0
    - Corresponding image file exists and is readable

    Returns a report dict with counts and a DataFrame of issues.
    """
    image_glob = os.path.join(root_dir, "**", "*.jpg")
    label_glob = os.path.join(root_dir, "**", "*.txt")

    images = set(glob.glob(image_glob, recursive=True))
    labels = sorted(glob.glob(label_glob, recursive=True))

    def image_for_label(lbl_path: str) -> str:
        base = lbl_path.rsplit(os.sep + "labels" + os.sep, 1)
        if len(base) == 2:
            prefix = base[0] + os.sep + "images" + os.sep + base[1]
        else:
            prefix = lbl_path.replace("labels", "images")
        return (
            prefix.replace(".txt", ".jpg").replace(".jpeg", ".jpg").replace(".png", ".jpg")
        )

    issues: List[Dict[str, object]] = []
    total_lines = 0
    for lbl in labels:
        try:
            with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().strip().splitlines()
        except Exception as e:
            issues.append({
                "file": lbl,
                "type": "read_error",
                "message": str(e),
                "line_number": None,
                "line": None,
            })
            continue

        if len(lines) == 0:
            issues.append({
                "file": lbl,
                "type": "empty_label",
                "message": "Label file is empty",
                "line_number": None,
                "line": None,
            })

        img_path = image_for_label(lbl)
        if not os.path.exists(img_path):
            issues.append({
                "file": lbl,
                "type": "missing_image",
                "message": f"Image not found for label: {img_path}",
                "line_number": None,
                "line": None,
            })

        for idx, line in enumerate(lines, start=1):
            total_lines += 1
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append({
                    "file": lbl,
                    "type": "format_error",
                    "message": "Expected 5 values per line",
                    "line_number": idx,
                    "line": line,
                })
                continue
            try:
                cid = int(parts[0])
                x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            except Exception:
                issues.append({
                    "file": lbl,
                    "type": "parse_error",
                    "message": "Failed to parse numeric values",
                    "line_number": idx,
                    "line": line,
                })
                continue

            if not (0 <= cid < len(class_names)):
                issues.append({
                    "file": lbl,
                    "type": "class_id_out_of_range",
                    "message": f"class_id {cid} outside [0,{len(class_names)-1}]",
                    "line_number": idx,
                    "line": line,
                })

            for name, val in ("x_center", x), ("y_center", y), ("width", w), ("height", h):
                if not (0.0 <= val <= 1.0):
                    issues.append({
                        "file": lbl,
                        "type": "value_out_of_bounds",
                        "message": f"{name}={val} not in [0,1]",
                        "line_number": idx,
                        "line": line,
                    })
            if w <= 0 or h <= 0:
                issues.append({
                    "file": lbl,
                    "type": "non_positive_dims",
                    "message": f"width={w}, height={h} must be > 0",
                    "line_number": idx,
                    "line": line,
                })

    # Images without labels
    expected_labels = set()
    for img in images:
        base = img.rsplit(os.sep + "images" + os.sep, 1)
        if len(base) == 2:
            prefix = base[0] + os.sep + "labels" + os.sep + base[1]
        else:
            prefix = img.replace("images", "labels")
        expected_labels.add(prefix.rsplit('.', 1)[0] + ".txt")
    labels_set = set(labels)
    imgs_without_label = [img for img in images if (img.rsplit(os.sep + "images" + os.sep, 1)[0] + os.sep + "labels" + os.sep + os.path.basename(img).rsplit('.',1)[0] + ".txt") not in labels_set]
    for img in imgs_without_label:
        issues.append({
            "file": img,
            "type": "image_without_label",
            "message": "Image has no corresponding label file",
            "line_number": None,
            "line": None,
        })

    issues_df = pd.DataFrame(issues)
    if not issues_df.empty:
        # Force textual columns to string to avoid Arrow int64 inference
        for col in ["file", "type", "message", "line"]:
            if col in issues_df.columns:
                issues_df[col] = issues_df[col].astype(str)

    report: Dict[str, object] = {
        "total_label_files": len(labels),
        "total_image_files": len(images),
        "total_annotation_lines": total_lines,
        "num_issues": int(len(issues_df)),
        "issues": issues_df,
    }
    return report
