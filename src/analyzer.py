"""
Class-based dataset analyzer focused on object detection with bounding-box annotations.

This module parses YOLO-style datasets, summarizes structure, computes per-split
class distributions, per-image object densities, bounding box statistics
(relative area and aspect ratio), class co-occurrence, and basic integrity checks.

PEP8 compliant with clear docstrings for public methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from dataset_utils import (
    get_image_label_files,
    load_classes_from_yaml,
    parse_yolo_label,
    summarize_class_distribution,
)


@dataclass
class SplitSummary:
    """Container for per-split summary and analytics results."""

    images: List[str]
    labels: List[str]
    class_distribution: pd.DataFrame
    object_counts_per_image: List[int]
    bbox_stats: pd.DataFrame
    cooccurrence: pd.DataFrame
    integrity: Dict[str, int]


class TrafficDatasetAnalyzer:
    """Analyzer for traffic object detection datasets using YOLO bbox annotations.

    Attributes
    ----------
    data_dir: str
        Root directory containing the dataset (train/valid/test folders and data.yaml).
    class_names: List[str]
        List of class names loaded from YAML.
    splits: List[str]
        Dataset splits to analyze. Defaults to ["train", "valid", "test"].
    """

    def __init__(self, data_dir: str, splits: Optional[List[str]] = None) -> None:
        self.data_dir = data_dir
        self.splits = splits or ["train", "valid", "test"]
        self.class_names = load_classes_from_yaml(os.path.join(self.data_dir, "data.yaml"))

    # ----------------------------- Public API ----------------------------- #
    def analyze(self) -> Dict[str, SplitSummary]:
        """Run analysis for all configured splits.

        Returns
        -------
        Dict[str, SplitSummary]
            Mapping from split name to its computed summary.
        """
        results: Dict[str, SplitSummary] = {}
        for split in self.splits:
            split_dir = os.path.join(self.data_dir, split)
            files = get_image_label_files(split_dir)
            images = files.get("images", [])
            labels = files.get("labels", [])

            class_dist = summarize_class_distribution(labels, self.class_names)
            object_counts = [len(safe_parse(lbl)) for lbl in labels]

            bbox_stats = self._compute_bbox_stats(labels)
            cooccurrence = self._compute_cooccurrence(labels)
            integrity = self._compute_integrity(images, labels)

            results[split] = SplitSummary(
                images=images,
                labels=labels,
                class_distribution=class_dist,
                object_counts_per_image=object_counts,
                bbox_stats=bbox_stats,
                cooccurrence=cooccurrence,
                integrity=integrity,
            )

        return results

    def find_rare_examples(self, split: str, max_per_class: int = 3, rarity_threshold: Optional[int] = None) -> Dict[str, List[Tuple[str, str]]]:
        """Find rare class examples in a split.

        Parameters
        ----------
        split: str
            Split name to search within.
        max_per_class: int
            Maximum number of examples to return per class.
        rarity_threshold: Optional[int]
            If provided, consider classes with count < threshold as rare; otherwise
            use the lowest 20% of classes by frequency (at least one class).

        Returns
        -------
        Dict[str, List[Tuple[str, str]]]
            Mapping from class name to list of (image_path, label_path).
        """
        split_dir = os.path.join(self.data_dir, split)
        files = get_image_label_files(split_dir)
        labels = files.get("labels", [])

        class_dist = summarize_class_distribution(labels, self.class_names)
        counts = class_dist["count"].to_numpy()
        if rarity_threshold is None:
            # bottom 20% by frequency, ensure at least 1 class selected
            cutoff = np.percentile(counts, 20) if len(counts) > 0 else 0
            rarity_threshold = max(int(cutoff), 1)

        rare_classes = [c for c, cnt in class_dist["count"].items() if cnt < rarity_threshold]
        rare_map: Dict[str, List[Tuple[str, str]]] = {c: [] for c in rare_classes}

        for lbl_path in labels:
            annots = safe_parse(lbl_path)
            for cls in list(rare_map.keys()):
                cls_id = self.class_names.index(cls)
                if any(a.get("class_id") == cls_id for a in annots):
                    img_path = _label_to_image(lbl_path)
                    if os.path.exists(img_path):
                        rare_map[cls].append((img_path, lbl_path))
                    if len(rare_map[cls]) >= max_per_class:
                        # stop collecting for this class
                        pass
        return rare_map

    # ---------------------------- Computations ---------------------------- #
    def _compute_bbox_stats(self, label_files: List[str]) -> pd.DataFrame:
        """Compute bounding box relative area and aspect ratio statistics per class.

        Returns a DataFrame with columns: class, rel_area, aspect_ratio
        """
        records: List[Tuple[str, float, float]] = []
        for lbl in label_files:
            annotations = safe_parse(lbl)
            for ann in annotations:
                cls_name = self.class_names[ann["class_id"]]
                w_rel = float(ann["width"])  # width normalized [0,1]
                h_rel = float(ann["height"])  # height normalized [0,1]
                rel_area = max(w_rel * h_rel, 0.0)
                aspect_ratio = float(w_rel / h_rel) if h_rel > 0 else np.nan
                records.append((cls_name, rel_area, aspect_ratio))

        if not records:
            return pd.DataFrame(columns=["class", "rel_area", "aspect_ratio"])

        df = pd.DataFrame(records, columns=["class", "rel_area", "aspect_ratio"])
        # Ensure 'class' is string dtype to avoid Arrow attempting int64
        df["class"] = df["class"].astype(str)
        return df

    def _compute_cooccurrence(self, label_files: List[str]) -> pd.DataFrame:
        """Compute class co-occurrence matrix per image."""
        num_classes = len(self.class_names)
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        for lbl in label_files:
            annots = safe_parse(lbl)
            present_ids = sorted(set(a.get("class_id") for a in annots))
            for i in range(len(present_ids)):
                for j in range(i, len(present_ids)):
                    ci, cj = present_ids[i], present_ids[j]
                    matrix[ci, cj] += 1
                    if ci != cj:
                        matrix[cj, ci] += 1

        co_df = pd.DataFrame(matrix, index=self.class_names, columns=self.class_names)
        return co_df

    def _compute_integrity(self, images: List[str], labels: List[str]) -> Dict[str, int]:
        """Compute simple integrity counters such as missing labels or empty files."""
        missing_label = 0
        empty_label = 0
        for lbl in labels:
            try:
                content = safe_parse(lbl)
                if len(content) == 0:
                    empty_label += 1
            except Exception:
                missing_label += 1

        # image without label pair
        label_set = set(labels)
        expected_label_paths = set(_image_to_label(p) for p in images)
        images_without_label = len([p for p in images if _image_to_label(p) not in label_set])

        return {
            "missing_label_files": missing_label,
            "empty_label_files": empty_label,
            "images_without_label": images_without_label,
            "total_images": len(images),
            "total_labels": len(labels),
        }


# ------------------------------ Helpers ------------------------------ #

def safe_parse(label_path: str) -> List[dict]:
    """Parse a YOLO label file and return annotations; never raises, returns [] on error."""
    try:
        return parse_yolo_label(label_path)
    except Exception:
        return []


def _label_to_image(lbl_path: str) -> str:
    """Infer image path from label path by replacing directories and extension."""
    base = lbl_path.rsplit(os.sep + "labels" + os.sep, 1)
    if len(base) == 2:
        prefix = base[0] + os.sep + "images" + os.sep + base[1]
    else:
        prefix = lbl_path.replace("labels", "images")
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = prefix.replace(".txt", ext)
        if os.path.exists(candidate):
            return candidate
    return prefix.replace(".txt", ".jpg")


def _image_to_label(img_path: str) -> str:
    """Infer label path from image path by replacing directories and extension."""
    base = img_path.rsplit(os.sep + "images" + os.sep, 1)
    if len(base) == 2:
        prefix = base[0] + os.sep + "labels" + os.sep + base[1]
    else:
        prefix = img_path.replace("images", "labels")
    return (
        prefix.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
    )


