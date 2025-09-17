# Source Folder Component Documentation

This document provides in-depth documentation for every module in the `src/` directory.

---

## 1. `analyzer.py`

**Purpose:**  
Traffic dataset analytics using YOLO-format annotation. Class-based design for extensibility.

**Main Class:**  
`TrafficAnalyzer`
- **Initialization:**  
  Loads dataset config (`data.yaml`), reads annotations and images.
- **Key Methods:**
  - `compute_class_distribution()`: Returns class count and proportion.
  - `image_stats()`: Per-image object counts, class diversity, avg bbox stats.
  - `bbox_stats()`: Distribution (width, height, aspect ratio, area) for bboxes.
  - `cooccurrence_matrix()`: Matrix of class co-occurrences per image.
  - `show_examples(num_images)`: Plots sample images with bboxes.
  - `summary()`: Aggregates all stats for dashboard.

**Usage:**
```python
from src.analyzer import TrafficAnalyzer
analyzer = TrafficAnalyzer("data/traffic-detection-project/data.yaml")
summary = analyzer.summary()
```

---

## 2. `analysis.py`

**Purpose:**  
Reusable utility functions for dataset analysis and visualization.

**Functions:**
- `load_annotations(yaml_path)`: Loads YOLO-format labels.
- `calc_bbox_metrics(bboxes)`: Computes bbox area, aspect ratio, etc.
- `plot_class_distribution(classes, class_names)`: Plots using matplotlib/seaborn.
- `get_image_samples(dataset_dir, num_samples)`: Random image sampling.

**Usage:**
```python
from src.analysis import load_annotations
annotations = load_annotations("data/traffic-detection-project/data.yaml")
```

---

## 3. `dashboard.py`

**Purpose:**  
Streamlit multipage app entry point.  
Initializes sidebar, loads project overview, and dynamically loads page modules.
Shows one-time pipeline status; actual pipeline execution is handled at container startup by `start.sh`.

**Key Actions:**
- Sets up Streamlit navigation.
- Displays summary and links.
- Loads `src/pages/` modules.

**Usage:**  
Run with Streamlit:
```bash
streamlit run src/dashboard.py
```

---

## 4. `train_yolo.py`

**Purpose:**  
Script for YOLOv8 training.

**Arguments:**  
- `--epochs`: Number of epochs (default: 1).
- `--data`: Path to dataset yaml.
- `--model`: Model type (default: yolov8n).
- `--imgsz`: Image size (default: 640).

**Behavior:**  
Loads dataset and model, runs training, saves weights/logs.

**Usage:**
```bash
python src/train_yolo.py --epochs 1 --data data/traffic-detection-project/data.yaml
```

---

## 5. `evaluate_yolo.py`

**Purpose:**  
Script for evaluating YOLOv8 models.

**Arguments:**  
- `--weights`: Path to trained weights.
- `--data`: Path to dataset yaml.
- `--imgsz`: Image size.

**Behavior:**  
Loads model and validation set, computes metrics and outputs visualizations. Persists a `metrics.json` to the eval run directory for UI reuse.

**Usage:**
```bash
python src/evaluate_yolo.py --weights runs/train/exp/weights/best.pt
```

---

## 6. `pipeline.py`

**Purpose:**  
Runs training and evaluation in sequence for full automation. Called by `start.sh` at container startup.

**Behavior:**  
Loads dataset, computes analytics, trains model, evaluates, and saves all outputs.

**Usage:**
```bash
python src/pipeline.py
```

---

**Notes:**
- All dataset paths are `/app/data/traffic-detection-project/` for Docker compatibility.
- All outputs are saved under `data/` for reproducibility.
- Modules use Python docstrings and follow PEP8 conventions.