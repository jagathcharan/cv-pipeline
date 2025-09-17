# Project Architecture

This document describes the architecture and data flow of the Traffic Detection Analysis project. It covers the relationships between code modules, dashboard pages, scripts, and their deployment in Docker.

---

## High-Level Diagram

```
+-------------------+    +------------------+    +-------------------+
|   Data Analyzer   |--->|   Streamlit UI   |<---| YOLOv8 Pipeline   |
+-------------------+    +------------------+    +-------------------+
         |                      |                       |
         |                      |                       |
         v                      v                       v
      Dockerized Environment (All dependencies, dataset, outputs)
```

---

## Component Roles

### 1. Data Analyzer (`src/analyzer.py`)
- Loads YOLO-format dataset config (`data.yaml`)
- Computes class stats, bounding box stats, co-occurrence, etc.
- Supplies analytics to dashboard and scripts.

### 2. Analysis Utilities (`src/analysis.py`)
- Helper functions for loading annotations, calculating metrics, plotting.
- Used by analyzer, dashboard, and scripts.

### 3. Streamlit Dashboard (`src/dashboard.py` + `src/pages/`)
- Multipage UI:
  - **About Data:** Interactive dataset analysis.
  - **Evaluation:** Read-only results visualization. Training/evaluation is performed once at startup by the container entrypoint.
- Entry point: `dashboard.py`
- Pages: `src/pages/` (modular, extendable)

### 4. YOLOv8 Pipeline (`src/train_yolo.py`, `src/evaluate_yolo.py`, `src/pipeline.py`)
- Training: Trains YOLOv8 model with dataset.
- Evaluation: Computes metrics and outputs visualizations.
- Pipeline: End-to-end analysis, training, and evaluation.

### 5. Docker Environment
- Ensures reproducibility and consistent paths.
- Handles dataset preparation, dependency installation, one-time pipeline execution (`start.sh`), and app runtime.

---

## Data Flow

1. **Dataset Preparation**
   - `download.sh` fetches/extracts dataset to `data/traffic-detection-project/`
   - Dataset referenced by all modules via `data.yaml`

2. **Analytics Workflow**
   - `analyzer.py` and `analysis.py` load and process dataset, produce stats.
   - Results displayed in Streamlit (About Data page).

3. **Training & Evaluation**
   - On first container start after an image rebuild, `start.sh` clears previous outputs and runs training (1 epoch) and evaluation.
   - Metrics and plots are saved under `data/outputs/yolo_eval/` and visualized in the dashboard.

---

## Extensibility

- New dashboard pages: add Python files to `src/pages/` (number prefix for order).
- New analytics: add functions/classes in `src/analyzer.py` and `src/analysis.py`.
- Model changes: adjust scripts or dashboard logic.

---

## Deployment

- **Docker Compose:**  
  One command starts dataset prep, model pipeline, and dashboard.
  ```bash
  docker compose up --build
  ```
- **Manual Python:**  
  Run scripts/Streamlit locally after preparing dataset and installing requirements.

---

## Summary

This modular, containerized architecture enables robust, reproducible traffic object detection analysis, making it easy to add new features, models, or analytics.