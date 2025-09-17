# Usage Guide

This document walks you through using the Traffic Detection Analysis project, both via the dashboard and command line scripts.

---

## 1. Using the Dockerized Dashboard (Recommended)

1. **Start the system:**
   ```bash
   docker-compose up --build
   ```
   - This builds the container, prepares the dataset, runs the one-time training+evaluation pipeline, and launches the Streamlit dashboard at [http://localhost:8501](http://localhost:8501).

2. **Dashboard Pages:**
   - **About Data:**  
     Analyze dataset statistics (class distribution, bbox stats, samples).
   - **Evaluation:**  
     Read-only visualization of the evaluation reports produced at startup (no buttons or dropdowns).

---

## 2. Using Scripts Manually

1. **Prepare Dataset:**
   ```bash
   bash download.sh
   ```
   - Or manually copy/extract dataset to `data/traffic-detection-project/`.
   - Kaggle API setup (if needed): set `KAGGLE_USERNAME` and `KAGGLE_KEY` env vars or place `kaggle.json` in `~/.kaggle/`.

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Analysis:**
   ```bash
   python src/analyzer.py
   ```

4. **Train Model:**
   ```bash
   python src/train_yolo.py --epochs 1 --data data/traffic-detection-project/data.yaml
   ```

5. **Evaluate Model:**
   ```bash
   python src/evaluate_yolo.py --weights runs/train/exp/weights/best.pt
   ```

6. **Run End-to-End Pipeline:**
   ```bash
   python src/pipeline.py
   ```

7. **Run Failure Analysis:**
   ```bash
   python src/analysis_failures.py \
     --data_yaml data/traffic-detection-project/data.yaml \
     --eval_dir data/outputs/yolo_eval
   ```

---

## 3. One-time pipeline semantics

- The image stores a build identifier. On first start after a rebuild, the container clears `data/outputs/` and runs the pipeline once, then serves the app.
- To force a fresh run without rebuilding, delete `data/.markers/build-id` (or the outputs directories) and restart.

Expected first-run time (CPU): a few minutes depending on dataset size; subsequent starts are instant.

## 4. Troubleshooting

- **Dataset Preparation Fails:**
  - Ensure `traffic-detection-project.zip` exists under `data/`.
  - Use correct Kaggle credentials for auto-download.
  - Mount your own dataset to `/app/data` if needed.

- **Port in Use:**  
  Change `STREAMLIT_SERVER_PORT` in Docker Compose or `run.sh`.

- **Script Errors:**  
  Double-check Python version (>3.8) and installed dependencies.

---

## 5. Extending/Customizing

- **Add dashboard pages:**  
  Add numbered `.py` files to `src/pages/` and define new analytics or visualization.

- **Change training settings:**  
  Edit `src/train_yolo.py` and `src/evaluate_yolo.py` for epochs, image size, model type.

- **Use custom dataset:**  
  Place your own YOLO-format dataset under `data/traffic-detection-project/` and update `data.yaml`.

---

**For development standards, see `DEVELOPER_GUIDE.md`.**
