# Streamlit Pages Documentation

This file explains each dashboard page in the `src/pages/` directory.

---

## Page: `1_About_Data.py`

**Purpose:**  
Interactive dashboard for traffic dataset analytics.

**Features:**
- Loads summary from `TrafficAnalyzer` and displays:
  - Class distribution as bar chart/pie chart
  - Per-image object stats (histogram)
  - Bounding box metrics (scatter, violin, or boxplots)
  - Co-occurrence matrix as heatmap
- Sample annotated images (grid/gallery)
- Expanders for detailed stats

**UI Elements:**
- Streamlit sidebar navigation
- Interactive matplotlib/plotly charts
- Image viewer

**How it works:**
- On load, calls `TrafficAnalyzer.summary()`
- Renders each analytic in a separate section
- Plots are interactive, with tooltips and zoom

**Extending:**
- Add new analytics by calling additional methods from `analyzer.py`
- Add more chart types or widgets for deeper insights

---

## Page: `3_Evaluation.py`

**Purpose:**  
Read-only visualization of evaluation results produced at container startup.

**Features:**
- Displays:
  - mAP, precision, recall, F1 metrics
  - PR and confusion matrix curves
  - Qualitative sample detections (images with predicted bboxes and classes)

**UI Elements:**
- Streamlit sidebar and main panel
- Metrics panels for key numbers
- Plots for curves
- Annotated image gallery

**How it works:**
- Loads cached outputs from `data/outputs/yolo_eval/` created by the startup pipeline
- Renders all outputs using Streamlit widgets

**Extending:**
- Add more evaluation metrics or qualitative outputs
- Allow tuning of training parameters via UI

---

**Adding New Pages:**
- Create a numbered `.py` file in `src/pages/` (e.g., `2_NewAnalytics.py`)
- Define a `main()` function with Streamlit UI code
- Import relevant modules from `src/`

---

**All pages are loaded automatically by `src/dashboard.py` and follow Streamlit best practices for modular UI design.**