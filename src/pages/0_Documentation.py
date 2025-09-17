import streamlit as st


def main():
    st.set_page_config(page_title="Documentation | Traffic Detection", layout="wide")
    st.title("Project Documentation")
    st.markdown(
        """
        ### Overview
        - Focus on object detection with YOLO (bounding boxes), no segmentation.
        - Automated end-to-end pipeline across pages: About Data and Evaluation only.

        ### Model Choice & Rationale
        - **YOLOv8n** chosen for speed and solid baseline on CPU.
        - One-stage detector with CSP backbone and PAN/FPN neck.

        ### Metrics
        - mAP@0.5 and mAP@0.5:0.95 for detection performance.
        - Confusion matrix, PR curves, and F1 vs confidence for diagnostic insights.

        ### How to Run
        ```bash
        ./run.sh
        # open http://localhost:8501
        ```

        ### Data
        - Expected under `/app/data/traffic-detection-project` mounted from `./data`.
        - Use the About Data page for distribution, co-occurrence, and sample visualization.

        ### Automated Training & Evaluation
        - Training is triggered automatically (CPU-friendly, 1 epoch) from the Evaluation page.
        - Best weights are saved to `/app/data/outputs/yolo_train/weights/best.pt`.
        - Evaluation runs automatically on the best weights and renders reports.

        ### Evaluation
        - No inputs required. Runs on the latest trained weights and shows metrics and plots.

        ### Reproducibility
        - Dockerized, with `.dockerignore` to preserve dependency cache.
        - Requirements installed once; source changes do not re-install deps.

        ### Class Matching in Evaluation
        - Evaluation checks the intersection between model classes and dataset classes.
        - If no overlap, evaluation is blocked to avoid misleading metrics.

        ### UI Scope
        - Pages: Documentation, About Data, Evaluation (outputs only).
        - No manual training/evaluation inputs; UI displays outputs and reports.
        """
    )


if __name__ == "__main__":
    main()


