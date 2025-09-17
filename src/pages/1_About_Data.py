import os
import random
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import cv2
from analyzer import TrafficDatasetAnalyzer
from dataset_utils import verify_yolo_dataset


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "class" or out[col].dtype == object:
            out[col] = out[col].astype(str)
    if out.index.dtype != object:
        try:
            out.index = out.index.map(str)
        except Exception:
            pass
    return out
from typing import List


def show_image_with_bboxes(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    if image is None:
        st.warning(f"Image not found: {image_path}")
        return
    height, width = image.shape[:2]
    from dataset_utils import parse_yolo_label  # local import to avoid circular issues

    annots = parse_yolo_label(label_path)
    for ann in annots:
        cls_id = ann['class_id']
        x_center = ann['x_center'] * width
        y_center = ann['y_center'] * height
        w = ann['width'] * width
        h = ann['height'] * height
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls_id]
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    _, img_bytes = cv2.imencode('.jpg', image)
    st.image(img_bytes.tobytes(), channels="BGR")


def _metric_card(label: str, value: str):
    st.metric(label, value)


@st.cache_data(show_spinner=False)
def _cached_analyze(data_dir: str, splits: List[str]):
    analyzer = TrafficDatasetAnalyzer(data_dir, splits=splits)  # CPU-friendly
    return analyzer, analyzer.analyze()


def main():
    st.set_page_config(page_title="About Data | Traffic Detection", layout="wide")
    st.title("Dataset Overview")
    DATA_DIR = "/app/data/traffic-detection-project"

    with st.sidebar:
        st.header("Filters")
        # Fixed to train/valid for minimal UI
        available = [s for s in ["train", "valid"] if os.path.isdir(os.path.join(DATA_DIR, s))]
        show_splits = available
        top_k_pairs = 15

    analyzer, results = _cached_analyze(DATA_DIR, show_splits)
    class_names = analyzer.class_names
    splits = [s for s in results.keys() if s in show_splits]

    st.markdown("""
    - **Task focus:** Object detection using bounding box annotations (YOLO). Semantic segmentation is ignored.
    - **Splits analyzed:** train and validation
    - **Classes:** {}
    """.format(
        ", ".join(class_names)
    ))

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Split Overview", "Class Distribution", "Per-image Stats", "BBox Stats", "Co-occurrence", "Examples", "Verification"
    ])

    with tab1:
        st.header("Data Split Overview")
        split_sizes = {split: len(results[split].images) for split in splits}
        st.bar_chart(split_sizes)
        st.write("Split sizes:", split_sizes)

        st.subheader("Integrity Checks")
        for split in splits:
            st.markdown(f"**{split.capitalize()}**")
            integ = results[split].integrity
            cols = st.columns(5)
            with cols[0]:
                _metric_card("Images", str(integ.get("total_images", 0)))
            with cols[1]:
                _metric_card("Labels", str(integ.get("total_labels", 0)))
            with cols[2]:
                _metric_card("Missing Labels", str(integ.get("missing_label_files", 0)))
            with cols[3]:
                _metric_card("Empty Labels", str(integ.get("empty_label_files", 0)))
            with cols[4]:
                _metric_card("Images w/o Label", str(integ.get("images_without_label", 0)))

    with tab2:
        st.header("Class Distribution")
        for split in splits:
            st.subheader(f"{split.capitalize()} Split")
            df = results[split].class_distribution
            # Ensure Arrow-friendly types
            if df.index.dtype != object:
                df = df.copy()
                df.index = df.index.map(str)
            st.dataframe(_arrow_safe(df))
            fig = px.bar(df, x=df.index, y="count", title=f"Class Distribution - {split}")
            st.plotly_chart(fig, use_container_width=True)
            pie = px.pie(df, names=df.index, values="count", title=f"Class Proportion - {split}")
            st.plotly_chart(pie, use_container_width=True)
            # Cumulative distribution
            cdf = df.copy()
            cdf["cum_pct"] = (cdf["count"].sort_values(ascending=False).cumsum() / cdf["count"].sum())
            cdf = cdf.sort_values("cum_pct", ascending=True)
            fig = px.line(cdf, y="cum_pct", title=f"Cumulative Share of Instances by Class - {split}")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Per-image Annotation Stats")
        for split in splits:
            st.subheader(f"{split.capitalize()} Split")
            ann_counts = results[split].object_counts_per_image
            st.write(
                f"Mean: {pd.Series(ann_counts).mean():.2f}, "
                f"Median: {pd.Series(ann_counts).median():.2f}, "
                f"Max: {max(ann_counts) if ann_counts else 0}"
            )
            fig, ax = plt.subplots()
            ax.hist(ann_counts, bins=range(0, max(ann_counts)+2 if ann_counts else 2), color='skyblue')
            ax.set_xlabel("Objects per image")
            ax.set_ylabel("Image count")
            st.pyplot(fig)

    with tab4:
        st.header("Bounding Box Statistics")
        for split in splits:
            st.subheader(f"{split.capitalize()} Split")
            stats = results[split].bbox_stats
            if stats.empty:
                st.info("No bounding boxes found.")
                continue
            # Cast potential object columns to string to avoid Arrow casting issues
            stats_display = stats.copy()
            if "class" in stats_display.columns:
                stats_display["class"] = stats_display["class"].astype(str)
            st.dataframe(_arrow_safe(stats_display.describe(include="all")))
            fig = px.violin(stats, x="class", y="rel_area", box=True, points=False, title=f"Relative Area Distribution - {split}")
            st.plotly_chart(fig, use_container_width=True)
            fig = px.violin(stats, x="class", y="aspect_ratio", box=True, points=False, title=f"Aspect Ratio Distribution - {split}")
            st.plotly_chart(fig, use_container_width=True)
            # Joint distribution heatmap of rel_area vs aspect_ratio
            heat = px.density_heatmap(stats, x="aspect_ratio", y="rel_area", nbinsx=40, nbinsy=40,
                                      title=f"BBox Area vs Aspect Ratio Density - {split}")
            st.plotly_chart(heat, use_container_width=True)

    with tab5:
        st.header("Class Co-occurrence")
        for split in splits:
            st.subheader(f"{split.capitalize()} Split")
            co = results[split].cooccurrence
            fig = ff.create_annotated_heatmap(
                z=co.values,
                x=list(co.columns),
                y=list(co.index),
                colorscale="Viridis",
                showscale=True,
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            # Top co-occurring pairs bar chart
            pairs = []
            for i, a in enumerate(co.index):
                for j, b in enumerate(co.columns):
                    if j > i:
                        pairs.append(((a, b), int(co.iloc[i, j])))
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k_pairs]
            if pairs:
                df_pairs = pd.DataFrame({
                    "pair": [f"{a} + {b}" for (a, b), _c in pairs],
                    "count": [c for _p, c in pairs]
                })
                st.plotly_chart(px.bar(df_pairs, x="pair", y="count", title=f"Top {top_k_pairs} Co-occurring Class Pairs - {split}"), use_container_width=True)

    with tab6:
        st.header("Example Images with Bounding Boxes (Rare & Random)")
        rarity = st.slider("Rarity threshold (counts < threshold are rare)", min_value=1, max_value=200, value=50)
        rare = analyzer.find_rare_examples("train", max_per_class=3, rarity_threshold=rarity)
        for cls, pairs in rare.items():
            if not pairs:
                continue
            st.subheader(f"Rare class: {cls}")
            cols = st.columns(min(3, len(pairs)))
            for idx, (img_path, lbl_path) in enumerate(pairs):
                with cols[idx % len(cols)]:
                    show_image_with_bboxes(img_path, lbl_path, class_names)
        st.subheader("Random Examples")
        train_labels = results["train"].labels
        random.shuffle(train_labels)
        for lbl_path in train_labels[:6]:
            img_path = lbl_path.replace("labels", "images").replace(".txt", ".jpg")
            show_image_with_bboxes(img_path, lbl_path, class_names)

    with tab7:
        st.header("YOLO Annotation Verification")
        st.info("Checks label formatting, class ids, normalized coordinates, and image-label pairing.")
        report = verify_yolo_dataset(DATA_DIR, class_names)
        cols = st.columns(4)
        with cols[0]:
            _metric_card("Images", str(report.get("total_image_files", 0)))
        with cols[1]:
            _metric_card("Label files", str(report.get("total_label_files", 0)))
        with cols[2]:
            _metric_card("Annot. lines", str(report.get("total_annotation_lines", 0)))
        with cols[3]:
            _metric_card("Issues found", str(report.get("num_issues", 0)))

        issues_df = report.get("issues")
        if issues_df is not None and not issues_df.empty:
            # Ensure string dtypes for Arrow
            issues_df = issues_df.copy()
            for col in ["file", "type", "message", "line"]:
                if col in issues_df.columns:
                    issues_df[col] = issues_df[col].astype(str)
            st.subheader("Issues Detail")
            st.dataframe(_arrow_safe(issues_df))
            # Quick breakdown by type
            st.subheader("Issues by Type")
            by_type = issues_df.groupby("type").size().reset_index(name="count")
            by_type["type"] = by_type["type"].astype(str)
            st.plotly_chart(px.bar(by_type, x="type", y="count", title="Issue Counts by Type"), use_container_width=True)
        else:
            st.success("No issues detected. Dataset annotations look consistent.")


if __name__ == "__main__":
    main()


