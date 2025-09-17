import os
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import plotly.express as px
from dataset_utils import get_image_label_files, load_classes_from_yaml, parse_yolo_label, summarize_class_distribution
from visualization import plot_class_distribution


def draw_bboxes(image_path, label_path, class_names, save_path=None):
    """
    Draw bounding boxes on image from YOLO label file.

    Args:
        image_path (str): Path to image file
        label_path (str): Path to YOLO label file
        class_names (list): List of class names
        save_path (str, optional): Path to save the image with bboxes. If None, show image.
    """
    abs_image_path = os.path.abspath(image_path)
    abs_label_path = os.path.abspath(label_path)
    print(f"Drawing bboxes - image: {abs_image_path}, label: {abs_label_path}")

    image = cv2.imread(abs_image_path)
    if image is None:
        print(f"ERROR: Could not read image at {abs_image_path}")
        return

    height, width = image.shape[:2]

    annotations = parse_yolo_label(abs_label_path)
    for ann in annotations:
        cls_id = ann['class_id']
        x_center = ann['x_center'] * width
        y_center = ann['y_center'] * height
        w = ann['width'] * width
        h = ann['height'] * height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        color = (0, 255, 0)  # Green box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls_id]
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if save_path:
        # Ensure target directory exists (if provided)
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(save_path, image)
        abs_save = os.path.abspath(save_path)
        print(f"Saved image with bboxes to {save_path} (absolute: {abs_save})")
    else:
        # When running in headless environments, avoid opening windows
        try:
            cv2.imshow("Image with BBoxes", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            print("Unable to display image window (likely headless environment).")


def analyze_dataset(data_dir, output_dir):
    """
    Analyze the traffic detection dataset.

    Args:
        data_dir (str): Root directory of the dataset (containing train, valid, test folders)
        output_dir (str): Directory to save analysis outputs

    Returns:
        dict: Summary statistics and dataframes for further use
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured: {output_dir} (absolute: {os.path.abspath(output_dir)})")

    # Load class names
    yaml_path = os.path.join(data_dir, "data.yaml")
    print(f"Loading classes from: {yaml_path}")
    class_names = load_classes_from_yaml(yaml_path)
    print(f"Loaded classes: {class_names}")

    results = {}

    # Analyze train and valid splits
    splits = ["train", "valid"]
    for split in splits:
        print(f"\nAnalyzing {split} split...")
        split_dir = os.path.join(data_dir, split)
        files = get_image_label_files(split_dir)
        images = files.get("images", [])
        labels = files.get("labels", [])

        print(f"Number of images: {len(images)}")
        print(f"Number of labels: {len(labels)}")

        # Summarize class distribution
        class_dist = summarize_class_distribution(labels, class_names)
        print(class_dist)

        # Plot and save class distribution via existing utility (may save its own file)
        try:
            plot_class_distribution(class_dist, output_dir)
        except Exception as e:
            print(f"plot_class_distribution failed: {e}")

        # Additional plot: class distribution bar chart with plotly (save as HTML to avoid Chrome dependency)
        fig = None
        plot_path = None
        try:
            # Ensure class names are a column for plotly
            df = class_dist.reset_index().rename(columns={"index": "class"})
            # Arrow-safe types for downstream usage
            df["class"] = df["class"].astype(str)
            fig = px.bar(df, x="class", y="count", title=f'Class Distribution - {split}')
            html_path = os.path.join(output_dir, f"class_distribution_{split}.html")
            fig.write_html(html_path)
            print(f"Saved interactive Plotly HTML to {html_path}")
        except Exception as e:
            print(f"Failed to create plotly class distribution for {split}: {e}")
            fig = None
            plot_path = None

        results[split] = {
            "num_images": len(images),
            "num_labels": len(labels),
            "class_distribution": class_dist,
            "plot_path": plot_path if 'plot_path' in locals() else None,
            "plotly_fig": fig
        }

    # Highlight rare classes examples
    print("\nHighlighting rare class examples...")
    train_labels = get_image_label_files(os.path.join(data_dir, "train")).get("labels", [])
    class_dist_train = summarize_class_distribution(train_labels, class_names)
    rare_classes = class_dist_train[class_dist_train["count"] < 100].index.tolist()
    print(f"Rare classes (less than 100 instances): {rare_classes}")

    rare_examples = {}
    # helper to map label file to image path by trying common extensions
    def label_to_image_path(lbl_path):
        base = lbl_path.rsplit(os.sep + "labels" + os.sep, 1)
        if len(base) == 2:
            prefix = base[0] + os.sep + "images" + os.sep + base[1]
        else:
            prefix = lbl_path.replace("labels", "images")
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = prefix.replace(".txt", ext)
            if os.path.exists(candidate):
                return candidate
        # fallback: try replacing .txt with .jpg even if not exists (draw_bboxes will report)
        return prefix.replace(".txt", ".jpg")

    for rare_cls in rare_classes:
        cls_id = class_names.index(rare_cls)
        count = 0
        rare_examples[rare_cls] = []
        for lbl_file in train_labels:
            try:
                annots = parse_yolo_label(lbl_file)
            except Exception:
                annots = []
            if any(a.get("class_id") == cls_id for a in annots):
                img_file = label_to_image_path(lbl_file)
                save_path = os.path.join(output_dir, f"example_{rare_cls}_{count}.jpg")
                draw_bboxes(img_file, lbl_file, class_names, save_path)
                if os.path.exists(save_path):
                    rare_examples[rare_cls].append(save_path)
                count += 1
                if count >= 3:
                    break

    # Return results for programmatic use; interactive UI provided by Streamlit dashboard
    return results


if __name__ == "__main__":
    DATA_DIR = "/app/data/traffic-detection-project"
    OUTPUT_DIR = "/app/data/plots"
    analyze_dataset(DATA_DIR, OUTPUT_DIR)

