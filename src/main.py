import os
from dataset_utils import get_image_label_files, summarize_class_distribution, load_classes_from_yaml
from visualization import plot_class_distribution

# Path to dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/traffic-detection-project")
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")

def main():
    print("Loading dataset files...")
    files = get_image_label_files(DATA_DIR)
    print(f"Found {len(files['images'])} images and {len(files['labels'])} labels.")

    # Load classes dynamically from data.yaml
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")
    class_names = load_classes_from_yaml(DATA_YAML)
    print("Classes:", class_names)

    # Summarize class distribution
    print("Summarizing class distribution...")
    df = summarize_class_distribution(files['labels'], class_names)
    print(df)

    # Plot class distribution and save to file
    plot_dir = os.path.join(DATA_DIR, "plots")
    plot_class_distribution(df, save_dir=plot_dir)
    print(f"Class distribution plot saved in: {plot_dir}")

if __name__ == "__main__":
    main()
