import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_class_distribution(df, save_dir):
    """
    Plot class frequency as bar chart and save to file.
    
    Args:
        df (pd.DataFrame): Class frequency table
        save_dir (str): Directory to save plot
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df.index, y="count", data=df)
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
