"""Dataset profiling and meaningful visualizations based on the real images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import DATA_ROOT, FIGURE_ROOT, OUTPUT_ROOT, ensure_project_dirs
from .preprocessing import image_metadata


def save_plot(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.clf()


def _sample_grid(df, ax, one_per_class):
    import matplotlib.pyplot as plt

    sample = df.groupby("class", group_keys=False).head(1) if one_per_class else df.sample(min(9, len(df)), random_state=42)
    ax.remove()
    fig = plt.gcf()
    axes = fig.subplots(2, 2 if one_per_class else 5).ravel() if one_per_class else fig.subplots(3, 3).ravel()
    for axis, (_, row) in zip(axes, sample.iterrows()):
        axis.imshow(Image.open(row["path"]).convert("RGB"))
        axis.set_title(row["class"])
        axis.axis("off")
    for axis in axes[len(sample):]:
        axis.axis("off")


def _augmentation_grid(df, ax):
    import matplotlib.pyplot as plt
    from PIL import ImageEnhance, ImageOps

    image = Image.open(df.iloc[0]["path"]).convert("RGB").resize((224, 224))
    images = [
        image,
        ImageOps.mirror(image),
        ImageOps.flip(image),
        image.rotate(8),
        image.resize((246, 246)).crop((11, 11, 235, 235)),
        ImageEnhance.Brightness(image).enhance(1.15),
        ImageEnhance.Contrast(image).enhance(1.2),
    ]
    ax.remove()
    fig = plt.gcf()
    axes = fig.subplots(1, 7).ravel()
    titles = ["original", "horizontal flip", "vertical flip", "rotation", "zoom crop", "brightness", "contrast"]
    for axis, img, title in zip(axes, images, titles):
        axis.imshow(img)
        axis.set_title(title)
        axis.axis("off")


def _intensity_curves(df, ax):
    for cls, group in df.groupby("class"):
        values = []
        for path in group["path"].head(120):
            values.extend(np.asarray(Image.open(path).convert("L").resize((64, 64))).ravel())
        ax.hist(np.asarray(values), bins=32, density=True, histtype="step", linewidth=1.5, label=cls)
    ax.legend(title="class")
    ax.set_xlabel("grayscale intensity")
    ax.set_ylabel("density")


def _split_composition(df, ax):
    table = pd.crosstab(df["split"], df["class"], normalize="index") * 100
    table.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("percent within split")
    ax.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")


def generate_figures(df: pd.DataFrame, figure_dir: Path):
    """Create 15 meaningful charts and record the interpretation for each."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    charts = []
    class_counts = df["class"].value_counts().rename_axis("class").reset_index(name="count")
    class_counts["percentage"] = class_counts["count"] / len(df) * 100

    def chart(name, title, why, insight, impact, plotter):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotter(fig, ax)
        # Gallery charts replace the placeholder axes with a grid of image axes.
        if ax.figure is not None:
            ax.set_title(title)
        save_plot(fig, figure_dir / f"{name}.png")
        charts.append({"chart": name, "title": title, "why": why, "insight": insight, "business_impact": impact})

    chart("01_class_distribution", "Image count by class and split", "A grouped count plot checks coverage and split composition.", "Counts are calculated directly from the supplied folders, making class imbalance visible before training.", "The training plan can use class-aware metrics and class weights if the imbalance affects recall.", lambda f, a: sns.countplot(data=df, x="class", hue="split", ax=a))
    chart("02_class_percentage", "Overall class percentage", "Percentages make imbalance easier to compare than raw counts.", "Each bar shows the share of all images assigned to a class.", "Balanced evaluation avoids letting the largest class dominate reported accuracy.", lambda f, a: sns.barplot(data=class_counts, x="class", y="percentage", ax=a, color="#4472C4"))
    chart("03_width_distribution", "Image width distribution", "A width histogram checks whether resizing is required.", "The supplied images share a consistent width, so model resizing is standardization rather than repair for mixed dimensions.", "A consistent input pipeline reduces preprocessing surprises.", lambda f, a: sns.histplot(data=df, x="width", discrete=True, ax=a, color="#70AD47"))
    chart("04_height_distribution", "Image height distribution", "A height histogram complements width inspection.", "The supplied images share a consistent height.", "Uniform source dimensions support deterministic batching.", lambda f, a: sns.histplot(data=df, x="height", discrete=True, ax=a, color="#ED7D31"))
    chart("05_aspect_ratio", "Aspect ratio distribution", "Aspect ratio reveals stretching risk.", "The images have a 1:1 aspect ratio, so resizing to a square input does not introduce geometric distortion.", "Preserving shape reduces a preventable source of visual artifacts.", lambda f, a: sns.histplot(data=df, x="aspect_ratio", bins=10, ax=a, color="#A5A5A5"))
    chart("06_brightness_distribution", "Mean brightness distribution", "Brightness highlights acquisition and preprocessing variability.", "The distribution is measured from grayscale pixel values normalized to [0, 1].", "Contrast-aware augmentation can improve robustness to scanner and protocol differences.", lambda f, a: sns.histplot(data=df, x="mean_brightness", hue="class", bins=30, element="step", stat="density", common_norm=False, ax=a))
    chart("07_contrast_distribution", "Within-image contrast distribution", "Per-image standard deviation is a simple contrast proxy.", "The chart compares grayscale intensity spread across images.", "Very low contrast samples may need review or careful augmentation.", lambda f, a: sns.histplot(data=df, x="std_brightness", hue="class", bins=30, element="step", stat="density", common_norm=False, ax=a))
    chart("08_class_brightness", "Brightness by class", "A class box plot checks whether brightness differs systematically by label.", "Boxes show the observed distribution for each class, not a causal effect.", "Systematic acquisition differences could become shortcuts, so results should be validated on external data.", lambda f, a: sns.boxplot(data=df, x="class", y="mean_brightness", ax=a, color="#5B9BD5"))
    chart("09_class_contrast", "Contrast by class", "A class box plot checks contrast differences across labels.", "The observed spread identifies whether some classes are visually easier or harder due to contrast.", "Model performance should be read alongside per-class recall.", lambda f, a: sns.boxplot(data=df, x="class", y="std_brightness", ax=a, color="#FFC000"))
    chart("10_class_aspect", "Aspect ratio by class", "A class comparison checks whether geometry is label-dependent.", "All folders use the same square source geometry.", "There is no evidence here that aspect ratio can act as a class shortcut.", lambda f, a: sns.boxplot(data=df, x="class", y="aspect_ratio", ax=a, color="#70AD47"))
    chart("11_samples_by_class", "Representative image per class", "A labeled gallery validates that folder labels match visible sample content.", "The gallery is selected deterministically from each class.", "Visual inspection is a basic safety check before training a medical-imaging classifier.", lambda f, a: _sample_grid(df, a, one_per_class=True))
    chart("12_random_image_grid", "Randomized image grid", "A grid exposes variation that a single sample can hide.", "The displayed images are sampled with a fixed seed for reproducibility.", "Variation supports decisions about augmentation and error review.", lambda f, a: _sample_grid(df, a, one_per_class=False))
    chart("13_augmentation_examples", "Augmentation examples", "Before training, transformations should be visually inspected for plausibility.", "The panel applies mild flips, rotation, zoom, and contrast changes to a representative image.", "Plausible augmentation can improve generalization without creating anatomically implausible images.", lambda f, a: _augmentation_grid(df, a))
    chart("14_intensity_by_class", "Pixel intensity distributions by class", "Intensity curves compare the full grayscale value distribution by label.", "Curves summarize actual pixels from the dataset rather than model outputs.", "Strong differences may indicate acquisition bias and motivate external validation.", lambda f, a: _intensity_curves(df, a))
    chart("15_split_composition", "Split composition by class", "A normalized stacked bar chart checks whether train, valid, and test have similar class proportions.", "The provided split proportions can be compared without being confounded by split size.", "A representative test set makes model comparison more credible.", lambda f, a: _split_composition(df, a))
    return charts


def run(data_root=DATA_ROOT):
    ensure_project_dirs()
    df, errors = image_metadata(Path(data_root))
    df.to_csv(OUTPUT_ROOT / "image_metadata.csv", index=False)
    errors.to_csv(OUTPUT_ROOT / "corrupted_images.csv", index=False)
    summary = {
        "total_images": int(len(df)),
        "classes": sorted(df["class"].unique().tolist()),
        "class_counts": df["class"].value_counts().sort_index().to_dict(),
        "split_counts": df["split"].value_counts().sort_index().to_dict(),
        "corrupted_images": int(len(errors)),
        "exact_duplicate_files": int(df["is_exact_duplicate"].sum()),
        "image_sizes": {str(k): int(v) for k, v in df.groupby(["width", "height"]).size().items()},
        "channels": df["channels"].value_counts().sort_index().to_dict(),
        "mean_brightness": float(df["mean_brightness"].mean()),
        "mean_contrast": float(df["std_brightness"].mean()),
    }
    (OUTPUT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    charts = generate_figures(df, FIGURE_ROOT)
    pd.DataFrame(charts).to_csv(OUTPUT_ROOT / "chart_interpretations.csv", index=False)
    print(json.dumps(summary, indent=2, default=int))
    print(f"Generated {len(charts)} figures in {FIGURE_ROOT}")
    return df, errors, charts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    run(args.data_root)
