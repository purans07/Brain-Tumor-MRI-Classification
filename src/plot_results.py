"""Plots generated after model training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_model_comparison(metrics: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    metric_names = [name for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1") if name in metrics.columns]
    ax = metrics.set_index("model")[metric_names].plot(kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0)
    ax.set_ylabel("score")
    ax.set_title("Model metric comparison")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def plot_training_histories(history_paths: dict[str, Path], output_dir: Path):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, path in history_paths.items():
            history = pd.read_csv(path)
            if metric in history:
                ax.plot(history[metric], label=f"{name} train")
            if f"val_{metric}" in history:
                ax.plot(history[f"val_{metric}"], linestyle="--", label=f"{name} valid")
        ax.set_title(f"Training and validation {metric}")
        ax.set_xlabel("epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"training_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_confusion_matrix(matrix, class_names, output_path: Path, normalized=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = np.asarray(matrix, dtype=float)
    if normalized:
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    annot = np.round(matrix, 2) if normalized else matrix.astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=annot, fmt="g", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Normalized confusion matrix" if normalized else "Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

