"""Create comparison and confusion-matrix figures from saved model results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, OUTPUT_ROOT, FIGURE_ROOT
from .plot_results import plot_confusion_matrix, plot_model_comparison, plot_training_histories


def main():
    model_root = OUTPUT_ROOT / "model_results"
    rows = []
    histories = {}
    for name in ("custom_cnn", "efficientnet_b0", "efficientnet_b0_finetuned"):
        result_root = model_root / name
        summary_path = result_root / "summary_metrics.csv"
        if summary_path.exists():
            row = pd.read_csv(summary_path).iloc[0].to_dict()
            row["model"] = name
            rows.append(row)
        history_path = model_root / f"{name}_history.csv"
        if name == "efficientnet_b0_finetuned":
            history_path = model_root / "efficientnet_b0_finetune_history.csv"
        if history_path.exists():
            histories[name] = history_path
        matrix_path = result_root / "confusion_matrix.csv"
        if matrix_path.exists():
            matrix = np.loadtxt(matrix_path, delimiter=",", dtype=int)
            plot_confusion_matrix(matrix, CLASS_NAMES, FIGURE_ROOT / f"{name}_confusion_matrix.png")
            plot_confusion_matrix(matrix, CLASS_NAMES, FIGURE_ROOT / f"{name}_confusion_matrix_normalized.png", normalized=True)
    if rows:
        comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
        comparison.to_csv(model_root / "model_comparison.csv", index=False)
        plot_model_comparison(comparison, FIGURE_ROOT / "model_comparison_metrics.png")
    if histories:
        plot_training_histories(histories, FIGURE_ROOT)
    print(f"Wrote reports to {model_root} and {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
