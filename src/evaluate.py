"""Evaluation metrics, confusion matrices, and error analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def classification_metrics(y_true, y_pred, class_names):
    """Compute accuracy, macro metrics, and one-vs-rest class metrics."""

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for index, name in enumerate(class_names):
        tp = int(np.sum((y_true == index) & (y_pred == index)))
        fp = int(np.sum((y_true != index) & (y_pred == index)))
        fn = int(np.sum((y_true == index) & (y_pred != index)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": name, "precision": precision, "recall": recall, "f1_score": f1, "support": int(np.sum(y_true == index))})
    by_class = pd.DataFrame(rows)
    summary = {"accuracy": float(np.mean(y_true == y_pred)), "macro_precision": float(by_class.precision.mean()), "macro_recall": float(by_class.recall.mean()), "macro_f1": float(by_class.f1_score.mean())}
    return summary, by_class


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def evaluate_keras_model(model, dataset, class_names, output_dir: Path):
    """Evaluate a Keras model and save machine-readable metrics."""

    y_true, probabilities = [], []
    for images, labels in dataset:
        y_true.extend(labels.numpy().tolist())
        probabilities.append(model.predict(images, verbose=0))
    y_true = np.asarray(y_true)
    y_pred = np.argmax(np.concatenate(probabilities), axis=1)
    summary, by_class = classification_metrics(y_true, y_pred, class_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_dir / "summary_metrics.csv", index=False)
    by_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    np.savetxt(output_dir / "confusion_matrix.csv", confusion_matrix(y_true, y_pred, len(class_names)), fmt="%d", delimiter=",")
    return summary, by_class, y_true, y_pred

