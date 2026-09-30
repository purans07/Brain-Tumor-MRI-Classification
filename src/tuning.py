"""Lightweight hyperparameter search for the supplied fixed validation split."""

from __future__ import annotations

import pandas as pd


def run_learning_rate_search(model_builder, train_ds, valid_ds, learning_rates=(1e-3, 3e-4, 1e-4), epochs=3, output_path=None):
    """Compare learning rates on the existing validation split.

    The source package already supplies train/valid/test folders. This helper
    does not reshuffle them, which avoids silently introducing leakage. If a
    project needs k-fold estimates, it should first obtain patient-level IDs
    and split by patient rather than by image.
    """

    rows = []
    for learning_rate in learning_rates:
        model = model_builder(learning_rate=learning_rate)
        history = model.fit(train_ds, validation_data=valid_ds, epochs=epochs, verbose=0)
        rows.append({"learning_rate": learning_rate, "best_val_accuracy": max(history.history.get("val_accuracy", [0])), "best_val_loss": min(history.history.get("val_loss", [float("inf")]))})
    result = pd.DataFrame(rows).sort_values("best_val_loss")
    if output_path is not None:
        result.to_csv(output_path, index=False)
    return result

