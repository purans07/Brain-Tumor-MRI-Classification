"""Evaluate one trained model on the supplied test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import BATCH_SIZE, CLASS_NAMES, DATA_ROOT, MODEL_ROOT, OUTPUT_ROOT
from .data_loader import build_tf_datasets
from .evaluate import evaluate_keras_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    import tensorflow as tf

    _, _, test = build_tf_datasets(args.data_root, BATCH_SIZE)
    model = tf.keras.models.load_model(args.model)
    name = args.model.stem.replace(".best", "")
    output_dir = OUTPUT_ROOT / "model_results" / name
    summary, by_class, _, _ = evaluate_keras_model(model, test, CLASS_NAMES, output_dir)
    print(summary)
    print(by_class.to_string(index=False))


if __name__ == "__main__":
    main()

