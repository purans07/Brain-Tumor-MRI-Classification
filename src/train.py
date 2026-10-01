"""Train and checkpoint the custom CNN and transfer-learning models."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import BATCH_SIZE, DATA_ROOT, IMAGE_SIZE, MODEL_ROOT, NUM_CLASSES, OUTPUT_ROOT, SEED, ensure_project_dirs
from .data_loader import build_tf_datasets, compute_class_weights
from .models import build_custom_cnn, build_efficientnet


def callbacks(model_name: str):
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(MODEL_ROOT / f"{model_name}.best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(OUTPUT_ROOT / "model_results" / f"{model_name}_history.csv"),
    ]


def train_one(name, train_ds, valid_ds, epochs, class_weights=None):
    if name == "custom_cnn":
        model = build_custom_cnn((*IMAGE_SIZE, 3), NUM_CLASSES)
    elif name == "efficientnet_b0":
        # Train the new classification head first. Fine-tuning remains
        # available through build_efficientnet(..., fine_tune_layers=20).
        model = build_efficientnet((*IMAGE_SIZE, 3), NUM_CLASSES, fine_tune_layers=0)
    else:
        raise ValueError(f"Unknown model: {name}")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks(name),
        class_weight=class_weights,
    )
    model.save(MODEL_ROOT / f"{name}.h5")
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["custom_cnn", "efficientnet_b0"])
    parser.add_argument("--no-class-weights", action="store_true", help="Disable training weights for an ablation run.")
    args = parser.parse_args()
    ensure_project_dirs()
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    train_ds, valid_ds, _ = build_tf_datasets(args.data_root, BATCH_SIZE)
    weights = None if args.no_class_weights else compute_class_weights(args.data_root)
    if weights:
        print(f"Using training class weights: {weights}")
    for name in args.models:
        train_one(name, train_ds, valid_ds, args.epochs, weights)


if __name__ == "__main__":
    main()
