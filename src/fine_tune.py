"""Fine-tune the top layers of the trained EfficientNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import BATCH_SIZE, DATA_ROOT, MODEL_ROOT, OUTPUT_ROOT
from .data_loader import build_tf_datasets, compute_class_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=MODEL_ROOT / "efficientnet_b0.best.keras")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--layers", type=int, default=40, help="Number of EfficientNet layers to unfreeze.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable training weights for an ablation run.")
    args = parser.parse_args()
    import tensorflow as tf

    train_ds, valid_ds, _ = build_tf_datasets(DATA_ROOT, BATCH_SIZE)
    model = tf.keras.models.load_model(args.checkpoint)
    backbone = next(layer for layer in model.layers if layer.name.startswith("efficientnet"))
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-args.layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    output = MODEL_ROOT / "efficientnet_b0_finetuned.best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(OUTPUT_ROOT / "model_results" / "efficientnet_b0_finetune_history.csv"),
    ]
    weights = None if args.no_class_weights else compute_class_weights(DATA_ROOT)
    model.fit(train_ds, validation_data=valid_ds, epochs=args.epochs, callbacks=callbacks, class_weight=weights)
    model.save(MODEL_ROOT / "efficientnet_b0_finetuned.h5")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
