"""Conservative MRI image augmentation used during training."""


def build_augmentation():
    """Create the training-only Keras augmentation pipeline.

    The dataset loader returns pixel values in the 0-255 range, so brightness
    uses that same range. The transformations remain intentionally mild to
    avoid creating anatomically implausible MRI images.
    """

    import tensorflow as tf

    return tf.keras.Sequential(
        [
            # Vertical flips can invert the anatomy and hurt generalization.
            # Horizontal flipping is a safer approximation for this dataset.
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.03, 0.03),
            tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="mri_augmentation",
    )
