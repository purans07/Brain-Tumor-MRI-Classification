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
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="mri_augmentation",
    )
