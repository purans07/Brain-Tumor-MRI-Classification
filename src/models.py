"""Custom CNN and transfer-learning model builders."""


def _compile(model, learning_rate: float = 1e-3):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4, learning_rate=1e-3):
    """Build a compact CNN from scratch with normalization and dropout."""

    import tensorflow as tf
    from .augmentation import build_augmentation

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = build_augmentation()(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return _compile(tf.keras.Model(inputs, outputs, name="custom_cnn"), learning_rate)


def build_efficientnet(input_shape=(224, 224, 3), num_classes=4, weights="imagenet", fine_tune_layers=20, learning_rate=1e-4):
    """Build EfficientNetB0 with a replaceable classification head."""

    import tensorflow as tf
    from .augmentation import build_augmentation

    try:
        backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    except Exception as exc:
        if weights is not None:
            print(f"Warning: pretrained weights unavailable ({exc}); using random initialization.")
            backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)
        else:
            raise
    backbone.trainable = False
    if fine_tune_layers:
        for layer in backbone.layers[-fine_tune_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = build_augmentation()(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return _compile(tf.keras.Model(inputs, outputs, name="efficientnet_b0_transfer"), learning_rate)
