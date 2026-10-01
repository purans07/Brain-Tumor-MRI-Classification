"""Grad-CAM explainability for the deployed convolutional model."""

from __future__ import annotations

from PIL import Image


def find_last_spatial_layer(model):
    """Find the deepest 4-D feature layer for Grad-CAM.

    EfficientNet is nested inside the application model, so this searches
    nested Keras models as well as top-level layers.
    """

    import tensorflow as tf

    candidates = []

    def visit(layer):
        if isinstance(layer, tf.keras.Model):
            for nested in layer.layers:
                visit(nested)
        try:
            shape = layer.output_shape
            if isinstance(shape, tuple) and len(shape) == 4 and shape[1] and shape[2]:
                candidates.append(layer)
        except (AttributeError, TypeError):
            return

    for layer in model.layers:
        visit(layer)
    if not candidates:
        raise ValueError("No spatial feature layer was found for Grad-CAM.")
    return candidates[-1]


def make_gradcam_heatmap(image_batch, model, last_conv_layer_name=None, class_index=None):
    """Return a normalized Grad-CAM heatmap for one image batch."""

    import tensorflow as tf

    if last_conv_layer_name is None:
        target_layer = find_last_spatial_layer(model)
    elif hasattr(last_conv_layer_name, "output"):
        target_layer = last_conv_layer_name
    else:
        target_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        score = predictions[:, class_index]
    gradients = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(tf.multiply(weights[:, None, None, :], conv_outputs), axis=-1)
    cam = tf.maximum(cam, 0)[0]
    cam /= tf.reduce_max(cam) + tf.keras.backend.epsilon()
    return cam.numpy(), int(class_index), predictions[0].numpy()


def overlay_gradcam(image: Image.Image, heatmap, alpha: float = 0.42) -> Image.Image:
    """Blend a blue-to-red attention heatmap over the original image."""

    import numpy as np

    values = np.asarray(heatmap, dtype=np.float32)
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(values * 2.0, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(values * 2.0 - 1.0) * 2.0, 0.0, 1.0)
    blue = np.clip(1.0 - values * 2.0, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    colored = Image.fromarray(np.uint8(rgb * 255), mode="RGB").resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), colored, alpha)
