"""Grad-CAM explainability for the deployed convolutional model."""

from __future__ import annotations


def make_gradcam_heatmap(image_batch, model, last_conv_layer_name: str, class_index=None):
    """Return a normalized Grad-CAM heatmap for one image batch."""

    import tensorflow as tf

    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
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

