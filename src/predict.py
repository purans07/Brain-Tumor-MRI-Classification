"""Single-image prediction helpers shared by CLI and Streamlit."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, IMAGE_SIZE


def load_model(model_path):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image):
    return np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)[None, ...]


def predict_image(model, image: Image.Image, class_names=CLASS_NAMES):
    probabilities = np.asarray(model.predict(preprocess_image(image), verbose=0))[0]
    index = int(np.argmax(probabilities))
    return {"class": class_names[index], "confidence": float(probabilities[index]), "probabilities": dict(zip(class_names, probabilities.astype(float)))}

