"""Generate one Grad-CAM overlay for a saved custom CNN checkpoint."""

from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, MODEL_ROOT, OUTPUT_ROOT, TEST_DIR, IMAGE_SIZE
from .explainability import make_gradcam_heatmap
from .predict import preprocess_image


def main():
    import tensorflow as tf
    import matplotlib.pyplot as plt

    model = tf.keras.models.load_model(MODEL_ROOT / "custom_cnn.best.keras")
    image_path = sorted(TEST_DIR.glob("*/[!_]*.jpg"))[0]
    image = Image.open(image_path).convert("RGB")
    batch = tf.convert_to_tensor(preprocess_image(image))
    heatmap, class_index, probabilities = make_gradcam_heatmap(batch, model, "conv2d_3")
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title("MRI image")
    axes[1].imshow(image)
    axes[1].imshow(heatmap_image, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM: {CLASS_NAMES[class_index]}")
    for axis in axes:
        axis.axis("off")
    output_path = OUTPUT_ROOT / "gradcam" / "custom_cnn_gradcam_example.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print({"image": str(image_path), "prediction": CLASS_NAMES[class_index], "confidence": float(probabilities[class_index]), "output": str(output_path)})


if __name__ == "__main__":
    main()

