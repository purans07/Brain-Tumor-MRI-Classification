"""Dataset discovery, validation, and TensorFlow input pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import CLASS_NAMES, IMAGE_SIZE, SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_paths(root: Path) -> Iterable[Path]:
    """Yield supported image paths below a split directory."""

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("_"):
            yield path


def validate_image(path: Path) -> tuple[bool, str]:
    """Check whether an image can be decoded fully."""

    try:
        with Image.open(path) as image:
            image.load()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def build_tf_datasets(data_root: Path, batch_size: int = 32):
    """Load the provided train, valid, and test directories as tf.data datasets."""

    import tensorflow as tf

    kwargs = dict(
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        class_names=CLASS_NAMES,
        seed=SEED,
    )
    train = tf.keras.utils.image_dataset_from_directory(data_root / "train", shuffle=True, **kwargs)
    valid = tf.keras.utils.image_dataset_from_directory(data_root / "valid", shuffle=False, **kwargs)
    test = tf.keras.utils.image_dataset_from_directory(data_root / "test", shuffle=False, **kwargs)
    autotune = tf.data.AUTOTUNE
    return tuple(dataset.prefetch(autotune) for dataset in (train, valid, test))
