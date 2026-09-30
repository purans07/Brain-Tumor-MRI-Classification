"""Image preprocessing and metadata utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data_loader import image_paths, validate_image


def image_metadata(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return image-level metadata and a report of files that failed decoding."""

    records = []
    errors = []
    hashes: dict[str, list[str]] = {}
    for path in image_paths(data_root):
        split, class_name = path.parts[-3], path.parts[-2]
        valid, error = validate_image(path)
        if not valid:
            errors.append({"path": str(path), "split": split, "class": class_name, "error": error})
            continue
        try:
            with Image.open(path) as image:
                image.load()
                rgb = image.convert("RGB")
                gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                hashes.setdefault(digest, []).append(str(path))
                records.append(
                    {
                        "filename": path.name,
                        "path": str(path),
                        "split": split,
                        "class": class_name,
                        "width": image.width,
                        "height": image.height,
                        "aspect_ratio": image.width / image.height,
                        "channels": len(image.getbands()),
                        "mode": image.mode,
                        "mean_brightness": float(gray.mean()),
                        "std_brightness": float(gray.std()),
                        "file_extension": path.suffix.lower(),
                        "md5": digest,
                    }
                )
        except Exception as exc:
            errors.append({"path": str(path), "split": split, "class": class_name, "error": str(exc)})

    metadata = pd.DataFrame(records)
    if not metadata.empty:
        duplicate_digests = {digest for digest, paths in hashes.items() if len(paths) > 1}
        metadata["is_exact_duplicate"] = metadata["md5"].isin(duplicate_digests)
    else:
        metadata["is_exact_duplicate"] = pd.Series(dtype=bool)
    return metadata, pd.DataFrame(errors)


def normalize_images(images):
    """Normalize uint8 image tensors to float32 in [0, 1]."""

    import tensorflow as tf

    return tf.cast(images, tf.float32) / 255.0

