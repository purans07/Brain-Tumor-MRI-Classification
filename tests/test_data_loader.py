import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data_loader import compute_class_weights


class DataLoaderTests(unittest.TestCase):
    def test_class_weights_use_only_training_class_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset" / "train"
            counts = {"glioma": 2, "meningioma": 1, "no_tumor": 1, "pituitary": 2}
            for class_name, count in counts.items():
                class_dir = root / class_name
                class_dir.mkdir(parents=True)
                for index in range(count):
                    Image.new("RGB", (8, 8), color="black").save(class_dir / f"{index}.png")

            weights = compute_class_weights(root.parent)

        self.assertAlmostEqual(weights[0], 0.75)
        self.assertAlmostEqual(weights[1], 1.5)
        self.assertAlmostEqual(weights[2], 1.5)
        self.assertAlmostEqual(weights[3], 0.75)


if __name__ == "__main__":
    unittest.main()
