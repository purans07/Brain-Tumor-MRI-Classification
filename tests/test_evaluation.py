import unittest

import numpy as np

from src.evaluate import classification_metrics, confusion_matrix


class EvaluationTests(unittest.TestCase):
    def test_classification_metrics_are_computed_per_class(self):
        summary, by_class = classification_metrics(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 1, 1, 2, 0]),
            ["glioma", "meningioma", "no_tumor"],
        )
        self.assertAlmostEqual(summary["accuracy"], 4 / 6)
        self.assertEqual(list(by_class["class"]), ["glioma", "meningioma", "no_tumor"])
        self.assertTrue((by_class["support"] == [2, 2, 2]).all())

    def test_confusion_matrix_rows_are_actual_classes(self):
        matrix = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), 3)
        np.testing.assert_array_equal(matrix, [[1, 1, 0], [0, 1, 0], [1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
