"""Central configuration for the brain tumor MRI project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Tumour dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
MODEL_ROOT = PROJECT_ROOT / "models"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"


def ensure_project_dirs() -> None:
    """Create output directories without touching the source dataset."""

    for path in (
        OUTPUT_ROOT,
        FIGURE_ROOT,
        MODEL_ROOT,
        OUTPUT_ROOT / "confusion_matrices",
        OUTPUT_ROOT / "model_results",
        OUTPUT_ROOT / "error_analysis",
        OUTPUT_ROOT / "gradcam",
    ):
        path.mkdir(parents=True, exist_ok=True)

