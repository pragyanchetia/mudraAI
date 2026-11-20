"""Shared configuration for the MudraAI pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

# Sattriya dataset (flat folder: P<person>_<Mudra>_<n|d>_<index>.jpg)
DATASET_PATH = (
    PROJECT_ROOT
    / "Sattriya_Size-Norm_Gray_Image_Dataset-20251013T134943Z-1-001"
    / "Sattriya_Size-Norm_Gray_Image_Dataset"
)

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_dataset.csv"
TEST_CSV = DATA_DIR / "test_dataset.csv"
FAILED_TRAIN_TXT = DATA_DIR / "failed_train_images.txt"
FAILED_TEST_TXT = DATA_DIR / "failed_test_images.txt"

CLASSIFIER_PATH = MODELS_DIR / "mudra-classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
GATEKEEPER_PATH = MODELS_DIR / "mudra_gatekeeper.joblib"
CLASSIFIER_ONNX = MODELS_DIR / "mudra-classifier.onnx"
GATEKEEPER_ONNX = MODELS_DIR / "gatekeeper.onnx"

# Data preparation
TEST_SIZE = 0.2
RANDOM_STATE = 42
AUGMENTATION_COUNT = 5      # noisy copies per training image
NOISE_LEVEL = 0.005         # std-dev of gaussian landmark noise
MP_DETECTION_CONFIDENCE = 0.2

# Typos in the dataset filenames -> canonical label
LABEL_FIXES = {"Sarpashiran": "Sarpashira"}

# Gatekeeper (One-Class SVM)
GATEKEEPER_NU = 0.05
