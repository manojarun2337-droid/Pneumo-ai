"""
Central configuration for PNEUMO.AI.

Every numeric value in this file was produced during the Colab
experimentation phase (see notebooks/pneumonia_detection.ipynb) and is
FINAL. Nothing here should be recalculated, re-tuned, or "improved" by
the application — the app only loads and displays these results.
"""

from pathlib import Path

# --------------------------------------------------------------------
# Paths (all relative to the project root so the app is portable
# between machines / works the same on Windows and Linux)
# --------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "densenet121_stage2_best.keras"
OVERLAP_CSV_PATH = ROOT_DIR / "data" / "densenet_gradcam_lung_overlap_full.csv"
REPRESENTATIVE_CASES_IMAGE = ROOT_DIR / "visuals" / "representative_gradcam_cases.png"
SAMPLE_IMAGES_DIR = ROOT_DIR / "sample_images"

# --------------------------------------------------------------------
# Model architecture / preprocessing (must match the notebook exactly)
# --------------------------------------------------------------------
IMAGE_SIZE = (224, 224)          # (height, width) expected by DenseNet121
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}  # from train_generator.class_indices
LAST_CONV_LAYER_NAME = "conv5_block16_2_conv"

# Locked classification threshold selected on the validation set.
# probability >= threshold -> PNEUMONIA, else NORMAL. Do not use 0.5.
CLASSIFICATION_THRESHOLD = 0.10

# --------------------------------------------------------------------
# Final held-out test-set performance (DenseNet121, threshold = 0.10)
# --------------------------------------------------------------------
DENSENET_TEST_METRICS = {
    "accuracy": 0.8109,
    "precision": 0.7756,
    "sensitivity": 0.9877,
    "specificity": 0.5043,
    "f1_score": 0.8689,
    "roc_auc": 0.9629,
}

DENSENET_CONFUSION_MATRIX = {
    "tn": 118,
    "fp": 116,
    "fn": 5,
    "tp": 401,
}

# CNN baseline, for comparison only (from the same notebook)
CNN_BASELINE_TEST_METRICS = {
    "accuracy": 0.7781,
    "precision": 0.7500,
    "sensitivity": 0.9754,
    "specificity": 0.4359,
    "f1_score": 0.8480,
    "roc_auc": 0.9313,
}

# --------------------------------------------------------------------
# Dataset composition (ImageDataGenerator.flow_from_directory output)
# --------------------------------------------------------------------
DATASET_SPLITS = {
    "train": {"total": 4457, "NORMAL": 1140, "PNEUMONIA": 3317},
    "validation": {"total": 786},
    "test": {"total": 640, "NORMAL": 234, "PNEUMONIA": 406},
}
VALIDATION_SPLIT_NOTE = (
    "The original dataset's validation folder was too small to be "
    "statistically useful, so a 15% stratified validation split was "
    "carved out of the training data instead."
)

# --------------------------------------------------------------------
# Explainability / anatomical-consistency findings
# (from the 640-image held-out test-set analysis)
# --------------------------------------------------------------------
ATTENTION_MEDIANS = {
    "NORMAL": 56.575,
    "PNEUMONIA": 43.078,
}

SPEARMAN_CORRELATIONS = {
    "overall": 0.764,
    "pneumonia": 0.863,
    "normal": 0.472,
}

# --------------------------------------------------------------------
# Text used across the UI
# --------------------------------------------------------------------
RESEARCH_DISCLAIMER = (
    "Research prototype only. This system is intended for educational "
    "and research use and is not a medical diagnostic device."
)

APP_TITLE = "PNEUMO.AI"
APP_TAGLINE = "Explainable Deep Learning for Pneumonia Detection from Chest X-Rays"
