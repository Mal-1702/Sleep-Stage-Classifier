"""Project-wide constants: paths, seeds, label maps and experiment settings.

Every label name, threshold and seed lives here so the rest of the code never
hard-codes them.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("SLEEPSTAGE_DATA", PROJECT_ROOT / "data" / "dream_features.csv"))
ARTIFACTS_DIR = Path(os.environ.get("SLEEPSTAGE_ARTIFACTS", PROJECT_ROOT / "artifacts"))

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
LABEL_COLUMN = "label"
FEATURE_COLUMNS = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "std_dev",
    "skewness",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "delta_beta_ratio",
    "alpha_delta_ratio",
    "shannon_entropy",
]
EPOCH_SECONDS = 30

# ---------------------------------------------------------------------------
# Labels (the single source of truth for stage names)
# ---------------------------------------------------------------------------
# Raw labels in the CSV (Rechtschaffen & Kales scoring) -> short stage codes.
RAW_LABEL_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "S3",
    "Sleep stage 4": "S4",
    "Sleep stage R": "REM",
}
SIX_CLASS_LABELS = ["W", "N1", "N2", "S3", "S4", "REM"]
# AASM (2007) merges R&K stages 3 and 4 into a single deep-sleep stage, N3.
N3_MERGE_MAP = {"S3": "N3", "S4": "N3"}
FIVE_CLASS_LABELS = ["W", "N1", "N2", "N3", "REM"]
CLASS_SETUPS = {"5class": FIVE_CLASS_LABELS, "6class": SIX_CLASS_LABELS}

WAKE_LABEL = "W"
# Hypnogram y-axis order, top to bottom.
HYPNOGRAM_ORDER = ["W", "REM", "N1", "N2", "N3", "S3", "S4"]
STAGE_DISPLAY_NAMES = {
    "W": "Wake",
    "N1": "N1 (light)",
    "N2": "N2",
    "N3": "N3 (deep, S3+S4)",
    "S3": "Stage 3 (deep)",
    "S4": "Stage 4 (deep)",
    "REM": "REM",
}

# ---------------------------------------------------------------------------
# Preprocessing and splits
# ---------------------------------------------------------------------------
# Keep this many epochs (60 x 30 s = 30 min) of wake before sleep onset and after final awakening.
WAKE_TRIM_MARGIN_EPOCHS = 60

TEST_SIZE = 0.2
CV_FOLDS = 5
N_BLOCKS = 10
N_TEST_BLOCKS = 2
# Training epochs this close to a test block are dropped, so neighbours of test epochs are not trained on.
BLOCK_EMBARGO_EPOCHS = 5

# ---------------------------------------------------------------------------
# Models and resampling
# ---------------------------------------------------------------------------
SMOTE_K_NEIGHBORS = 5
# Selective SMOTE only oversamples classes with MORE than this many training samples.
SELECTIVE_SMOTE_MIN_SAMPLES = 15

RF_N_ESTIMATORS = 200
PCA_COMPONENTS = 6
PERMUTATION_REPEATS = 5

SPLIT_TYPES = ["random", "blocked"]
TRIM_OPTIONS = ["untrimmed", "trimmed"]
RESAMPLING_STRATEGIES = ["none", "smote", "borderline_smote", "adasyn", "selective_smote"]
BASELINE_MODELS = ["dummy_most_frequent", "dummy_stratified"]
TRAINABLE_MODELS = ["random_forest", "random_forest_pca", "hist_gradient_boosting", "logistic_regression"]
MODELS = BASELINE_MODELS + TRAINABLE_MODELS
DEFAULT_RESAMPLING = "smote"
PRIMARY_MODEL = "random_forest"

MODEL_DISPLAY_NAMES = {
    "dummy_most_frequent": "Baseline: always most frequent",
    "dummy_stratified": "Baseline: random by class frequency",
    "random_forest": "Random Forest",
    "random_forest_pca": "Random Forest + PCA(6)",
    "hist_gradient_boosting": "Hist Gradient Boosting",
    "logistic_regression": "Logistic Regression",
}
RESAMPLING_DISPLAY_NAMES = {
    "none": "None (class weights only)",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-SMOTE",
    "adasyn": "ADASYN",
    "selective_smote": "Selective SMOTE",
}

DISCLAIMER = "Rule-based illustration only — not trained, not validated, not medical advice."
