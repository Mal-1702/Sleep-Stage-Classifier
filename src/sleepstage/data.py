"""Data loading, validation, label handling, wake trimming and train/test splits."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from . import config

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when a dataset does not match the expected schema."""


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def validate_features(df: pd.DataFrame, require_label: bool = True) -> None:
    """Check that `df` has every feature column, numeric and finite, plus known labels."""
    missing = [col for col in config.FEATURE_COLUMNS if col not in df.columns]
    if require_label and config.LABEL_COLUMN not in df.columns:
        missing.append(config.LABEL_COLUMN)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    non_numeric = [col for col in config.FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise DataValidationError(f"Feature columns must be numeric: {non_numeric}")
    if not np.isfinite(df[config.FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise DataValidationError("Feature columns contain NaN or infinite values")

    if require_label:
        unknown = sorted(set(df[config.LABEL_COLUMN].unique()) - set(config.RAW_LABEL_MAP))
        if unknown:
            raise DataValidationError(f"Unknown labels (not in RAW_LABEL_MAP): {unknown}")


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the feature CSV. Adds an `epoch` column with the original row order."""
    path = Path(path) if path is not None else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put dream_features.csv in data/ or set SLEEPSTAGE_DATA."
        )
    df = pd.read_csv(path)
    validate_features(df)
    df = df.reset_index(drop=True)
    df["epoch"] = np.arange(len(df))
    logger.info("Loaded %d epochs with %d features from %s", len(df), len(config.FEATURE_COLUMNS), path)
    return df


# ---------------------------------------------------------------------------
# Label mapping and N3 merging
# ---------------------------------------------------------------------------
def to_stage_codes(raw_labels: pd.Series) -> pd.Series:
    """Map raw CSV labels ('Sleep stage W', ...) to stage codes ('W', ...)."""
    stages = raw_labels.map(config.RAW_LABEL_MAP)
    if stages.isna().any():
        unknown = sorted(raw_labels[stages.isna()].unique())
        raise DataValidationError(f"Unknown labels (not in RAW_LABEL_MAP): {unknown}")
    return stages


def apply_class_setup(stages: pd.Series, class_setup: str) -> pd.Series:
    """Return stage codes for a class setup: '6class' keeps S3/S4, '5class' merges them into N3."""
    if class_setup == "6class":
        return stages.copy()
    if class_setup == "5class":
        return stages.replace(config.N3_MERGE_MAP)
    raise ValueError(f"Unknown class setup: {class_setup!r}")


def labels_for_setup(class_setup: str) -> list[str]:
    if class_setup not in config.CLASS_SETUPS:
        raise ValueError(f"Unknown class setup: {class_setup!r}")
    return list(config.CLASS_SETUPS[class_setup])


def six_to_five(labels) -> np.ndarray:
    """Map 6-class labels to 5-class (S3/S4 -> N3), so 6-class models can be scored like 5-class ones."""
    return np.array([config.N3_MERGE_MAP.get(label, label) for label in labels], dtype=object)


# ---------------------------------------------------------------------------
# Wake trimming
# ---------------------------------------------------------------------------
def trim_wake(
    df: pd.DataFrame,
    stage_column: str = "stage",
    margin: int = config.WAKE_TRIM_MARGIN_EPOCHS,
) -> pd.DataFrame:
    """Drop the long wake periods before sleep onset and after the final awakening.

    Keeps `margin` epochs of wake on each side of the sleep period. Assumes rows are
    in temporal order and come from one recording.
    """
    is_sleep = (df[stage_column] != config.WAKE_LABEL).to_numpy()
    if not is_sleep.any():
        logger.warning("No sleep epochs found; wake trimming skipped")
        return df.copy()
    sleep_positions = np.flatnonzero(is_sleep)
    start = max(0, sleep_positions[0] - margin)
    stop = min(len(df), sleep_positions[-1] + margin + 1)
    return df.iloc[start:stop].copy()
