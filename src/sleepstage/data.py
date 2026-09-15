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
