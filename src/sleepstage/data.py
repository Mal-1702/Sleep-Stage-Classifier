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


# ---------------------------------------------------------------------------
# Train/test splits
# ---------------------------------------------------------------------------
@dataclass
class Split:
    """Positional train/test indices plus the CV splitter to use on the training rows."""

    split_type: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    cv: object
    groups: np.ndarray | None = None
    test_blocks: list[int] | None = None


def random_split(stratify_labels, random_state: int = config.RANDOM_STATE) -> Split:
    """Stratified random split of individual epochs (leaks between neighbouring epochs)."""
    positions = np.arange(len(stratify_labels))
    train_idx, test_idx = train_test_split(
        positions, test_size=config.TEST_SIZE, random_state=random_state, stratify=stratify_labels
    )
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=random_state)
    return Split("random", np.sort(train_idx), np.sort(test_idx), cv)


def make_blocks(n_samples: int, n_blocks: int = config.N_BLOCKS) -> np.ndarray:
    """Assign each time-ordered sample to one of `n_blocks` contiguous, near-equal blocks."""
    if not 2 <= n_blocks <= n_samples:
        raise ValueError(f"n_blocks must be between 2 and n_samples ({n_samples}), got {n_blocks}")
    block_ids = np.empty(n_samples, dtype=int)
    for block, positions in enumerate(np.array_split(np.arange(n_samples), n_blocks)):
        block_ids[positions] = block
    return block_ids


def choose_test_blocks(block_ids: np.ndarray, labels, n_test_blocks: int = config.N_TEST_BLOCKS) -> list[int]:
    """Pick the test blocks whose class mix best matches the whole recording.

    First maximises the number of classes present in BOTH train and test, then minimises
    the average gap between each class's test share and the overall test fraction, so no
    class (e.g. deep sleep, which clusters early in the night) ends up almost entirely in
    the test set. Deterministic: ties go to the lowest block numbers.
    """
    labels = np.asarray(labels)
    classes, totals = np.unique(labels, return_counts=True)
    n_blocks = int(block_ids.max()) + 1
    target_share = n_test_blocks / n_blocks
    best_score, best_combo = None, None
    for combo in combinations(range(n_blocks), n_test_blocks):
        test_labels = labels[np.isin(block_ids, combo)]
        test_counts = np.array([np.sum(test_labels == cls) for cls in classes])
        in_both = int(np.sum((test_counts > 0) & (test_counts < totals)))
        share_gap = float(np.mean(np.abs(test_counts / totals - target_share)))
        score = (in_both, -share_gap)
        if best_score is None or score > best_score:
            best_score, best_combo = score, combo
    return list(best_combo)


def blocked_split(
    stratify_labels,
    n_blocks: int = config.N_BLOCKS,
    n_test_blocks: int = config.N_TEST_BLOCKS,
    embargo: int = config.BLOCK_EMBARGO_EPOCHS,
) -> Split:
    """Hold out whole contiguous blocks of time as the test set.

    Training epochs within `embargo` epochs of a test block are dropped, and CV on the
    training rows uses StratifiedGroupKFold over block ids, so no block is split across folds.
    """
    n_samples = len(stratify_labels)
    block_ids = make_blocks(n_samples, n_blocks)
    test_blocks = choose_test_blocks(block_ids, stratify_labels, n_test_blocks)
    is_test = np.isin(block_ids, test_blocks)

    window = np.ones(2 * embargo + 1)
    near_test = np.convolve(is_test.astype(float), window, mode="same") > 0
    train_idx = np.flatnonzero(~near_test)
    test_idx = np.flatnonzero(is_test)

    groups = block_ids[train_idx]
    n_splits = min(config.CV_FOLDS, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    return Split("blocked", train_idx, test_idx, cv, groups=groups, test_blocks=test_blocks)


def make_split(split_type: str, stratify_labels, random_state: int = config.RANDOM_STATE) -> Split:
    if split_type == "random":
        return random_split(stratify_labels, random_state)
    if split_type == "blocked":
        return blocked_split(stratify_labels)
    raise ValueError(f"Unknown split type: {split_type!r}")
