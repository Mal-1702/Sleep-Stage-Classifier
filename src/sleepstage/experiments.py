"""Run the experiment grid and write artifacts for the dashboard.

Experiment = split type x wake trimming x class setup x resampling strategy x model.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import apply_class_setup, labels_for_setup, load_dataset, make_split, to_stage_codes, trim_wake
from .evaluate import (
    compute_metrics,
    fit_and_cross_validate,
    five_class_equivalent,
    impurity_importance,
    permutation_importance_scores,
)
from .models import build_pipeline, is_baseline, param_grid_for, resample

logger = logging.getLogger(__name__)

METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.csv"
CV_FOLDS_FILE = "cv_folds.csv"
IMPORTANCE_FILE = "feature_importance.json"
MODELS_DIR = "models"


# ---------------------------------------------------------------------------
# Experiment specs and data contexts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentSpec:
    split_type: str
    trimming: str
    class_setup: str
    resampling: str
    model: str

    @property
    def key(self) -> str:
        return "|".join((self.split_type, self.trimming, self.class_setup, self.resampling, self.model))

    @property
    def context_key(self) -> str:
        return "|".join((self.split_type, self.trimming, self.class_setup))


def model_filename(key: str) -> str:
    return key.replace("|", "__") + ".joblib"


def build_experiment_specs(quick: bool = False) -> list[ExperimentSpec]:
    """All experiments, ordered so each context's default-resampling runs come first.

    Full: every model with the default resampling (baselines without), plus the primary
    model with every other resampling strategy. Quick: baselines + primary model with
    'none' and the default strategy only.
    """
    models = config.MODELS if not quick else config.BASELINE_MODELS + [config.PRIMARY_MODEL]
    extra_strategies = [s for s in config.RESAMPLING_STRATEGIES if s != config.DEFAULT_RESAMPLING]
    if quick:
        extra_strategies = ["none"]

    specs = []
    for split_type in config.SPLIT_TYPES:
        for trimming in config.TRIM_OPTIONS:
            for class_setup in config.CLASS_SETUPS:
                for model in models:
                    resampling = "none" if is_baseline(model) else config.DEFAULT_RESAMPLING
                    specs.append(ExperimentSpec(split_type, trimming, class_setup, resampling, model))
                for strategy in extra_strategies:
                    specs.append(ExperimentSpec(split_type, trimming, class_setup, strategy, config.PRIMARY_MODEL))
    return specs


@dataclass
class DataContext:
    frame: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    split: object


def prepare_context(df: pd.DataFrame, split_type: str, trimming: str, class_setup: str) -> DataContext:
    """Apply label setup, optional wake trimming and the split for one context.

    Splits are chosen on the 6-class labels, so 5- and 6-class experiments share the
    exact same test epochs and can be compared fairly.
    """
    six_class = to_stage_codes(df[config.LABEL_COLUMN])
    frame = df.assign(stage_6class=six_class, stage=apply_class_setup(six_class, class_setup))
    if trimming == "trimmed":
        frame = trim_wake(frame, stage_column="stage_6class")
    elif trimming != "untrimmed":
        raise ValueError(f"Unknown trimming option: {trimming!r}")
    frame = frame.reset_index(drop=True)
    split = make_split(split_type, frame["stage_6class"].to_numpy())
    return DataContext(frame, frame[config.FEATURE_COLUMNS].to_numpy(), frame["stage"].to_numpy(dtype=object), split)
