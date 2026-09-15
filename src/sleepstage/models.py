"""Resampling strategies, model pipelines, baselines and hyperparameter grids."""

from __future__ import annotations

import logging
from collections import Counter

from imblearn import FunctionSampler
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def resample(X, y, strategy: str, random_state: int = config.RANDOM_STATE, k_neighbors: int = config.SMOTE_K_NEIGHBORS):
    """Oversample minority classes of a TRAINING set up to the majority class size.

    Used inside imblearn pipelines, so it only runs during `fit` and never touches
    validation or test rows. `k_neighbors` shrinks automatically for tiny classes;
    classes with a single sample are left unchanged.
    """
    if strategy == "none":
        return X, y
    if strategy not in config.RESAMPLING_STRATEGIES:
        raise ValueError(f"Unknown resampling strategy: {strategy!r}")

    counts = Counter(y)
    majority = max(counts.values())
    if strategy == "selective_smote":
        min_count = config.SELECTIVE_SMOTE_MIN_SAMPLES + 1
    else:
        min_count = 2
    targets = {cls: majority for cls, count in counts.items() if min_count <= count < majority}
    if not targets:
        return X, y

    k = min(k_neighbors, min(counts[cls] for cls in targets) - 1)
    if strategy in ("smote", "selective_smote"):
        sampler = SMOTE(sampling_strategy=targets, k_neighbors=k, random_state=random_state)
    elif strategy == "borderline_smote":
        sampler = BorderlineSMOTE(sampling_strategy=targets, k_neighbors=k, random_state=random_state)
    else:
        sampler = ADASYN(sampling_strategy=targets, n_neighbors=k, random_state=random_state)

    try:
        return sampler.fit_resample(X, y)
    except (ValueError, RuntimeError) as exc:
        # ADASYN/Borderline-SMOTE refuse some class layouts (e.g. no borderline samples).
        logger.warning("%s could not resample this training set, using it unchanged: %s", strategy, exc)
        return X, y


def make_sampler(strategy: str, random_state: int = config.RANDOM_STATE) -> FunctionSampler:
    return FunctionSampler(func=resample, kw_args={"strategy": strategy, "random_state": random_state})
