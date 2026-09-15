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


# ---------------------------------------------------------------------------
# Models, pipelines and hyperparameter grids
# ---------------------------------------------------------------------------
def is_baseline(model: str) -> bool:
    return model in config.BASELINE_MODELS


def build_classifier(model: str, random_state: int = config.RANDOM_STATE):
    if model == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")
    if model == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=random_state)
    if model in ("random_forest", "random_forest_pca"):
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS, class_weight="balanced", n_jobs=-1, random_state=random_state
        )
    if model == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=random_state)
    if model == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=2000)
    raise ValueError(f"Unknown model: {model!r}")


def build_pipeline(model: str, resampling: str = "none", random_state: int = config.RANDOM_STATE) -> ImbPipeline:
    """Resampling (train-only) -> optional scaling/PCA -> classifier, as one pipeline."""
    if is_baseline(model) and resampling != "none":
        raise ValueError("Baselines are evaluated without resampling")
    steps = []
    if resampling != "none":
        steps.append(("resample", make_sampler(resampling, random_state)))
    if model == "logistic_regression":
        steps.append(("scale", StandardScaler()))
    if model == "random_forest_pca":
        steps.append(("pca", PCA(n_components=config.PCA_COMPONENTS, random_state=random_state)))
    steps.append(("clf", build_classifier(model, random_state)))
    return ImbPipeline(steps)


PARAM_GRIDS = {
    "random_forest": {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 4]},
    "random_forest_pca": {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 4]},
    "hist_gradient_boosting": {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]},
    "logistic_regression": {"clf__C": [0.1, 1.0, 10.0]},
}


def param_grid_for(model: str) -> dict:
    """Small search grid for a model; empty for baselines."""
    return PARAM_GRIDS.get(model, {})
