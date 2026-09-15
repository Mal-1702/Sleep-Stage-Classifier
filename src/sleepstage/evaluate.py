"""Metrics, cross-validation with tuning, and feature importance."""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, cross_validate

from . import config
from .data import six_to_five


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def macro_f1_present(y_true, y_pred) -> float:
    """Macro F1 over the classes present in `y_true`.

    Blocked folds and test blocks don't always contain every class; scoring an absent
    class as F1=0 would punish the model for data it was never shown.
    """
    return float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0))


MACRO_F1_SCORER = make_scorer(macro_f1_present)


def _finite_or_none(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Headline metrics, per-class scores and confusion matrices for one set of predictions."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_totals, out=np.zeros(cm.shape, dtype=float), where=row_totals > 0)

    with warnings.catch_warnings():
        # Expected when a model predicts a class absent from this test set.
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": macro_f1_present(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy),
        "cohen_kappa": _finite_or_none(kappa),
        "per_class": per_class,
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "n_samples": int(len(y_true)),
    }


def five_class_equivalent(y_true, y_pred) -> dict:
    """Score 6-class predictions after merging S3/S4 into N3, so they compare with 5-class models."""
    return compute_metrics(six_to_five(y_true), six_to_five(y_pred), config.FIVE_CLASS_LABELS)


# ---------------------------------------------------------------------------
# Cross-validation and tuning
# ---------------------------------------------------------------------------
def fit_and_cross_validate(pipeline, param_grid: dict, X_train, y_train, cv, groups=None):
    """Tune (if a grid is given) and cross-validate on the training rows only, then refit.

    Returns (fitted_pipeline, best_params, cv_fold_scores). With a grid, fold scores are
    those of the winning parameters, so they are slightly optimistic; the held-out test
    set is never used for tuning.
    """
    if param_grid:
        search = GridSearchCV(pipeline, param_grid, scoring=MACRO_F1_SCORER, cv=cv, refit=True, error_score="raise")
        search.fit(X_train, y_train, groups=groups)
        best = search.best_index_
        fold_scores = [float(search.cv_results_[f"split{i}_test_score"][best]) for i in range(search.n_splits_)]
        return search.best_estimator_, search.best_params_, fold_scores

    results = cross_validate(
        pipeline, X_train, y_train, groups=groups, cv=cv, scoring=MACRO_F1_SCORER, error_score="raise"
    )
    fitted = clone(pipeline).fit(X_train, y_train)
    return fitted, {}, [float(score) for score in results["test_score"]]


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def impurity_importance(pipeline, feature_names: list[str]) -> dict | None:
    """Tree impurity importance per input feature; None when the model has none or uses PCA."""
    if "pca" in pipeline.named_steps:
        return None
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "feature_importances_"):
        return None
    return {name: float(value) for name, value in zip(feature_names, classifier.feature_importances_)}


def permutation_importance_scores(pipeline, X_test, y_test, feature_names: list[str], random_state: int = config.RANDOM_STATE) -> dict:
    """Drop in test macro-F1 when each feature is shuffled."""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=MACRO_F1_SCORER,
        n_repeats=config.PERMUTATION_REPEATS,
        random_state=random_state,
    )
    return {
        name: {"mean": float(result.importances_mean[i]), "std": float(result.importances_std[i])}
        for i, name in enumerate(feature_names)
    }
