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


# ---------------------------------------------------------------------------
# Running one experiment
# ---------------------------------------------------------------------------
def run_experiment(spec: ExperimentSpec, ctx: DataContext, tune: bool = True, fixed_params: dict | None = None):
    """Fit, cross-validate and test one experiment. Returns (result, fitted_pipeline, predictions)."""
    split = ctx.split
    X_train, y_train = ctx.X[split.train_idx], ctx.y[split.train_idx]
    X_test, y_test = ctx.X[split.test_idx], ctx.y[split.test_idx]
    labels = labels_for_setup(spec.class_setup)

    pipeline = build_pipeline(spec.model, spec.resampling)
    param_grid = param_grid_for(spec.model) if tune else {}
    if fixed_params:
        pipeline.set_params(**fixed_params)
        param_grid = {}

    fitted, best_params, fold_scores = fit_and_cross_validate(
        pipeline, param_grid, X_train, y_train, split.cv, groups=split.groups
    )
    y_pred = fitted.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, labels)
    if spec.class_setup == "6class":
        metrics["five_class_equivalent"] = five_class_equivalent(y_test, y_pred)

    cv_mean = float(np.mean(fold_scores))
    result = {
        **asdict(spec),
        "key": spec.key,
        "best_params": {name: value for name, value in (fixed_params or best_params).items()},
        "tuned": bool(param_grid),
        "cv_fold_scores": fold_scores,
        "cv_f1_mean": cv_mean,
        "cv_f1_std": float(np.std(fold_scores)),
        "cv_test_gap": abs(cv_mean - metrics["f1_macro"]),
        "test": metrics,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_class_counts": dict(Counter(y_train)),
        "test_blocks": split.test_blocks,
    }

    predictions = pd.DataFrame(
        {
            "key": spec.key,
            "epoch": ctx.frame["epoch"].to_numpy()[split.test_idx],
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    probabilities = fitted.predict_proba(X_test)
    for i, cls in enumerate(fitted.classes_):
        predictions[f"prob_{cls}"] = probabilities[:, i]
    return result, fitted, predictions


def resampling_counts(ctx: DataContext) -> dict:
    """Training-set class counts before and after each resampling strategy."""
    X_train, y_train = ctx.X[ctx.split.train_idx], ctx.y[ctx.split.train_idx]
    counts = {}
    for strategy in config.RESAMPLING_STRATEGIES:
        _, y_resampled = resample(X_train, y_train, strategy)
        counts[strategy] = {str(cls): int(n) for cls, n in Counter(y_resampled).items()}
    return counts


# ---------------------------------------------------------------------------
# Running the grid and saving artifacts
# ---------------------------------------------------------------------------
def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def pca_decision(results: list[dict]) -> list[dict]:
    """Compare Random Forest with and without PCA on blocked-split CV scores."""
    decisions = []
    for trimming in config.TRIM_OPTIONS:
        for class_setup in config.CLASS_SETUPS:
            scores = {}
            for model in ("random_forest", "random_forest_pca"):
                key = "|".join(("blocked", trimming, class_setup, config.DEFAULT_RESAMPLING, model))
                match = next((r for r in results if r["key"] == key), None)
                if match:
                    scores[model] = match["cv_f1_mean"]
            if len(scores) == 2:
                decisions.append(
                    {
                        "trimming": trimming,
                        "class_setup": class_setup,
                        "cv_f1_without_pca": scores["random_forest"],
                        "cv_f1_with_pca": scores["random_forest_pca"],
                        "choice": max(scores, key=scores.get),
                    }
                )
    return decisions


def run_all(
    output_dir: str | Path = config.ARTIFACTS_DIR,
    data_path: str | Path | None = None,
    quick: bool = False,
    tune: bool = True,
    save_all_models: bool = False,
    progress=None,
) -> dict:
    """Run every experiment and write metrics, predictions, CV folds, importances and models."""
    output_dir = Path(output_dir)
    models_dir = output_dir / MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    tune = tune and not quick

    df = load_dataset(data_path)
    specs = build_experiment_specs(quick)
    results, prediction_frames, importances, counts_by_context = [], [], {}, {}
    contexts: dict[str, DataContext] = {}
    default_params: dict[str, dict] = {}

    for index, spec in enumerate(specs, start=1):
        if spec.context_key not in contexts:
            contexts[spec.context_key] = prepare_context(df, spec.split_type, spec.trimming, spec.class_setup)
            counts_by_context[spec.context_key] = resampling_counts(contexts[spec.context_key])
        ctx = contexts[spec.context_key]

        # Other resampling strategies reuse the params tuned with the default strategy.
        fixed = None
        if not is_baseline(spec.model) and spec.resampling != config.DEFAULT_RESAMPLING:
            fixed = default_params.get(f"{spec.context_key}|{spec.model}")

        logger.info("[%d/%d] %s", index, len(specs), spec.key)
        if progress is not None:
            progress(index, len(specs), spec.key)
        result, fitted, predictions = run_experiment(spec, ctx, tune=tune, fixed_params=fixed)
        results.append(result)
        prediction_frames.append(predictions)

        is_default = is_baseline(spec.model) or spec.resampling == config.DEFAULT_RESAMPLING
        if not is_baseline(spec.model) and spec.resampling == config.DEFAULT_RESAMPLING:
            default_params[f"{spec.context_key}|{spec.model}"] = result["best_params"]

        if is_default and not is_baseline(spec.model):
            X_test, y_test = ctx.X[ctx.split.test_idx], ctx.y[ctx.split.test_idx]
            importances[spec.key] = {
                "impurity": impurity_importance(fitted, config.FEATURE_COLUMNS),
                "permutation": permutation_importance_scores(fitted, X_test, y_test, config.FEATURE_COLUMNS),
            }
        if is_default or save_all_models:
            joblib.dump(fitted, models_dir / model_filename(spec.key), compress=3)

    stage_counts = to_stage_codes(df[config.LABEL_COLUMN]).value_counts()
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_run": quick,
        "tuned": tune,
        "config": {
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "cv_folds": config.CV_FOLDS,
            "n_blocks": config.N_BLOCKS,
            "n_test_blocks": config.N_TEST_BLOCKS,
            "block_embargo_epochs": config.BLOCK_EMBARGO_EPOCHS,
            "wake_trim_margin_epochs": config.WAKE_TRIM_MARGIN_EPOCHS,
            "default_resampling": config.DEFAULT_RESAMPLING,
            "pca_components": config.PCA_COMPONENTS,
        },
        "dataset": {
            "path": str(data_path or config.DATA_PATH),
            "n_epochs": int(len(df)),
            "stage_counts": {stage: int(n) for stage, n in stage_counts.items()},
        },
        "pca_decision": pca_decision(results),
        "resampling_counts": counts_by_context,
        "experiments": results,
    }

    (output_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    (output_dir / IMPORTANCE_FILE).write_text(json.dumps(importances, indent=2, default=_json_default), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / PREDICTIONS_FILE, index=False)
    folds = [
        {"key": r["key"], "fold": fold, "f1_macro": score}
        for r in results
        for fold, score in enumerate(r["cv_fold_scores"])
    ]
    pd.DataFrame(folds).to_csv(output_dir / CV_FOLDS_FILE, index=False)
    logger.info("Wrote %d experiments to %s", len(results), output_dir)
    return metrics
