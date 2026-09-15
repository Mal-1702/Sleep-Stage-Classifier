from collections import Counter

import numpy as np
import pytest

from sleepstage import config, models
from sleepstage.data import blocked_split, random_split, to_stage_codes
from sleepstage.evaluate import compute_metrics, fit_and_cross_validate, five_class_equivalent


def _xy(df):
    return df[config.FEATURE_COLUMNS].to_numpy(), to_stage_codes(df[config.LABEL_COLUMN]).to_numpy(dtype=object)


def test_resampling_is_applied_only_to_training_data(synthetic_df, monkeypatch):
    calls = []
    original = models.resample

    def spy(X, y, **kwargs):
        calls.append(len(y))
        return original(X, y, **kwargs)

    monkeypatch.setattr(models, "resample", spy)

    X, y = _xy(synthetic_df)
    split = random_split(y)
    X_test, y_test = X[split.test_idx].copy(), y[split.test_idx].copy()

    pipeline = models.build_pipeline("random_forest", "smote").fit(X[split.train_idx], y[split.train_idx])
    assert calls == [len(split.train_idx)]

    predictions = pipeline.predict(X[split.test_idx])
    pipeline.predict_proba(X[split.test_idx])
    assert calls == [len(split.train_idx)], "resampling must not run at predict time"
    assert len(predictions) == len(y_test)
    np.testing.assert_array_equal(X[split.test_idx], X_test)
    np.testing.assert_array_equal(y[split.test_idx], y_test)


@pytest.mark.parametrize("strategy", ["smote", "borderline_smote", "adasyn"])
def test_resample_grows_minorities_and_leaves_singletons(strategy):
    rng = np.random.default_rng(0)
    y = np.array(["A"] * 120 + ["B"] * 40 + ["C"] * 6 + ["D"] * 1, dtype=object)
    X = rng.normal(size=(len(y), 4)) + np.array([{"A": 0, "B": 1, "C": 2, "D": 3}[v] for v in y])[:, None] * 0.5
    _, y_res = models.resample(X, y, strategy)
    counts = Counter(y_res)
    assert counts["A"] == 120
    assert counts["D"] == 1
    assert counts["B"] >= 40 and counts["C"] >= 6
    if strategy == "smote":
        assert counts["B"] == counts["C"] == 120


def test_selective_smote_skips_classes_at_or_below_threshold():
    rng = np.random.default_rng(1)
    threshold = config.SELECTIVE_SMOTE_MIN_SAMPLES
    y = np.array(["A"] * 100 + ["B"] * (threshold + 5) + ["C"] * threshold, dtype=object)
    X = rng.normal(size=(len(y), 3))
    _, y_res = models.resample(X, y, "selective_smote")
    counts = Counter(y_res)
    assert counts["B"] == 100
    assert counts["C"] == threshold


def test_resample_none_returns_input_unchanged():
    X, y = np.zeros((4, 2)), np.array(["A", "A", "B", "B"])
    X_res, y_res = models.resample(X, y, "none")
    assert X_res is X and y_res is y


def test_unknown_resampling_strategy_raises():
    with pytest.raises(ValueError):
        models.resample(np.zeros((4, 2)), np.array(["A", "A", "A", "B"]), "oversample_everything")


def test_baselines_reject_resampling():
    with pytest.raises(ValueError):
        models.build_pipeline("dummy_most_frequent", "smote")


@pytest.mark.parametrize("model", config.MODELS)
def test_every_model_pipeline_fits_and_predicts(synthetic_df, model):
    X, y = _xy(synthetic_df)
    split = random_split(y)
    resampling = "none" if models.is_baseline(model) else config.DEFAULT_RESAMPLING
    pipeline = models.build_pipeline(model, resampling).fit(X[split.train_idx], y[split.train_idx])
    predictions = pipeline.predict(X[split.test_idx])
    probabilities = pipeline.predict_proba(X[split.test_idx])
    assert len(predictions) == len(split.test_idx)
    assert set(predictions) <= set(config.SIX_CLASS_LABELS)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    if not models.is_baseline(model):
        assert np.mean(predictions == y[split.test_idx]) > 0.5


def test_fit_and_cross_validate_tunes_with_blocked_groups(synthetic_df):
    X, y = _xy(synthetic_df)
    split = blocked_split(y)
    pipeline = models.build_pipeline("random_forest", "smote")
    fitted, best_params, fold_scores = fit_and_cross_validate(
        pipeline, {"clf__min_samples_leaf": [1, 4]}, X[split.train_idx], y[split.train_idx], split.cv, split.groups
    )
    assert set(best_params) == {"clf__min_samples_leaf"}
    assert len(fold_scores) == split.cv.get_n_splits()
    assert len(fitted.predict(X[split.test_idx])) == len(split.test_idx)


def test_compute_metrics_reports_all_headline_metrics():
    y_true = ["W", "W", "N1", "N2", "S3", "S4", "REM"]
    y_pred = ["W", "N1", "N1", "N2", "S4", "S3", "REM"]
    metrics = compute_metrics(y_true, y_pred, config.SIX_CLASS_LABELS)
    for key in ("accuracy", "f1_macro", "balanced_accuracy", "cohen_kappa", "per_class", "confusion_matrix"):
        assert key in metrics
    assert len(metrics["confusion_matrix"]) == 6
    # Swapping S3 and S4 is a miss in 6-class, but correct once both are N3.
    merged = five_class_equivalent(y_true, y_pred)
    assert merged["per_class"]["N3"]["recall"] == 1.0
    assert merged["accuracy"] > metrics["accuracy"]
