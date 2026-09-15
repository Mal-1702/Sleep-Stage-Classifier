"""One render function per dashboard page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts
from data_access import HEADLINE_METRICS, METRIC_LABELS, Artifacts, dataset_view
from sleepstage import config
from sleepstage.data import DataValidationError, apply_class_setup, to_stage_codes, validate_features
from sleepstage.insights_demo import sleep_insights_demo


@dataclass
class PageState:
    artifacts: Artifacts | None
    selection: dict
    data_path: Path

    @property
    def result(self) -> dict | None:
        return self.artifacts.find(**self.selection) if self.artifacts else None

    def with_changes(self, **changes) -> dict:
        return {**self.selection, **changes}

    @property
    def context(self) -> dict:
        return {name: self.selection[name] for name in ("split_type", "trimming", "class_setup")}

    @property
    def context_key(self) -> str:
        return "|".join(self.context.values())


def show(fig) -> None:
    st.plotly_chart(fig, use_container_width=True)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _describe(selection: dict) -> str:
    return (
        f"{config.MODEL_DISPLAY_NAMES[selection['model']]} · {config.RESAMPLING_DISPLAY_NAMES[selection['resampling']]} · "
        f"{selection['split_type']} split · {selection['trimming']} · {selection['class_setup']}"
    )


def _require_result(state: PageState) -> dict | None:
    if state.artifacts is None:
        st.warning("No training artifacts yet. Run `python scripts/train.py` (or use Retrain in the sidebar).")
        return None
    result = state.result
    if result is None:
        st.warning("This combination was not trained. Pick another model or resampling strategy in the sidebar.")
    return result


def _default_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Baselines plus each trainable model with the default resampling strategy."""
    is_default = np.where(
        frame["model"].isin(config.BASELINE_MODELS),
        frame["resampling"] == "none",
        frame["resampling"] == config.DEFAULT_RESAMPLING,
    )
    order = {model: i for i, model in enumerate(config.MODELS)}
    return frame[is_default].sort_values("model", key=lambda s: s.map(order))


def _dataset(state: PageState) -> pd.DataFrame | None:
    if not state.data_path.exists():
        st.error(f"Dataset not found at `{state.data_path}`. Put `dream_features.csv` in `data/`.")
        return None
    return dataset_view(state.data_path, state.selection["trimming"], state.selection["class_setup"])


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
def render_overview(state: PageState) -> None:
    st.title("Automated sleep stage classification from EEG")
    st.markdown(
        "Sleep disorders are diagnosed from overnight recordings that a trained technician scores **by hand**, "
        "30 seconds at a time — hours of work per patient. This project classifies those 30-second epochs "
        "automatically from 13 EEG features, and is honest about how well that works."
    )

    frame = _dataset(state)
    if frame is not None:
        full = dataset_view(state.data_path, "untrimmed", "6class")
        cols = st.columns(4)
        cols[0].metric("Epochs in recording", f"{len(full):,}")
        cols[1].metric("Recording length", f"{len(full) * config.EPOCH_SECONDS / 3600:.1f} h")
        cols[2].metric("Features", len(config.FEATURE_COLUMNS))
        cols[3].metric("Wake share", f"{(full['stage_6class'] == 'W').mean():.0%}")
        st.caption("One overnight recording with no subject id; features were standardised before export.")

    result = _require_result(state)
    if result is None:
        return

    st.subheader("Selected experiment")
    st.caption(_describe(state.selection))
    baseline = state.artifacts.find(**state.with_changes(model="dummy_most_frequent", resampling="none"))
    cols = st.columns(4)
    for col, metric in zip(cols, HEADLINE_METRICS):
        value = result["test"][metric]
        base = baseline["test"][metric] if baseline else None
        delta = None if value is None or base is None else f"{value - base:+.3f} vs baseline"
        col.metric(METRIC_LABELS[metric], _fmt(value), delta=delta)
    if baseline:
        st.caption(
            "Baseline = always predicting the most frequent stage in training: "
            + " · ".join(f"{METRIC_LABELS[m]} {_fmt(baseline['test'][m])}" for m in HEADLINE_METRICS)
        )
    if "five_class_equivalent" in result["test"]:
        merged = result["test"]["five_class_equivalent"]
        st.info(
            "6-class model scored after merging S3/S4 into N3 (comparable with 5-class models): "
            + " · ".join(f"{METRIC_LABELS[m]} {_fmt(merged[m])}" for m in HEADLINE_METRICS)
        )

    random_result = state.artifacts.find(**state.with_changes(split_type="random"))
    blocked_result = state.artifacts.find(**state.with_changes(split_type="blocked"))
    if random_result and blocked_result:
        st.subheader("The most important finding")
        st.markdown(
            f"With a **random** split this model scores Cohen's kappa **{_fmt(random_result['test']['cohen_kappa'])}**; "
            f"holding out **whole blocks of time** drops it to **{_fmt(blocked_result['test']['cohen_kappa'])}**. "
            "Neighbouring epochs are near-copies of each other, so the random split partly measures memorisation. "
            "See *Leakage check*."
        )
    st.caption(f"Artifacts generated {state.artifacts.metrics['generated_at']} · quick run: {state.artifacts.metrics['quick_run']}")


# ---------------------------------------------------------------------------
# 2. Data explorer
# ---------------------------------------------------------------------------
def render_data_explorer(state: PageState) -> None:
    st.title("Data explorer")
    frame = _dataset(state)
    if frame is None:
        return
    st.caption(f"{len(frame):,} epochs · {state.selection['trimming']} · {state.selection['class_setup']}")

    show(charts.class_distribution(frame["stage_6class"], apply_class_setup(frame["stage_6class"], "5class")))

    left, right = st.columns([3, 1])
    feature = left.selectbox("Feature", config.FEATURE_COLUMNS)
    kind = right.radio("Plot", ["box", "violin"], horizontal=True)
    show(charts.feature_by_stage(frame, feature, kind))

    show(charts.correlation_heatmap(frame))
    st.caption("Many features are almost perfectly correlated (the five band powers especially), which is why PCA looked attractive.")
    show(charts.pca_scatter(frame))
    show(charts.hypnogram(frame["epoch"], frame["stage"], title="Hypnogram of the true labels"))


# ---------------------------------------------------------------------------
# 3. Class imbalance and resampling
# ---------------------------------------------------------------------------
def render_imbalance(state: PageState) -> None:
    st.title("Class imbalance & resampling")
    if state.artifacts is None:
        _require_result(state)
        return
    counts = state.artifacts.metrics["resampling_counts"].get(state.context_key)
    if counts:
        show(charts.resampling_counts(counts))
        st.caption(
            "Resampling runs inside the pipeline, on training folds only. Classes with a single training epoch "
            f"can't be interpolated and stay unchanged. Selective SMOTE only oversamples classes with more than "
            f"{config.SELECTIVE_SMOTE_MIN_SAMPLES} training epochs, so it matches SMOTE whenever no class falls between 2 and "
            f"{config.SELECTIVE_SMOTE_MIN_SAMPLES}."
        )

    rows = state.artifacts.results_frame(**state.context, model=config.PRIMARY_MODEL)
    if rows.empty:
        st.info("No Random Forest runs for this context.")
        return
    order = {strategy: i for i, strategy in enumerate(config.RESAMPLING_STRATEGIES)}
    rows = rows.sort_values("resampling", key=lambda s: s.map(order))
    show(charts.strategy_comparison(rows))
    st.dataframe(
        rows[["resampling_name", "cv_f1_mean", "cv_f1_std", "test_f1_macro", "test_cohen_kappa", "test_balanced_accuracy"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "resampling_name": "Resampling",
            "cv_f1_mean": st.column_config.NumberColumn("CV macro F1", format="%.3f"),
            "cv_f1_std": st.column_config.NumberColumn("CV sd", format="%.3f"),
            "test_f1_macro": st.column_config.NumberColumn("Test macro F1", format="%.3f"),
            "test_cohen_kappa": st.column_config.NumberColumn("Test kappa", format="%.3f"),
            "test_balanced_accuracy": st.column_config.NumberColumn("Test balanced acc.", format="%.3f"),
        },
    )
