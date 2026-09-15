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


# ---------------------------------------------------------------------------
# 4. Model performance
# ---------------------------------------------------------------------------
def render_performance(state: PageState) -> None:
    st.title("Model performance")
    result = _require_result(state)
    if result is None:
        return
    st.caption(_describe(state.selection))

    test = result["test"]
    if "five_class_equivalent" in test and st.toggle("Score as 5 classes (merge S3/S4 into N3)"):
        test = test["five_class_equivalent"]
    labels = test["labels"]

    cols = st.columns(4)
    for col, metric in zip(cols, HEADLINE_METRICS):
        col.metric(METRIC_LABELS[metric], _fmt(test[metric]))

    left, right = st.columns(2)
    with left:
        show(charts.confusion_matrix(test["confusion_matrix"], labels, normalized=False))
    with right:
        show(charts.confusion_matrix(test["confusion_matrix_normalized"], labels, normalized=True))
    show(charts.per_class_scores(test["per_class"], labels))

    rows = _default_rows(state.artifacts.results_frame(**state.context))
    folds = state.artifacts.folds_for(list(rows["key"])).merge(rows[["key", "model_name"]], on="key")
    show(charts.cv_fold_scores(folds))
    show(charts.cv_vs_test(rows))
    if result["tuned"]:
        st.caption(
            f"Tuned parameters: `{result['best_params']}`. CV scores come from the winning grid point, so they are "
            "slightly optimistic; the test set was never used for tuning."
        )
    if state.selection["split_type"] == "blocked":
        st.caption("Blocked CV folds are uneven: some blocks are almost pure wake, so per-fold scores vary a lot.")

    predictions = state.artifacts.predictions_for(result["key"])
    model_labels = result["test"]["labels"]
    left, right = st.columns(2)
    with left:
        show(charts.one_vs_rest_curves(predictions, model_labels, "roc"))
    with right:
        show(charts.one_vs_rest_curves(predictions, model_labels, "pr"))
    st.caption("Curves are skipped for classes with no positive or no negative test epochs.")


# ---------------------------------------------------------------------------
# 5. Leakage check
# ---------------------------------------------------------------------------
def render_leakage(state: PageState) -> None:
    st.title("Leakage check: random vs blocked split")
    if state.artifacts is None:
        _require_result(state)
        return
    random_result = state.artifacts.find(**state.with_changes(split_type="random"))
    blocked_result = state.artifacts.find(**state.with_changes(split_type="blocked"))
    if not (random_result and blocked_result):
        st.warning("This combination is missing for one of the split types.")
        return

    st.caption(_describe(state.selection).replace(f"{state.selection['split_type']} split", "random vs blocked split"))
    show(charts.leakage_comparison(random_result["test"], blocked_result["test"]))

    st.markdown(
        """
**Why the numbers differ**

EEG is recorded continuously, and each row is a 30-second slice of it. Two neighbouring slices usually
show the same brain state and nearly identical features.

- **Random split:** epochs are shuffled, so almost every test epoch has its neighbours in the training set.
  The model can recognise a near-copy instead of learning what the stage looks like.
- **Blocked split:** the recording is cut into 10 contiguous blocks of time and 2 whole blocks are held out,
  with a 5-epoch gap dropped on each side. The model must generalise to a part of the night it has never seen.

The blocked score is the more honest estimate. Real-world use (a new patient, a new night) is harder still,
because this dataset has only one recording.
"""
    )

    rows = _default_rows(state.artifacts.results_frame(trimming=state.selection["trimming"], class_setup=state.selection["class_setup"]))
    table = rows.pivot_table(index="model_name", columns="split_type", values=["test_cohen_kappa", "test_f1_macro"], sort=False)
    table.columns = [f"{METRIC_LABELS[metric.removeprefix('test_')]} ({split})" for metric, split in table.columns]
    st.subheader("Every model, both splits")
    st.dataframe(table.style.format("{:.3f}", na_rep="n/a"), use_container_width=True)


# ---------------------------------------------------------------------------
# 6. Model comparison
# ---------------------------------------------------------------------------
def render_model_comparison(state: PageState) -> None:
    st.title("Model comparison")
    if state.artifacts is None:
        _require_result(state)
        return
    rows = state.artifacts.results_frame(**state.context)
    left, right = st.columns([2, 1])
    metric = left.selectbox("Metric", HEADLINE_METRICS, index=2, format_func=METRIC_LABELS.get)
    include_variants = right.checkbox("Include every resampling strategy", value=False)
    if not include_variants:
        rows = _default_rows(rows)

    baseline = state.artifacts.find(**state.with_changes(model="dummy_most_frequent", resampling="none"))
    show(charts.model_comparison(rows, metric, baseline["test"][metric] if baseline else None))

    st.dataframe(
        rows[["model_name", "resampling_name", "test_accuracy", "test_f1_macro", "test_cohen_kappa", "test_balanced_accuracy", "cv_f1_mean", "cv_f1_std", "cv_test_gap"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "model_name": "Model",
            "resampling_name": "Resampling",
            **{
                column: st.column_config.NumberColumn(label, format="%.3f")
                for column, label in {
                    "test_accuracy": "Accuracy",
                    "test_f1_macro": "Macro F1",
                    "test_cohen_kappa": "Kappa",
                    "test_balanced_accuracy": "Balanced acc.",
                    "cv_f1_mean": "CV F1",
                    "cv_f1_std": "CV sd",
                    "cv_test_gap": "CV-test gap",
                }.items()
            },
        },
    )
    st.caption("Click a column header to sort.")

    decisions = [d for d in state.artifacts.metrics.get("pca_decision", []) if d["trimming"] == state.selection["trimming"]]
    if decisions:
        st.subheader("Does PCA(6) help?")
        st.markdown("Decided on **blocked-split cross-validation** scores only, never on the test set.")
        st.dataframe(
            pd.DataFrame(decisions).assign(choice=lambda d: d["choice"].map(config.MODEL_DISPLAY_NAMES)),
            hide_index=True,
            use_container_width=True,
            column_config={
                "cv_f1_without_pca": st.column_config.NumberColumn("CV F1 without PCA", format="%.3f"),
                "cv_f1_with_pca": st.column_config.NumberColumn("CV F1 with PCA(6)", format="%.3f"),
            },
        )


# ---------------------------------------------------------------------------
# 7. Feature importance
# ---------------------------------------------------------------------------
def render_feature_importance(state: PageState) -> None:
    st.title("Feature importance")
    result = _require_result(state)
    if result is None:
        return
    st.caption(_describe(state.selection))
    importance = state.artifacts.importance.get(result["key"])
    if importance is None:
        st.info(
            "Importance is computed for trainable models with the default resampling "
            f"({config.RESAMPLING_DISPLAY_NAMES[config.DEFAULT_RESAMPLING]}). Pick one of those in the sidebar."
        )
        return

    if importance["impurity"]:
        show(charts.feature_importance(importance["impurity"], "Random Forest impurity importance (training data)"))
        st.caption("Impurity importance is biased towards features with many split points and splits credit between correlated features.")
    else:
        st.info("Impurity importance isn't available for this model (no trees, or trees built on PCA components).")

    permutation = importance["permutation"]
    show(
        charts.feature_importance(
            {name: value["mean"] for name, value in permutation.items()},
            "Permutation importance: drop in test macro F1 when a feature is shuffled",
            errors={name: value["std"] for name, value in permutation.items()},
        )
    )
    st.caption(
        "Near-zero permutation importance can mean a feature is useless, or that a correlated feature "
        "(e.g. another band power) carries the same information."
    )


# ---------------------------------------------------------------------------
# 8. Hypnogram
# ---------------------------------------------------------------------------
def render_hypnogram(state: PageState) -> None:
    st.title("Hypnogram: true vs predicted")
    result = _require_result(state)
    if result is None:
        return
    st.caption(_describe(state.selection))
    predictions = state.artifacts.predictions_for(result["key"])
    contiguous = state.selection["split_type"] == "blocked"
    if not contiguous:
        st.info("With a random split the test epochs are scattered across the night, so they're drawn as points. Switch to the blocked split for continuous stretches.")
    elif result.get("test_blocks") is not None:
        st.caption(f"Test blocks {result['test_blocks']} of {state.artifacts.metrics['config']['n_blocks']}.")
    show(charts.hypnogram_comparison(predictions, contiguous))

    wrong = predictions[predictions["y_true"] != predictions["y_pred"]]
    st.metric("Misclassified test epochs", f"{len(wrong)} of {len(predictions)}", help="Each epoch is 30 seconds")
    if not wrong.empty:
        confusions = wrong.groupby(["y_true", "y_pred"]).size().reset_index(name="epochs").sort_values("epochs", ascending=False)
        st.subheader("Most common mistakes")
        st.dataframe(confusions.rename(columns={"y_true": "True stage", "y_pred": "Predicted as"}), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# 9. Predict
# ---------------------------------------------------------------------------
def _prediction_model(state: PageState):
    """The selected model if it was saved, otherwise this context's default Random Forest."""
    candidates = [state.selection, state.with_changes(model=config.PRIMARY_MODEL, resampling=config.DEFAULT_RESAMPLING)]
    for selection in candidates:
        result = state.artifacts.find(**selection)
        if result and state.artifacts.model_path(result["key"]):
            return selection, state.artifacts.load_model(result["key"])
    return None, None


def render_predict(state: PageState) -> None:
    st.title("Predict sleep stages")
    if state.artifacts is None:
        _require_result(state)
        return
    selection, model = _prediction_model(state)
    if model is None:
        st.warning("No saved model for this context. Run `python scripts/train.py`.")
        return
    if selection != state.selection:
        st.info("The selected combination has no saved model, so the default Random Forest for this context is used.")
    st.caption(f"Model: {_describe(selection)}")

    st.markdown(
        f"Upload a CSV with these {len(config.FEATURE_COLUMNS)} columns, one row per 30-second epoch in time order, "
        "standardised the same way as the training data. A `label` column is optional and enables scoring."
    )
    st.code(", ".join(config.FEATURE_COLUMNS), language=None)
    if state.data_path.exists():
        # Held-out test epochs only: a sample the model trained on would show an inflated score.
        test_epochs = state.artifacts.predictions_for(state.artifacts.find(**selection)["key"])["epoch"].head(120)
        full = dataset_view(state.data_path, "untrimmed", "6class")
        template = full[full["epoch"].isin(test_epochs)][config.FEATURE_COLUMNS + [config.LABEL_COLUMN]]
        st.download_button(
            f"Download a sample CSV ({len(template)} held-out test epochs)",
            template.to_csv(index=False),
            "sample_epochs.csv",
            "text/csv",
        )

    uploaded = st.file_uploader("Feature CSV", type="csv")
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded)
        validate_features(frame, require_label=False)
    except (DataValidationError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Can't use this file: {exc}")
        return

    X = frame[config.FEATURE_COLUMNS].to_numpy()
    predicted = model.predict(X)
    probabilities = pd.DataFrame(model.predict_proba(X), columns=[f"P({cls})" for cls in model.classes_])
    output = pd.concat([pd.DataFrame({"epoch": np.arange(len(frame)), "predicted_stage": predicted}), probabilities], axis=1)

    if config.LABEL_COLUMN in frame:
        try:
            truth = apply_class_setup(to_stage_codes(frame[config.LABEL_COLUMN]), selection["class_setup"]).to_numpy()
        except DataValidationError as exc:
            st.warning(f"Ignoring the label column: {exc}")
        else:
            output.insert(2, "true_stage", truth)
            st.metric("Accuracy on your labelled file", f"{np.mean(truth == predicted):.1%}")

    left, right = st.columns([3, 2])
    with left:
        show(charts.hypnogram(output["epoch"], output["predicted_stage"], title="Predicted hypnogram"))
    with right:
        shares = output["predicted_stage"].value_counts(normalize=True).to_dict()
        show(charts.stage_shares(shares))
    st.dataframe(output, hide_index=True, use_container_width=True)
    st.download_button("Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv")


# ---------------------------------------------------------------------------
# 10. Sleep insights (demo)
# ---------------------------------------------------------------------------
def render_insights(state: PageState) -> None:
    st.title("Sleep insights (demo)")
    st.warning(f"**{config.DISCLAIMER}**")
    st.markdown(
        "This page summarises a **sequence** of stages with standard descriptive measures (sleep efficiency, "
        "awakenings, stage shares) and applies hand-picked thresholds. It replaces the old “good day / bad day” "
        "feature, which judged a whole day from a single 30-second epoch."
    )

    sources = ["True labels for the recording"]
    if state.artifacts is not None and state.result is not None:
        sources.append("Model predictions on the test set")
    source = st.radio("Stage sequence", sources, horizontal=True)

    if source == sources[0]:
        frame = _dataset(state)
        if frame is None:
            return
        stages = frame["stage"].tolist()
        st.caption(f"{len(stages):,} epochs, {state.selection['trimming']}. Untrimmed recordings include hours of wake, which lowers sleep efficiency.")
    else:
        predictions = state.artifacts.predictions_for(state.result["key"])
        stages = predictions["y_pred"].tolist()
        if state.selection["split_type"] == "random":
            st.info("Random-split test epochs are scattered through the night, so sequence measures like awakenings are not meaningful here. Use the blocked split.")
        st.caption(f"{len(stages):,} predicted test epochs · {_describe(state.selection)}")

    insight = sleep_insights_demo(stages)
    summary = insight["summary"]
    st.subheader(f"{insight['category']} (illustrative score {insight['score']})")
    for note in insight["notes"]:
        st.markdown(f"- {note}")

    cols = st.columns(4)
    cols[0].metric("Sleep efficiency", f"{summary['sleep_efficiency']:.0%}")
    cols[1].metric("Total sleep", f"{summary['total_sleep_min'] / 60:.1f} h")
    cols[2].metric("Awakenings", summary["awakenings"])
    latency = summary["sleep_onset_latency_min"]
    cols[3].metric("Sleep onset latency", "n/a" if latency is None else f"{latency:.0f} min")
    cols = st.columns(3)
    cols[0].metric("Wake after sleep onset", f"{summary['wake_after_sleep_onset_min']:.0f} min")
    cols[1].metric("Stage transitions", summary["stage_transitions"])
    cols[2].metric("Longest N3 run", f"{summary['longest_n3_run_min']:.1f} min")
    show(charts.stage_shares(summary["stage_share"]))


# ---------------------------------------------------------------------------
# 11. Limitations
# ---------------------------------------------------------------------------
def render_limitations(state: PageState) -> None:
    st.title("Limitations")
    st.markdown(
        """
**Data**
- **One recording, no subject id.** Every epoch comes from a single night, so nothing here shows the model
  works on a different person. The blocked split is the best available substitute; real subject-wise
  evaluation needs features re-extracted from raw Sleep-EDF with a subject id
  (plan in `src/sleepstage/extract_features.py`).
- **Pre-scaled features.** The CSV was standardised over the whole file before any split, which leaks a
  little information from test to training. It can't be undone without the raw signals.
- **Tiny classes.** Stage 4 has 9 epochs and Stage 3 has 96, so their per-class scores rest on very few
  test epochs. The blocked test set for the untrimmed recording holds only a handful of deep-sleep epochs.
- **Single public dataset.** Features and labels come from one source (Sleep-EDF format), one EEG montage,
  one scorer.

**Method**
- Hyperparameters were tuned with cross-validation on training rows only, but reported CV scores come from
  the winning grid point and are slightly optimistic. The held-out test score is the one to trust.
- Blocked CV folds are uneven: some training blocks are almost pure wake.
- Resampling creates synthetic epochs by interpolation; it can't add real information about rare stages.

**The sleep-insights page**
- Rule-based illustration only. It is not trained, not validated against any outcome, and not medical advice.

**How this compares with published work**
- Deep models trained on raw EEG from many subjects, such as **DeepSleepNet** (Supratak et al., 2017) and
  **U-Time** (Perslev et al., 2019), report roughly 80–85% accuracy and macro F1 around 0.75–0.80 on
  Sleep-EDF under **subject-wise** evaluation.
- Those numbers are not directly comparable with this project: they test on unseen people, while this
  project tests on unseen parts of one night. A fair comparison needs the subject-wise pipeline above.
"""
    )
