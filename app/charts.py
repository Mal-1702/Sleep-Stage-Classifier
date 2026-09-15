"""Plotly figures for the dashboard. Pure functions: data in, figure out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from data_access import METRIC_LABELS
from sleepstage import config

STAGE_COLORS = {
    "W": "#F2B134",
    "N1": "#8FD3F4",
    "N2": "#3A86FF",
    "N3": "#7B4FC9",
    "S3": "#7B4FC9",
    "S4": "#C77DFF",
    "REM": "#E4572E",
}
LABEL_ORDER = ["W", "N1", "N2", "N3", "S3", "S4", "REM"]


def ordered(labels) -> list[str]:
    present = set(labels)
    return [label for label in LABEL_ORDER if label in present]


def _layout(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="", **kwargs)
    return fig


# ---------------------------------------------------------------------------
# Data explorer
# ---------------------------------------------------------------------------
def class_distribution(six_class: pd.Series, five_class: pd.Series) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("6 classes (R&K stages 3 and 4)", "5 classes (S3+S4 merged into N3)"))
    for col, stages in enumerate((six_class, five_class), start=1):
        counts = stages.value_counts()
        labels = ordered(counts.index)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[counts[label] for label in labels],
                marker_color=[STAGE_COLORS[label] for label in labels],
                text=[counts[label] for label in labels],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{x}: %{y} epochs<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_yaxes(title_text="Epochs", row=1, col=1)
    return _layout(fig, title="Class distribution before and after merging deep sleep")


def feature_by_stage(frame: pd.DataFrame, feature: str, kind: str = "box") -> go.Figure:
    plot = px.box if kind == "box" else px.violin
    fig = plot(
        frame,
        x="stage",
        y=feature,
        color="stage",
        category_orders={"stage": ordered(frame["stage"])},
        color_discrete_map=STAGE_COLORS,
    )
    fig.update_traces(showlegend=False)
    return _layout(fig, title=f"{feature} by sleep stage (standardised units)", xaxis_title="Stage")


def correlation_heatmap(frame: pd.DataFrame) -> go.Figure:
    corr = frame[config.FEATURE_COLUMNS].corr()
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto")
    return _layout(fig, height=620, title="Feature correlation (Pearson)")


def pca_scatter(frame: pd.DataFrame) -> go.Figure:
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(frame[config.FEATURE_COLUMNS])
    plot_frame = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "stage": frame["stage"].to_numpy()})
    fig = px.scatter(
        plot_frame,
        x="PC1",
        y="PC2",
        color="stage",
        opacity=0.6,
        category_orders={"stage": ordered(plot_frame["stage"])},
        color_discrete_map=STAGE_COLORS,
    )
    explained = pca.explained_variance_ratio_
    fig.update_traces(marker=dict(size=5))
    return _layout(fig, height=520, title=f"2D PCA of the 13 features (PC1 {explained[0]:.0%}, PC2 {explained[1]:.0%} of variance)")


# ---------------------------------------------------------------------------
# Hypnograms
# ---------------------------------------------------------------------------
def _stage_axis(labels) -> dict:
    order = [label for label in config.HYPNOGRAM_ORDER if label in set(labels)]
    return {label: len(order) - i for i, label in enumerate(order)}


def hypnogram(epochs, stages, title: str = "Hypnogram") -> go.Figure:
    positions = _stage_axis(stages)
    hours = np.asarray(epochs) * config.EPOCH_SECONDS / 3600
    fig = go.Figure(
        go.Scatter(
            x=hours,
            y=[positions[s] for s in stages],
            mode="lines",
            line_shape="hv",
            line=dict(color="#3A86FF", width=1.5),
            hovertemplate="%{x:.2f} h<extra></extra>",
        )
    )
    fig.update_yaxes(tickvals=list(positions.values()), ticktext=list(positions.keys()), title="Stage")
    fig.update_xaxes(title="Hours from start of recording")
    return _layout(fig, height=320, title=title, showlegend=False)


def hypnogram_comparison(predictions: pd.DataFrame, contiguous: bool) -> go.Figure:
    positions = _stage_axis(list(predictions["y_true"]) + list(predictions["y_pred"]))
    hours = predictions["epoch"].to_numpy() * config.EPOCH_SECONDS / 3600
    true_y = predictions["y_true"].map(positions)
    pred_y = predictions["y_pred"].map(positions)
    mode = "lines" if contiguous else "markers"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=true_y, mode=mode, line_shape="hv", name="True", line=dict(color="#3A86FF", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=hours, y=pred_y + 0.12, mode=mode, line_shape="hv", name="Predicted", line=dict(color="#E4572E", width=1.2), marker=dict(size=4), opacity=0.85))
    wrong = predictions["y_true"] != predictions["y_pred"]
    fig.add_trace(
        go.Scatter(
            x=hours[wrong],
            y=true_y[wrong],
            mode="markers",
            name="Misclassified",
            marker=dict(symbol="x", size=7, color="#D62728"),
            customdata=np.stack([predictions["y_true"][wrong], predictions["y_pred"][wrong]], axis=1),
            hovertemplate="%{x:.2f} h — true %{customdata[0]}, predicted %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_yaxes(tickvals=list(positions.values()), ticktext=list(positions.keys()), title="Stage")
    fig.update_xaxes(title="Hours from start of recording")
    return _layout(fig, height=420, title="True vs predicted stages on the test set")


# ---------------------------------------------------------------------------
# Imbalance and resampling
# ---------------------------------------------------------------------------
def resampling_counts(counts: dict) -> go.Figure:
    rows = [
        {"strategy": config.RESAMPLING_DISPLAY_NAMES[strategy], "stage": stage, "epochs": n}
        for strategy, per_class in counts.items()
        for stage, n in per_class.items()
    ]
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="strategy",
        y="epochs",
        color="stage",
        barmode="group",
        category_orders={"stage": ordered(frame["stage"])},
        color_discrete_map=STAGE_COLORS,
    )
    return _layout(fig, title="Training-set class counts after each resampling strategy", xaxis_title="")


def strategy_comparison(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=results["resampling_name"],
            y=results["cv_f1_mean"],
            error_y=dict(type="data", array=results["cv_f1_std"]),
            name="CV macro F1 (mean ± sd)",
            marker_color="#8FD3F4",
        )
    )
    fig.add_trace(go.Bar(x=results["resampling_name"], y=results["test_f1_macro"], name="Test macro F1", marker_color="#3A86FF"))
    fig.update_yaxes(range=[0, 1], title="Macro F1")
    return _layout(fig, title="Random Forest macro F1 by resampling strategy", barmode="group")


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
def confusion_matrix(matrix, labels: list[str], normalized: bool) -> go.Figure:
    values = np.asarray(matrix, dtype=float)
    fig = px.imshow(
        values,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1 if normalized else None,
        text_auto=".2f" if normalized else ".0f",
        aspect="auto",
    )
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="True")
    title = "Confusion matrix (row-normalised = recall)" if normalized else "Confusion matrix (epoch counts)"
    return _layout(fig, height=440, title=title, coloraxis_showscale=False)


def per_class_scores(per_class: dict, labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for metric, color in (("precision", "#8FD3F4"), ("recall", "#3A86FF"), ("f1", "#7B4FC9")):
        fig.add_trace(go.Bar(x=labels, y=[per_class[label][metric] for label in labels], name=metric.capitalize(), marker_color=color))
    supports = [per_class[label]["support"] for label in labels]
    fig.update_xaxes(ticktext=[f"{label}<br>n={n}" for label, n in zip(labels, supports)], tickvals=labels)
    fig.update_yaxes(range=[0, 1.05])
    return _layout(fig, title="Per-class precision, recall and F1 (n = test epochs)", barmode="group")


def cv_fold_scores(folds: pd.DataFrame) -> go.Figure:
    fig = px.box(folds, x="model_name", y="f1_macro", points="all", color="model_name")
    fig.update_traces(showlegend=False)
    fig.update_yaxes(range=[0, 1.05], title="Macro F1 per fold")
    return _layout(fig, title="Cross-validation macro F1 per fold", xaxis_title="")


def cv_vs_test(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=results["model_name"],
            y=results["cv_f1_mean"],
            error_y=dict(type="data", array=results["cv_f1_std"]),
            name="CV mean ± sd",
            marker_color="#8FD3F4",
        )
    )
    fig.add_trace(go.Bar(x=results["model_name"], y=results["test_f1_macro"], name="Held-out test", marker_color="#3A86FF"))
    fig.update_yaxes(range=[0, 1.05], title="Macro F1")
    return _layout(fig, title="Cross-validation vs held-out test macro F1", barmode="group")


def one_vs_rest_curves(predictions: pd.DataFrame, labels: list[str], kind: str) -> go.Figure:
    fig = go.Figure()
    for label in labels:
        column = f"prob_{label}"
        truth = (predictions["y_true"] == label).to_numpy()
        if column not in predictions or truth.all() or not truth.any():
            continue
        scores = predictions[column].to_numpy()
        if kind == "roc":
            fpr, tpr, _ = roc_curve(truth, scores)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} (AUC {auc(fpr, tpr):.2f})", line=dict(color=STAGE_COLORS[label])))
        else:
            precision, recall, _ = precision_recall_curve(truth, scores)
            ap = average_precision_score(truth, scores)
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{label} (AP {ap:.2f})", line=dict(color=STAGE_COLORS[label])))
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey")))
        return _layout(fig, title="ROC curves (one-vs-rest)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    return _layout(fig, title="Precision-recall curves (one-vs-rest)", xaxis_title="Recall", yaxis_title="Precision")


# ---------------------------------------------------------------------------
# Leakage check and model comparison
# ---------------------------------------------------------------------------
def leakage_comparison(random_metrics: dict, blocked_metrics: dict) -> go.Figure:
    metrics = [m for m in METRIC_LABELS if random_metrics.get(m) is not None and blocked_metrics.get(m) is not None]
    names = [METRIC_LABELS[m] for m in metrics]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[random_metrics[m] for m in metrics], name="Random epoch split", marker_color="#F2B134", text=[f"{random_metrics[m]:.2f}" for m in metrics], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[blocked_metrics[m] for m in metrics], name="Blocked time split", marker_color="#3A86FF", text=[f"{blocked_metrics[m]:.2f}" for m in metrics], textposition="outside"))
    fig.update_yaxes(range=[0, 1.1])
    return _layout(fig, title="Same model, two ways of splitting the data", barmode="group")


def model_comparison(results: pd.DataFrame, metric: str, baseline: float | None) -> go.Figure:
    column = f"test_{metric}"
    frame = results.dropna(subset=[column]).sort_values(column)
    colors = ["#BDBDBD" if model in config.BASELINE_MODELS else "#3A86FF" for model in frame["model"]]
    labels = frame["model_name"] + " · " + frame["resampling_name"]
    fig = go.Figure(go.Bar(x=frame[column], y=labels, orientation="h", marker_color=colors, text=frame[column].map("{:.3f}".format), textposition="outside"))
    if baseline is not None:
        fig.add_vline(x=baseline, line_dash="dash", line_color="#D62728", annotation_text="always-most-frequent baseline", annotation_position="top")
    fig.update_xaxes(range=[min(0, frame[column].min()), 1.1], title=METRIC_LABELS[metric])
    return _layout(fig, height=max(320, 34 * len(frame) + 120), title=f"Test {METRIC_LABELS[metric]} by model (grey = baselines)")


def feature_importance(values: dict, title: str, errors: dict | None = None) -> go.Figure:
    frame = pd.DataFrame({"feature": list(values), "value": list(values.values())})
    frame["error"] = frame["feature"].map(errors) if errors else 0.0
    frame = frame.sort_values("value")
    fig = go.Figure(
        go.Bar(
            x=frame["value"],
            y=frame["feature"],
            orientation="h",
            marker_color="#7B4FC9",
            error_x=dict(type="data", array=frame["error"]) if errors else None,
        )
    )
    return _layout(fig, height=460, title=title)


def stage_shares(shares: dict) -> go.Figure:
    labels = ordered(k for k, v in shares.items() if v > 0) or list(shares)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[shares[label] for label in labels],
            marker_color=[STAGE_COLORS[label] for label in labels],
            text=[f"{shares[label]:.0%}" for label in labels],
            textposition="outside",
        )
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
    return _layout(fig, height=340, title="Share of epochs per stage", showlegend=False)
