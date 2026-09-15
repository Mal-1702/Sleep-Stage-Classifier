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
