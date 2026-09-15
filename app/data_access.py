"""Cached loading of training artifacts, saved models and the dataset for the dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sleepstage import config  # noqa: E402
from sleepstage.data import apply_class_setup, load_dataset, to_stage_codes, trim_wake  # noqa: E402
from sleepstage.experiments import (  # noqa: E402
    CV_FOLDS_FILE,
    IMPORTANCE_FILE,
    METRICS_FILE,
    MODELS_DIR,
    PREDICTIONS_FILE,
    model_filename,
)

REQUIRED_FILES = (METRICS_FILE, PREDICTIONS_FILE, CV_FOLDS_FILE, IMPORTANCE_FILE)
HEADLINE_METRICS = ("accuracy", "f1_macro", "cohen_kappa", "balanced_accuracy")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1_macro": "Macro F1",
    "cohen_kappa": "Cohen's kappa",
    "balanced_accuracy": "Balanced accuracy",
}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def missing_artifacts(artifacts_dir: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (artifacts_dir / name).exists()]


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def _load_model(path: str, mtime: float):
    return joblib.load(path)


class Artifacts:
    """Everything `scripts/train.py` wrote, indexed by experiment key."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.metrics = self._json(METRICS_FILE)
        self.importance = self._json(IMPORTANCE_FILE)
        self.predictions = self._csv(PREDICTIONS_FILE)
        self.cv_folds = self._csv(CV_FOLDS_FILE)
        self.experiments = {result["key"]: result for result in self.metrics["experiments"]}

    def _json(self, name: str) -> dict:
        path = self.directory / name
        return _read_json(str(path), _mtime(path))

    def _csv(self, name: str) -> pd.DataFrame:
        path = self.directory / name
        return _read_csv(str(path), _mtime(path))

    def options(self, field: str, **filters) -> list[str]:
        """Distinct values of `field` among experiments matching `filters`, in config order."""
        values = {
            r[field] for r in self.experiments.values() if all(r[name] == value for name, value in filters.items())
        }
        order = {
            "split_type": config.SPLIT_TYPES,
            "trimming": config.TRIM_OPTIONS,
            "class_setup": list(config.CLASS_SETUPS),
            "model": config.MODELS,
            "resampling": config.RESAMPLING_STRATEGIES,
        }[field]
        return [value for value in order if value in values]

    def find(self, **fields) -> dict | None:
        key = "|".join(fields[name] for name in ("split_type", "trimming", "class_setup", "resampling", "model"))
        return self.experiments.get(key)

    def results_frame(self, **filters) -> pd.DataFrame:
        rows = []
        for r in self.experiments.values():
            if not all(r[name] == value for name, value in filters.items()):
                continue
            row = {
                "key": r["key"],
                "split_type": r["split_type"],
                "trimming": r["trimming"],
                "class_setup": r["class_setup"],
                "model": r["model"],
                "model_name": config.MODEL_DISPLAY_NAMES[r["model"]],
                "resampling": r["resampling"],
                "resampling_name": config.RESAMPLING_DISPLAY_NAMES[r["resampling"]],
                "cv_f1_mean": r["cv_f1_mean"],
                "cv_f1_std": r["cv_f1_std"],
                "cv_test_gap": r["cv_test_gap"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
            }
            row.update({f"test_{metric}": r["test"][metric] for metric in HEADLINE_METRICS})
            rows.append(row)
        return pd.DataFrame(rows)

    def predictions_for(self, key: str) -> pd.DataFrame:
        frame = self.predictions[self.predictions["key"] == key]
        return frame.dropna(axis=1, how="all").sort_values("epoch").reset_index(drop=True)

    def folds_for(self, keys: list[str]) -> pd.DataFrame:
        return self.cv_folds[self.cv_folds["key"].isin(keys)]

    def model_path(self, key: str) -> Path | None:
        path = self.directory / MODELS_DIR / model_filename(key)
        return path if path.exists() else None

    def load_model(self, key: str):
        path = self.model_path(key)
        return None if path is None else _load_model(str(path), _mtime(path))


def load_artifacts(directory: Path) -> Artifacts:
    return Artifacts(directory)


@st.cache_data(show_spinner=False)
def _load_frame(path: str, mtime: float) -> pd.DataFrame:
    df = load_dataset(path)
    return df.assign(stage_6class=to_stage_codes(df[config.LABEL_COLUMN]))


def dataset_view(data_path: Path, trimming: str, class_setup: str) -> pd.DataFrame:
    """The dataset as a given experiment context sees it: label setup and optional wake trimming."""
    df = _load_frame(str(data_path), _mtime(data_path))
    frame = df.assign(stage=apply_class_setup(df["stage_6class"], class_setup))
    if trimming == "trimmed":
        frame = trim_wake(frame, stage_column="stage_6class")
    return frame.reset_index(drop=True)
