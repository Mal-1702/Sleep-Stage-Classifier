"""Streamlit dashboard for the sleep stage classification experiments.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import views  # noqa: E402
from data_access import load_artifacts, missing_artifacts  # noqa: E402
from sleepstage import config  # noqa: E402
from sleepstage.experiments import run_all  # noqa: E402

PAGES = {
    "Overview": views.render_overview,
    "Data explorer": views.render_data_explorer,
    "Class imbalance & resampling": views.render_imbalance,
    "Model performance": views.render_performance,
    "Leakage check": views.render_leakage,
    "Model comparison": views.render_model_comparison,
    "Feature importance": views.render_feature_importance,
    "Hypnogram": views.render_hypnogram,
    "Predict": views.render_predict,
    "Sleep insights (demo)": views.render_insights,
    "Limitations": views.render_limitations,
}
PAGES_WITHOUT_ARTIFACTS = {"Data explorer", "Sleep insights (demo)", "Limitations"}

SPLIT_NAMES = {"random": "Random epoch split (leaky)", "blocked": "Blocked time split (honest)"}
TRIM_NAMES = {"untrimmed": "Full recording", "trimmed": "Wake trimmed to ±30 min"}
SETUP_NAMES = {"5class": "5 classes (S3+S4 = N3)", "6class": "6 classes (S3, S4 separate)"}
DEFAULTS = {
    "split_type": "blocked",
    "trimming": "untrimmed",
    "class_setup": "5class",
    "model": config.PRIMARY_MODEL,
    "resampling": config.DEFAULT_RESAMPLING,
}


@st.cache_resource(show_spinner=False)
def retrain(output_dir: str, data_path: str, quick: bool, data_mtime: float, _progress=None) -> str:
    """Run the experiment grid once per (settings, dataset version); repeat clicks reuse the result."""
    run_all(output_dir=output_dir, data_path=data_path, quick=quick, progress=_progress)
    return datetime.now().isoformat(timespec="seconds")


def _select(label: str, field: str, options: list[str], names: dict) -> str:
    # Seed the value through session state rather than `index`: changing `index` between
    # reruns gives the widget a new identity and silently resets the user's choice.
    if st.session_state.get(field) not in options:
        st.session_state[field] = DEFAULTS[field] if DEFAULTS[field] in options else options[0]
    return st.sidebar.selectbox(label, options, format_func=names.get, key=field)


def selection_controls(artifacts) -> dict:
    st.sidebar.subheader("Experiment")
    if artifacts is None:
        return {
            **DEFAULTS,
            "trimming": _select("Wake trimming", "trimming", config.TRIM_OPTIONS, TRIM_NAMES),
            "class_setup": _select("Classes", "class_setup", list(config.CLASS_SETUPS), SETUP_NAMES),
        }
    selection = {
        "split_type": _select("Split", "split_type", artifacts.options("split_type"), SPLIT_NAMES),
        "trimming": _select("Wake trimming", "trimming", artifacts.options("trimming"), TRIM_NAMES),
        "class_setup": _select("Classes", "class_setup", artifacts.options("class_setup"), SETUP_NAMES),
    }
    selection["model"] = _select("Model", "model", artifacts.options("model", **selection), config.MODEL_DISPLAY_NAMES)
    selection["resampling"] = _select(
        "Resampling", "resampling", artifacts.options("resampling", **selection), config.RESAMPLING_DISPLAY_NAMES
    )
    return selection


def retrain_controls(data_path: Path) -> None:
    with st.sidebar.expander("Retrain models"):
        quick = st.checkbox(
            "Quick run (about 2 minutes)",
            value=True,
            help="Baselines and Random Forest only, no hyperparameter search. Replaces the current artifacts.",
        )
        if not quick:
            st.caption("A full run with hyperparameter search takes around 10–20 minutes.")
        if st.button("Retrain", disabled=not data_path.exists()):
            bar = st.progress(0.0, text="Starting")

            def progress(index: int, total: int, key: str) -> None:
                bar.progress(index / total, text=f"{index}/{total} · {key}")

            retrain(str(config.ARTIFACTS_DIR), str(data_path), quick, data_path.stat().st_mtime, _progress=progress)
            st.cache_data.clear()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Sleep Stage Classification", page_icon="🌙", layout="wide")
    st.sidebar.title("Sleep stage classifier")
    page = st.sidebar.radio("Page", list(PAGES), key="page")

    artifacts = None
    missing = missing_artifacts(config.ARTIFACTS_DIR)
    if not missing:
        try:
            artifacts = load_artifacts(config.ARTIFACTS_DIR)
        except (json.JSONDecodeError, KeyError, pd.errors.ParserError) as exc:
            st.error(f"Artifacts in `{config.ARTIFACTS_DIR}` look corrupted ({exc}). Retrain to regenerate them.")

    selection = selection_controls(artifacts)
    retrain_controls(config.DATA_PATH)

    if artifacts is None and page not in PAGES_WITHOUT_ARTIFACTS:
        st.title(page)
        st.error(
            f"No training artifacts found in `{config.ARTIFACTS_DIR}`"
            + (f" (missing: {', '.join(missing)})." if missing else ".")
            + " Train the models first from the project root:"
        )
        st.code("python scripts/train.py", language="bash")
        st.caption("Or use **Retrain models** in the sidebar. Data explorer, Sleep insights and Limitations work without artifacts.")
        return

    PAGES[page](views.PageState(artifacts, selection, config.DATA_PATH))


main()
