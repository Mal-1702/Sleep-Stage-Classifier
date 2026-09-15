"""Shared fixtures: a small synthetic night and (when available) the real dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleepstage import config
from sleepstage.data import load_dataset

RAW_BY_STAGE = {stage: raw for raw, stage in config.RAW_LABEL_MAP.items()}

# A short, time-ordered "night": wake, light sleep, deep sleep, REM cycles, wake.
SYNTHETIC_PATTERN = [
    ("W", 80), ("N1", 25), ("N2", 60), ("S3", 30), ("S4", 12), ("N2", 40),
    ("REM", 35), ("N1", 10), ("N2", 50), ("REM", 30), ("W", 80),
]


def make_synthetic_frame(seed: int = 0) -> pd.DataFrame:
    """Feature frame in the real CSV's format, with stage-dependent feature means so models can learn."""
    rng = np.random.default_rng(seed)
    stages = [stage for stage, n in SYNTHETIC_PATTERN for _ in range(n)]
    offsets = {stage: i for i, stage in enumerate(config.SIX_CLASS_LABELS)}
    X = rng.normal(size=(len(stages), len(config.FEATURE_COLUMNS)))
    X += np.array([offsets[stage] for stage in stages], dtype=float)[:, None]
    df = pd.DataFrame(X, columns=config.FEATURE_COLUMNS)
    df[config.LABEL_COLUMN] = [RAW_BY_STAGE[stage] for stage in stages]
    df["epoch"] = np.arange(len(df))
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture(scope="session")
def real_df() -> pd.DataFrame:
    if not config.DATA_PATH.exists():
        pytest.skip(f"Real dataset not found at {config.DATA_PATH}")
    return load_dataset()
