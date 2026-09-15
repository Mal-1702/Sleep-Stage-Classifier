import numpy as np
import pytest

from sleepstage import config
from sleepstage.data import (
    DataValidationError,
    apply_class_setup,
    load_dataset,
    six_to_five,
    to_stage_codes,
    trim_wake,
)


def test_label_map_covers_every_label_in_csv(real_df):
    assert set(real_df[config.LABEL_COLUMN].unique()) <= set(config.RAW_LABEL_MAP)
    assert set(config.RAW_LABEL_MAP.values()) == set(config.SIX_CLASS_LABELS)
    assert not to_stage_codes(real_df[config.LABEL_COLUMN]).isna().any()


def test_n3_merging_produces_exactly_five_classes(real_df):
    stages = apply_class_setup(to_stage_codes(real_df[config.LABEL_COLUMN]), "5class")
    assert sorted(stages.unique()) == sorted(config.FIVE_CLASS_LABELS)
    assert stages.nunique() == 5


def test_n3_merging_keeps_every_deep_sleep_epoch(synthetic_df):
    six = to_stage_codes(synthetic_df[config.LABEL_COLUMN])
    five = apply_class_setup(six, "5class")
    assert (five == "N3").sum() == six.isin(["S3", "S4"]).sum()
    assert not five.isin(["S3", "S4"]).any()


def test_six_class_setup_keeps_stage_3_and_4(synthetic_df):
    stages = apply_class_setup(to_stage_codes(synthetic_df[config.LABEL_COLUMN]), "6class")
    assert sorted(stages.unique()) == sorted(config.SIX_CLASS_LABELS)


def test_unknown_class_setup_raises(synthetic_df):
    with pytest.raises(ValueError):
        apply_class_setup(to_stage_codes(synthetic_df[config.LABEL_COLUMN]), "4class")


def test_six_to_five_maps_only_deep_sleep():
    assert list(six_to_five(["W", "S3", "S4", "REM", "N2"])) == ["W", "N3", "N3", "REM", "N2"]


def test_load_dataset_rejects_unknown_labels(tmp_path, synthetic_df):
    df = synthetic_df.drop(columns="epoch")
    df.loc[0, config.LABEL_COLUMN] = "Movement time"
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Movement time"):
        load_dataset(path)


def test_load_dataset_rejects_missing_columns(tmp_path, synthetic_df):
    path = tmp_path / "missing.csv"
    synthetic_df.drop(columns=["epoch", "delta_power"]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="delta_power"):
        load_dataset(path)


def test_load_dataset_rejects_nan_features(tmp_path, synthetic_df):
    df = synthetic_df.drop(columns="epoch")
    df.loc[3, "alpha_power"] = np.nan
    path = tmp_path / "nan.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="NaN"):
        load_dataset(path)


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_trim_wake_keeps_margin_around_sleep(synthetic_df):
    frame = synthetic_df.assign(stage=to_stage_codes(synthetic_df[config.LABEL_COLUMN]))
    trimmed = trim_wake(frame, margin=10)
    # The synthetic night has 80 wake epochs on each side.
    assert len(trimmed) == len(frame) - 2 * 70
    assert (trimmed["stage"].iloc[:10] == "W").all()
    assert trimmed["stage"].iloc[10] != "W"
    assert (trimmed["stage"].iloc[-10:] == "W").all()
    assert trimmed["stage"].iloc[-11] != "W"


def test_trim_wake_without_sleep_returns_everything(synthetic_df):
    frame = synthetic_df.assign(stage="W")
    assert len(trim_wake(frame)) == len(frame)
