import numpy as np
import pytest

from sleepstage import config
from sleepstage.data import (
    apply_class_setup,
    blocked_split,
    make_blocks,
    make_split,
    random_split,
    to_stage_codes,
    trim_wake,
)


def _stages(df):
    return to_stage_codes(df[config.LABEL_COLUMN]).to_numpy()


def test_make_blocks_are_contiguous_and_complete():
    block_ids = make_blocks(103, 10)
    assert len(block_ids) == 103
    assert np.all(np.diff(block_ids) >= 0)
    assert sorted(set(block_ids)) == list(range(10))


def test_make_blocks_rejects_bad_counts():
    with pytest.raises(ValueError):
        make_blocks(5, 1)
    with pytest.raises(ValueError):
        make_blocks(5, 6)


def test_blocked_split_never_puts_a_block_in_train_and_test(synthetic_df):
    y = _stages(synthetic_df)
    split = blocked_split(y)
    block_ids = make_blocks(len(y))
    assert not set(block_ids[split.train_idx]) & set(block_ids[split.test_idx])
    assert not set(split.train_idx) & set(split.test_idx)


def test_blocked_split_tests_on_whole_blocks(synthetic_df):
    y = _stages(synthetic_df)
    split = blocked_split(y)
    block_ids = make_blocks(len(y))
    np.testing.assert_array_equal(split.test_idx, np.flatnonzero(np.isin(block_ids, split.test_blocks)))


def test_blocked_split_leaves_an_embargo_gap(synthetic_df):
    y = _stages(synthetic_df)
    split = blocked_split(y, embargo=5)
    distances = np.abs(split.train_idx[:, None] - split.test_idx[None, :])
    assert distances.min() > 5


def test_blocked_cv_folds_never_share_blocks(synthetic_df):
    y = _stages(synthetic_df)
    split = blocked_split(y)
    X_train = synthetic_df[config.FEATURE_COLUMNS].to_numpy()[split.train_idx]
    for train_fold, val_fold in split.cv.split(X_train, y[split.train_idx], groups=split.groups):
        assert not set(split.groups[train_fold]) & set(split.groups[val_fold])


def test_blocked_split_is_deterministic(synthetic_df):
    y = _stages(synthetic_df)
    first, second = blocked_split(y), blocked_split(y)
    assert first.test_blocks == second.test_blocks
    np.testing.assert_array_equal(first.train_idx, second.train_idx)


@pytest.mark.parametrize("trimming", ["untrimmed", "trimmed"])
def test_blocked_split_on_real_data_keeps_every_five_class_stage_in_train_and_test(real_df, trimming):
    frame = real_df.assign(stage=to_stage_codes(real_df[config.LABEL_COLUMN]))
    if trimming == "trimmed":
        frame = trim_wake(frame).reset_index(drop=True)
    split = blocked_split(frame["stage"].to_numpy())
    five = apply_class_setup(frame["stage"], "5class").to_numpy()
    for stage in config.FIVE_CLASS_LABELS:
        assert stage in set(five[split.train_idx]), f"{stage} missing from training set"
        assert stage in set(five[split.test_idx]), f"{stage} missing from test set"


def test_random_split_is_disjoint_and_complete(synthetic_df):
    y = _stages(synthetic_df)
    split = random_split(y)
    assert not set(split.train_idx) & set(split.test_idx)
    assert len(split.train_idx) + len(split.test_idx) == len(y)
    assert set(y[split.test_idx]) == set(y)


def test_make_split_rejects_unknown_type(synthetic_df):
    with pytest.raises(ValueError):
        make_split("shuffled", _stages(synthetic_df))
