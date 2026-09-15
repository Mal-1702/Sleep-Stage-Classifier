import pytest

from sleepstage import config, insights_demo
from sleepstage.insights_demo import CATEGORIES, VALID_STAGES, sleep_insights_demo, summarize_night


@pytest.mark.parametrize("stage", sorted(VALID_STAGES))
def test_insights_demo_returns_valid_output_for_every_stage_label(stage):
    result = sleep_insights_demo([stage] * 40)
    assert result["category"] in CATEGORIES
    assert result["disclaimer"] == config.DISCLAIMER
    assert isinstance(result["notes"], list)
    assert sum(result["summary"]["stage_share"].values()) == pytest.approx(1.0)


def test_mixed_night_returns_valid_output():
    stages = ["W"] * 10 + ["N1"] * 5 + ["N2"] * 40 + ["S3"] * 15 + ["S4"] * 5 + ["REM"] * 20 + ["W"] * 5
    result = sleep_insights_demo(stages)
    assert result["category"] in CATEGORIES
    assert result["summary"]["stage_share"]["N3"] == pytest.approx(20 / len(stages))


def test_wake_only_sequence_reports_no_sleep():
    result = sleep_insights_demo(["W"] * 30)
    assert result["category"] == "No sleep detected"
    assert result["summary"]["sleep_onset_latency_min"] is None


def test_summary_measures_are_computed_over_the_sleep_period():
    stages = ["W"] * 4 + ["N2"] * 10 + ["W"] * 2 + ["N2"] * 5 + ["REM"] * 5 + ["W"] * 3
    summary = summarize_night(stages)
    assert summary["sleep_onset_latency_min"] == 2.0
    assert summary["awakenings"] == 1
    assert summary["wake_after_sleep_onset_min"] == 1.0
    assert summary["sleep_efficiency"] == pytest.approx(20 / 29)


def test_longest_n3_run_merges_stage_3_and_4():
    assert summarize_night(["N2", "S3", "S4", "S4", "N2", "N3"])["longest_n3_run_min"] == 1.5


def test_unknown_or_empty_stages_raise():
    with pytest.raises(ValueError):
        summarize_night(["W", "Sleep stage W"])
    with pytest.raises(ValueError):
        summarize_night([])


def test_module_carries_the_disclaimer():
    assert config.DISCLAIMER in insights_demo.__doc__
