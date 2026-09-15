"""Sleep insights DEMO.

Rule-based illustration only — not trained, not validated, not medical advice.

It summarises a whole sequence of sleep stages (a night, or a contiguous part of
one) with standard descriptive measures, then applies hand-picked thresholds to
label the night. Nothing here has been checked against real outcomes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from . import config

DISCLAIMER = config.DISCLAIMER

VALID_STAGES = set(config.SIX_CLASS_LABELS) | set(config.FIVE_CLASS_LABELS)
CATEGORIES = ("Consolidated night", "Mixed night", "Fragmented night", "No sleep detected")

# Illustrative thresholds, loosely based on typical adult values.
GOOD_SLEEP_EFFICIENCY = 0.85
FAIR_SLEEP_EFFICIENCY = 0.70
TYPICAL_N3_SHARE = 0.15
TYPICAL_REM_SHARE = 0.15
MANY_AWAKENINGS_PER_HOUR = 4


def summarize_night(stages: Sequence[str], epoch_seconds: int = config.EPOCH_SECONDS) -> dict:
    """Descriptive sleep measures for a time-ordered sequence of stage codes."""
    stages = list(stages)
    if not stages:
        raise ValueError("stages must not be empty")
    unknown = sorted(set(stages) - VALID_STAGES)
    if unknown:
        raise ValueError(f"Unknown stage codes: {unknown}")

    merged = [config.N3_MERGE_MAP.get(stage, stage) for stage in stages]
    minutes_per_epoch = epoch_seconds / 60
    n_epochs = len(merged)
    counts = Counter(merged)
    is_sleep = [stage != config.WAKE_LABEL for stage in merged]
    sleep_epochs = sum(is_sleep)

    summary = {
        "n_epochs": n_epochs,
        "time_in_record_min": n_epochs * minutes_per_epoch,
        "total_sleep_min": sleep_epochs * minutes_per_epoch,
        "sleep_efficiency": sleep_epochs / n_epochs,
        "stage_share": {stage: counts.get(stage, 0) / n_epochs for stage in config.FIVE_CLASS_LABELS},
        "stage_transitions": sum(a != b for a, b in zip(merged, merged[1:])),
        "sleep_onset_latency_min": None,
        "wake_after_sleep_onset_min": 0.0,
        "awakenings": 0,
        "longest_n3_run_min": 0.0,
    }
    if sleep_epochs == 0:
        return summary

    first_sleep = is_sleep.index(True)
    last_sleep = n_epochs - 1 - is_sleep[::-1].index(True)
    sleep_period = merged[first_sleep : last_sleep + 1]
    summary["sleep_onset_latency_min"] = first_sleep * minutes_per_epoch
    summary["wake_after_sleep_onset_min"] = sleep_period.count(config.WAKE_LABEL) * minutes_per_epoch
    summary["awakenings"] = sum(
        a != config.WAKE_LABEL and b == config.WAKE_LABEL for a, b in zip(sleep_period, sleep_period[1:])
    )

    longest = current = 0
    for stage in merged:
        current = current + 1 if stage == "N3" else 0
        longest = max(longest, current)
    summary["longest_n3_run_min"] = longest * minutes_per_epoch
    return summary


def label_night(summary: dict) -> dict:
    """Apply the illustrative thresholds to a `summarize_night` result."""
    if summary["total_sleep_min"] == 0:
        return {"category": "No sleep detected", "score": 0, "notes": ["No sleep epochs in this sequence."]}

    score = 0
    notes = []
    efficiency = summary["sleep_efficiency"]
    if efficiency >= GOOD_SLEEP_EFFICIENCY:
        score += 2
        notes.append(f"Sleep efficiency {efficiency:.0%} is at or above {GOOD_SLEEP_EFFICIENCY:.0%}.")
    elif efficiency >= FAIR_SLEEP_EFFICIENCY:
        score += 1
        notes.append(f"Sleep efficiency {efficiency:.0%} is moderate.")
    else:
        score -= 1
        notes.append(f"Sleep efficiency {efficiency:.0%} is low (long wake periods in the sequence).")

    share = summary["stage_share"]
    if share["N3"] >= TYPICAL_N3_SHARE:
        score += 1
        notes.append(f"Deep sleep (N3) makes up {share['N3']:.0%} of the sequence.")
    if share["REM"] >= TYPICAL_REM_SHARE:
        score += 1
        notes.append(f"REM makes up {share['REM']:.0%} of the sequence.")

    sleep_hours = summary["total_sleep_min"] / 60
    awakenings_per_hour = summary["awakenings"] / sleep_hours
    if awakenings_per_hour >= MANY_AWAKENINGS_PER_HOUR:
        score -= 1
        notes.append(f"{awakenings_per_hour:.1f} awakenings per hour of sleep.")

    if score >= 3:
        category = "Consolidated night"
    elif score <= 0:
        category = "Fragmented night"
    else:
        category = "Mixed night"
    return {"category": category, "score": score, "notes": notes}


def sleep_insights_demo(stages: Sequence[str], epoch_seconds: int = config.EPOCH_SECONDS) -> dict:
    """Summary measures plus an illustrative label for a stage sequence."""
    summary = summarize_night(stages, epoch_seconds)
    return {"summary": summary, **label_night(summary), "disclaimer": DISCLAIMER}
