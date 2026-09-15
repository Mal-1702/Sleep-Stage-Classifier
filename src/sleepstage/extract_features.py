"""Feature extraction from raw Sleep-EDF recordings — NOT IMPLEMENTED YET.

Why this matters
----------------
`data/dream_features.csv` has no subject or recording id, and its features were
standardised on the whole dataset before any split. That rules out true
subject-wise evaluation and leaks a little test-set information into training.
Re-extracting features from the raw recordings fixes both.

Plan (TODO)
-----------
1. Download Sleep-EDF Expanded (PhysioNet) — `mne.datasets.sleep_physionet.age.fetch_data`
   gives PSG + hypnogram EDF pairs for many subjects.
2. For each recording, load with `mne.io.read_raw_edf`, keep the `EEG Fpz-Cz` channel,
   and attach annotations from the hypnogram file.
3. Cut non-overlapping 30-second epochs with `mne.events_from_annotations` +
   `mne.Epochs` (tmax = 30 - 1/sfreq). Drop "Movement time" and "Sleep stage ?".
4. Compute the same 13 features per epoch (see `FEATURE_COLUMNS` in config.py):
   - delta/theta/alpha/beta/gamma band power from a Welch PSD
     (0.5-4, 4-8, 8-13, 13-30, 30-45 Hz)
   - std_dev, skewness
   - Hjorth activity, mobility, complexity
   - delta/beta and alpha/delta power ratios
   - Shannon entropy of the normalised signal histogram or PSD
   Do NOT standardise here; scaling belongs inside the model pipeline.
5. Write one row per epoch with columns: subject_id, night, epoch, <13 features>, label.
6. Evaluate with GroupKFold / a held-out subject set on `subject_id`
   (see `data.blocked_split` for the current, weaker, single-recording substitute).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def extract_recording_features(psg_path: str | Path, hypnogram_path: str | Path, subject_id: str) -> pd.DataFrame:
    """Return one row per 30-second epoch: subject_id, night, epoch, 13 features, label."""
    raise NotImplementedError("See the TODO plan in this module's docstring.")


def build_dataset(recordings: list[tuple[Path, Path, str]], output_path: str | Path) -> pd.DataFrame:
    """Extract features for many recordings and save them as one CSV with a subject_id column."""
    raise NotImplementedError("See the TODO plan in this module's docstring.")
