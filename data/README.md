# Data

`dream_features.csv` holds pre-extracted EEG features for **one overnight recording**,
one row per 30-second epoch, in time order.

| Property | Value |
| --- | --- |
| Rows (epochs) | 2,802 (~23.4 hours) |
| Features | 13 (band powers, Hjorth parameters, statistics, entropy) |
| Label column | `label` — Rechtschaffen & Kales stages |
| Subject / recording id | **none** |
| Scaling | already standardised (mean 0, sd 1) over the whole file |

Label counts: Sleep stage W 1,856 · 2 562 · R 170 · 1 109 · 3 96 · 4 9.

The labels match the Sleep-EDF (PhysioNet) hypnogram format. Rows 0-717 and
1701-2801 are wake before and after the night, which is why `WAKE_TRIM_MARGIN_EPOCHS`
exists.

## Using a different file

Keep the same 13 feature columns and `label` column, then either replace this file,
pass `python scripts/train.py --data path/to/file.csv`, or set the `SLEEPSTAGE_DATA`
environment variable.

## Known limitations

- No subject id, so true subject-wise evaluation is impossible (see `src/sleepstage/extract_features.py`).
- Features were standardised on the full file before splitting, which leaks a small
  amount of test-set information. This cannot be undone without the raw signals.
