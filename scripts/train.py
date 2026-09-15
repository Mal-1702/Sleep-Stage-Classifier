"""Run every experiment and save artifacts for the dashboard.

Usage:
    python scripts/train.py                 # full run with hyperparameter search
    python scripts/train.py --quick         # fast run: fewer models, no tuning
    python scripts/train.py --data path/to/features.csv --output artifacts/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleepstage import config  # noqa: E402
from sleepstage.experiments import run_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate sleep stage classifiers.")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH, help="Feature CSV (default: data/dream_features.csv)")
    parser.add_argument("--output", type=Path, default=config.ARTIFACTS_DIR, help="Artifacts directory (default: artifacts/)")
    parser.add_argument("--quick", action="store_true", help="Baselines + Random Forest only, no tuning")
    parser.add_argument("--no-tune", action="store_true", help="Skip the hyperparameter search")
    parser.add_argument("--save-all-models", action="store_true", help="Also save models for non-default resampling runs")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def print_summary(metrics: dict) -> None:
    rows = [
        r for r in metrics["experiments"]
        if r["class_setup"] == "5class" and (r["resampling"] in ("none", config.DEFAULT_RESAMPLING))
    ]
    header = f"{'split':<8} {'trim':<10} {'model':<24} {'resampling':<11} {'acc':>6} {'f1':>6} {'kappa':>6} {'bal_acc':>7}"
    print("\n5-class results on the held-out test set")
    print(header)
    print("-" * len(header))
    for r in rows:
        t = r["test"]
        kappa = f"{t['cohen_kappa']:.3f}" if t["cohen_kappa"] is not None else "n/a"
        print(
            f"{r['split_type']:<8} {r['trimming']:<10} {r['model']:<24} {r['resampling']:<11} "
            f"{t['accuracy']:>6.3f} {t['f1_macro']:>6.3f} {kappa:>6} {t['balanced_accuracy']:>7.3f}"
        )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()
    try:
        metrics = run_all(
            output_dir=args.output,
            data_path=args.data,
            quick=args.quick,
            tune=not args.no_tune,
            save_all_models=args.save_all_models,
        )
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1
    print_summary(metrics)
    print(f"\nDone in {time.perf_counter() - started:.0f}s. Artifacts in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
