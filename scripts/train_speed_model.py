#!/usr/bin/env python3
"""Fit the speed model on data/raw/dataset.csv and report held-out accuracy.

    python scripts/train_speed_model.py

Writes the fitted model to data/processed/speed_model.pkl and a metrics
summary to reports/speed_model_metrics.json. The split holds out whole days,
not random rows, so the reported error is not inflated by leakage.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.cleaning import add_time_bin, clean_speeds  # noqa: E402
from eta_router.config import DEFAULT_CONFIG  # noqa: E402
from eta_router.data import load_dataset  # noqa: E402
from eta_router.evaluation import (  # noqa: E402
    constant_speed_baseline,
    evaluate_speed_model,
    split_by_day,
)
from eta_router.speed_model import SpeedModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    output = args.output or (cfg.processed_dir / "speed_model.pkl")

    print("Loading dataset ...")
    raw = load_dataset(args.dataset, cfg)
    print(f"  {len(raw):,} rows, {raw['edge_id'].nunique():,} distinct edges")

    print("Cleaning ...")
    cleaned, report = clean_speeds(raw, cfg)
    print("  " + report.summary())
    cleaned = add_time_bin(cleaned, cfg)

    train, validation = split_by_day(cleaned, cfg)
    split_kind = "day-wise" if "day" in cleaned.columns else "random (no day column)"
    print(f"Split: {split_kind} | train={len(train):,} validation={len(validation):,}")

    print("Fitting hierarchical speed model ...")
    started = time.perf_counter()
    model = SpeedModel(config=cfg).fit(train)
    fit_seconds = time.perf_counter() - started
    print(f"  fitted in {fit_seconds:.2f}s | coverage: {model.coverage()}")

    model_metrics = evaluate_speed_model(model, validation)
    baseline = constant_speed_baseline(train, validation)
    print("\nHeld-out speed accuracy")
    print(f"  model:            {model_metrics.summary()}")
    print(f"  per-edge median:  {baseline.summary()}")
    if baseline.mae_kmh and baseline.mae_kmh > 0:
        gain = 100.0 * (1 - model_metrics.mae_kmh / baseline.mae_kmh)
        print(f"  MAE improvement over the naive baseline: {gain:.1f}%")

    print("\nRefitting on all data for the final model ...")
    final_model = SpeedModel(config=cfg).fit(cleaned)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(final_model, handle)
    print(f"  saved -> {output}")

    metrics_path = cfg.reports_dir / "speed_model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "cleaning": {
                    "rows_in": report.n_input,
                    "out_of_bounds": report.n_out_of_bounds,
                    "mad_outliers": report.n_mad_outliers,
                    "rows_kept": report.n_output,
                },
                "split": split_kind,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "fit_seconds": round(fit_seconds, 3),
                "model": model_metrics.__dict__,
                "baseline_per_edge_median": baseline.__dict__,
            },
            indent=2,
        )
    )
    print(f"  metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
