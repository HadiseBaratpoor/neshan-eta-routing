#!/usr/bin/env python3
"""Generate a synthetic city into data/raw so the pipeline runs on a clean clone.

    python scripts/make_synthetic_data.py --rows 8 --cols 8 --days 14

Delete the generated files and drop the real challenge exports in the same
place to switch to real data; nothing else changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.config import DEFAULT_CONFIG  # noqa: E402
from eta_router.synthetic import write_synthetic_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=14, help="grid rows (default: 14)")
    parser.add_argument("--cols", type=int, default=14, help="grid columns (default: 14)")
    parser.add_argument("--days", type=int, default=14, help="days of speed history (default: 14)")
    parser.add_argument("--cases", type=int, default=300, help="test cases (default: 300)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = write_synthetic_data(
        config=DEFAULT_CONFIG,
        rows=args.rows,
        cols=args.cols,
        n_days=args.days,
        n_cases=args.cases,
        seed=args.seed,
    )

    print("Synthetic data written:")
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name:<11} {path}  ({size_kb:,.1f} KB)")
    print(
        "\nThe *_synthetic_truth.gpickle file holds the hidden generating function "
        "and is used only by scripts/evaluate.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
