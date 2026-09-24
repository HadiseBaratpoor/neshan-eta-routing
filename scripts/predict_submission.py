#!/usr/bin/env python3
"""Answer every row of test_cases.csv and write the submission file.

    python scripts/predict_submission.py

Reads data/raw/test_cases.csv, fills the ``eta`` and ``route`` columns with
time-dependent Dijkstra answers, validates the format, and writes
reports/test_cases_filled.csv.

Test cases are grouped by (weather, is_holiday) so the time-dependent graph is
built once per scenario instead of once per query.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.cleaning import add_time_bin, clean_speeds  # noqa: E402
from eta_router.config import DEFAULT_CONFIG  # noqa: E402
from eta_router.data import load_dataset, load_graph, load_test_cases  # noqa: E402
from eta_router.routing import NoRouteError, time_dependent_dijkstra  # noqa: E402
from eta_router.speed_model import SpeedModel  # noqa: E402
from eta_router.submission import (  # noqa: E402
    build_submission,
    validate_submission,
    write_submission,
)
from eta_router.travel_time import TimeDependentGraph  # noqa: E402


def load_or_fit_model(cfg, model_path: Path) -> SpeedModel:
    if model_path.exists():
        print(f"Loading fitted model from {model_path}")
        with model_path.open("rb") as handle:
            return pickle.load(handle)
    print("No saved model found; fitting from dataset.csv ...")
    raw = load_dataset(None, cfg)
    cleaned, report = clean_speeds(raw, cfg)
    print("  " + report.summary())
    return SpeedModel(config=cfg).fit(add_time_bin(cleaned, cfg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--trust-pickle", action="store_true",
        help="Acknowledge that graph/model pickle loading can execute arbitrary code; use only trusted local files.",
    )
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    model_path = args.model or (cfg.processed_dir / "speed_model.pkl")
    output = args.output or (cfg.reports_dir / "test_cases_filled.csv")

    if not args.trust_pickle:
        parser.error("graph/model pickle files can execute arbitrary code; inspect inputs and pass --trust-pickle to acknowledge")
    graph = load_graph(None, cfg, trusted_pickle=True)
    print(f"Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} directed edges")

    test_cases = load_test_cases(None, cfg)
    print(f"Test cases: {len(test_cases):,}")

    model = load_or_fit_model(cfg, model_path)

    etas: list[int] = [0] * len(test_cases)
    routes: list[list[int]] = [[] for _ in range(len(test_cases))]
    failures = 0

    started = time.perf_counter()
    grouped = test_cases.groupby(["weather", "is_holiday"], sort=False)
    for (weather, holiday), part in grouped:
        td = TimeDependentGraph(
            graph=graph,
            model=model,
            weather=int(weather),
            is_holiday=int(holiday),
            config=cfg,
        )
        for position, row in zip(test_cases.index.get_indexer(part.index), part.itertuples()):
            try:
                result = time_dependent_dijkstra(
                    td,
                    int(row.src),
                    int(row.dest),
                    float(row.route_start_t) * 60.0,
                )
                etas[position] = int(round(result.eta_seconds))
                routes[position] = [int(n) for n in result.path]
            except (NoRouteError, KeyError) as exc:
                failures += 1
                print(f"  ! row {position}: {exc}")
        print(f"  scenario weather={weather} holiday={holiday}: {len(part):,} cases answered")

    elapsed = time.perf_counter() - started
    print(f"\nAnswered {len(test_cases):,} cases in {elapsed:.2f}s ({elapsed / max(len(test_cases), 1) * 1000:.1f} ms/case)")
    if failures:
        print(f"  ERROR: {failures} unreachable pair(s); refusing to fabricate routes")
        return 1

    submission = build_submission(test_cases, etas, routes)
    problems = validate_submission(submission, expected_rows=len(test_cases), graph=graph)
    if problems:
        print("\nFORMAT PROBLEMS - do not submit:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    write_submission(submission, output)
    print(f"\nFormat validated. Submission written -> {output}")
    print(submission.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
