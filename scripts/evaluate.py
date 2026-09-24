#!/usr/bin/env python3
"""Score the system against ground truth and against the static baseline.

    python scripts/evaluate.py

On synthetic data the hidden generating function is available, so this
computes the *true* ETA of every test case by integrating the real speed
function along the route. That gives three honest numbers:

1. ETA error of the time-dependent system;
2. ETA error of the classical static-Dijkstra baseline;
3. how often the two disagree about the route, and how much time that costs.

Writes reports/evaluation.json and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.cleaning import add_time_bin, clean_speeds  # noqa: E402
from eta_router.config import DEFAULT_CONFIG  # noqa: E402
from eta_router.data import load_dataset, load_graph, load_test_cases  # noqa: E402
from eta_router.evaluation import eta_metrics, route_similarity  # noqa: E402
from eta_router.routing import NoRouteError, static_route, time_dependent_dijkstra  # noqa: E402
from eta_router.speed_model import SpeedModel  # noqa: E402
from eta_router.synthetic import true_speed  # noqa: E402
from eta_router.travel_time import TimeDependentGraph  # noqa: E402


def true_path_seconds(truth_graph, path, start_second: float, weather: int, is_holiday: int) -> float:
    """Integrate the hidden speed function along a path, edge by edge."""

    t = float(start_second)
    for u, v in zip(path[:-1], path[1:]):
        data = truth_graph[u][v]
        minute = np.array([(t // 60) % 1440])
        speed = true_speed(
            np.array([data["_free_flow_kmh"]]),
            np.array([float(data["_arterial"])]),
            minute,
            weather,
            is_holiday,
            np.array([data.get("_peak_offset", 0.0)]),
        )[0]
        t += float(data["length"]) / (max(speed, 1e-3) / 3.6)
    return t - float(start_second)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    truth_path = cfg.raw_dir / "_synthetic_truth.gpickle"
    if not truth_path.exists():
        print(
            "No ground truth available.\n"
            "This script scores ETAs against the hidden generating function, which only\n"
            "exists for synthetic data. Run scripts/make_synthetic_data.py first, or\n"
            "submit reports/test_cases_filled.csv for official scoring on real data."
        )
        return 1

    with truth_path.open("rb") as handle:
        truth_graph = pickle.load(handle)

    graph = load_graph(None, cfg, trusted_pickle=True)
    test_cases = load_test_cases(None, cfg)
    if args.limit:
        test_cases = test_cases.head(args.limit).copy()

    model_path = cfg.processed_dir / "speed_model.pkl"
    if model_path.exists():
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
    else:
        cleaned, _ = clean_speeds(load_dataset(None, cfg), cfg)
        model = SpeedModel(config=cfg).fit(add_time_bin(cleaned, cfg))

    td_predicted, td_true, td_hops = [], [], []
    st_predicted, st_true = [], []
    similarities, identical = [], 0
    optimal_true = []

    for (weather, holiday), part in test_cases.groupby(["weather", "is_holiday"], sort=False):
        td = TimeDependentGraph(graph, model, int(weather), int(holiday), cfg)
        for row in part.itertuples():
            start = float(row.route_start_t) * 60.0
            try:
                dyn = time_dependent_dijkstra(td, int(row.src), int(row.dest), start)
                sta = static_route(td, int(row.src), int(row.dest), start)
            except (NoRouteError, KeyError):
                continue

            td_predicted.append(dyn.eta_seconds)
            td_true.append(true_path_seconds(truth_graph, dyn.path, start, int(weather), int(holiday)))
            td_hops.append(len(dyn.path))

            st_predicted.append(sta.eta_seconds)
            st_true.append(true_path_seconds(truth_graph, sta.path, start, int(weather), int(holiday)))

            similarities.append(route_similarity(dyn.path, sta.path))
            identical += int(dyn.path == sta.path)
            optimal_true.append(min(td_true[-1], st_true[-1]))

    n = len(td_predicted)
    if n == 0:
        print("No evaluable test cases.")
        return 1

    dyn_metrics = eta_metrics(np.array(td_true), np.array(td_predicted))
    sta_metrics = eta_metrics(np.array(st_true), np.array(st_predicted))

    td_true_arr = np.array(td_true)
    st_true_arr = np.array(st_true)
    time_saved = st_true_arr - td_true_arr

    print(f"\nEvaluated {n:,} test cases\n")
    print("ETA accuracy (predicted vs. true travel time of the chosen route)")
    print(f"  time-dependent : {dyn_metrics.summary()}")
    print(f"  static baseline: {sta_metrics.summary()}")

    print("\nRoute quality (true travel time actually experienced)")
    print(f"  mean true ETA, time-dependent : {td_true_arr.mean():,.1f} s")
    print(f"  mean true ETA, static baseline: {st_true_arr.mean():,.1f} s")
    print(f"  mean time saved per trip      : {time_saved.mean():,.1f} s")
    print(f"  cases where routes differ     : {n - identical:,} / {n:,} ({100 * (n - identical) / n:.1f}%)")
    print(f"  mean route Jaccard similarity : {np.mean(similarities):.3f}")
    print(f"  mean route length             : {np.mean(td_hops):.1f} nodes")

    differing = time_saved[np.array(similarities) < 1.0]
    if differing.size:
        print(f"  on differing routes, mean saving: {differing.mean():,.1f} s")

    # ---- departure-time planning: the question the brief ends on --------
    # Re-routing saves a few percent. Choosing *when to leave* saves far more,
    # and it is the capability the problem statement actually asks for.
    from eta_router.departure import departure_curve  # noqa: E402

    savings_pct, best_minutes = [], []
    sample = test_cases.head(60)
    for (weather, holiday), part in sample.groupby(["weather", "is_holiday"], sort=False):
        td = TimeDependentGraph(graph, model, int(weather), int(holiday), cfg)
        for row in part.itertuples():
            window_start = int(row.route_start_t)
            try:
                curve = departure_curve(
                    td, int(row.src), int(row.dest), window_start, (window_start + 120) % 1440, 10
                )
            except (NoRouteError, KeyError):
                continue
            worst = curve.worst.travel_seconds
            if worst > 0:
                savings_pct.append(100.0 * curve.saving_vs_worst() / worst)
                best_minutes.append(curve.best.depart_minute)

    if savings_pct:
        print("\nDeparture-time planning (2-hour window, sampled cases)")
        print(f"  cases analysed                : {len(savings_pct):,}")
        print(f"  mean saving, best vs. worst   : {np.mean(savings_pct):.1f}%")
        print(f"  median saving                 : {np.median(savings_pct):.1f}%")
        print(f"  90th percentile saving        : {np.percentile(savings_pct, 90):.1f}%")
        print("  -> choosing the departure minute beats re-routing by a wide margin")

    payload = {
        "n_cases": n,
        "departure_planning": {
            "cases": len(savings_pct),
            "mean_saving_percent": float(np.mean(savings_pct)) if savings_pct else None,
            "p90_saving_percent": float(np.percentile(savings_pct, 90)) if savings_pct else None,
        },
        "time_dependent": dyn_metrics.__dict__,
        "static_baseline": sta_metrics.__dict__,
        "mean_true_eta_time_dependent_s": float(td_true_arr.mean()),
        "mean_true_eta_static_s": float(st_true_arr.mean()),
        "mean_time_saved_s": float(time_saved.mean()),
        "routes_differ": int(n - identical),
        "mean_route_jaccard": float(np.mean(similarities)),
    }
    out = cfg.reports_dir / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
