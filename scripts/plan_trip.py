#!/usr/bin/env python3
"""Ask the system a single question from the command line.

Plan one trip:

    python scripts/plan_trip.py --src 12 --dest 180 --depart 480 --weather 1

Find the best time to leave in a window - the question the brief ends on:

    python scripts/plan_trip.py --src 12 --dest 180 --window 360 600

Work backwards from a deadline:

    python scripts/plan_trip.py --src 12 --dest 180 --arrive-by 540
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.cleaning import add_time_bin, clean_speeds  # noqa: E402
from eta_router.config import DEFAULT_CONFIG  # noqa: E402
from eta_router.data import load_dataset, load_graph  # noqa: E402
from eta_router.departure import (  # noqa: E402
    departure_curve,
    latest_departure_for_arrival,
)
from eta_router.routing import NoRouteError, static_route, time_dependent_dijkstra  # noqa: E402
from eta_router.speed_model import SpeedModel  # noqa: E402
from eta_router.travel_time import TimeDependentGraph  # noqa: E402

WEATHER_NAMES = {0: "sunny", 1: "rainy", 2: "snowy"}


def hhmm(minute: float) -> str:
    minute = int(minute) % 1440
    return f"{minute // 60:02d}:{minute % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=int, required=True, help="source node id")
    parser.add_argument("--dest", type=int, required=True, help="destination node id")
    parser.add_argument("--depart", type=int, default=480, help="departure, minutes after midnight (default 480 = 08:00)")
    parser.add_argument("--weather", type=int, default=0, choices=[0, 1, 2], help="0 sunny, 1 rainy, 2 snowy")
    parser.add_argument("--holiday", type=int, default=0, choices=[0, 1])
    parser.add_argument("--window", type=int, nargs=2, metavar=("START", "END"), help="sweep departures in this window")
    parser.add_argument("--arrive-by", type=int, default=None, help="latest departure that arrives by this minute")
    parser.add_argument("--step", type=int, default=5, help="window step in minutes (default 5)")
    parser.add_argument("--compare-static", action="store_true", help="also show the frozen-weight baseline")
    parser.add_argument("--trust-pickle", action="store_true", help="Acknowledge executable pickle risk for local graph/model files")
    args = parser.parse_args()
    if not args.trust_pickle:
        parser.error("pickle files can execute arbitrary code; inspect them and pass --trust-pickle to acknowledge")

    cfg = DEFAULT_CONFIG
    graph = load_graph(None, cfg, trusted_pickle=True)

    model_path = cfg.processed_dir / "speed_model.pkl"
    if model_path.exists():
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
    else:
        print("No saved model; fitting from dataset.csv (run train_speed_model.py to cache it) ...")
        cleaned, _ = clean_speeds(load_dataset(None, cfg), cfg)
        model = SpeedModel(config=cfg).fit(add_time_bin(cleaned, cfg))

    td = TimeDependentGraph(graph, model, args.weather, args.holiday, cfg)
    conditions = f"{WEATHER_NAMES[args.weather]}, {'holiday' if args.holiday else 'working day'}"
    print(f"\n{args.src} -> {args.dest}  ({conditions})")
    print("-" * 64)

    try:
        if args.arrive_by is not None:
            option = latest_departure_for_arrival(td, args.src, args.dest, args.arrive_by, step_minutes=args.step)
            if option is None:
                print(f"Cannot arrive by {hhmm(args.arrive_by)} even leaving 3 hours early.")
                return 1
            print(f"To arrive by {hhmm(args.arrive_by)}, leave at {hhmm(option.depart_minute)}")
            print(f"  travel time : {option.travel_minutes:.1f} min ({option.travel_seconds:.0f} s)")
            print(f"  arrival     : {hhmm(option.arrive_minute)}")
            print(f"  route       : {option.path}")
            return 0

        if args.window:
            curve = departure_curve(td, args.src, args.dest, args.window[0], args.window[1], args.step)
            print(f"Departure sweep {hhmm(args.window[0])} - {hhmm(args.window[1])} every {args.step} min")
            print(f"  {curve.summary()}")
            print("\n  depart   travel   arrive")
            worst = curve.worst.travel_seconds
            for option in curve.options:
                bar = "#" * int(round(28 * option.travel_seconds / max(worst, 1e-6)))
                marker = " <- best" if option is curve.best else ""
                print(f"  {hhmm(option.depart_minute)}   {option.travel_minutes:6.1f}m  {hhmm(option.arrive_minute)}  {bar}{marker}")
            print(f"\n  best route: {curve.best.path}")
            return 0

        result = time_dependent_dijkstra(td, args.src, args.dest, float(args.depart) * 60.0)
        print(f"Leaving at {hhmm(args.depart)}")
        print(f"  ETA        : {result.eta_seconds:.0f} s ({result.eta_seconds / 60:.1f} min)")
        print(f"  arrival    : {hhmm(args.depart + result.eta_seconds / 60)}")
        print(f"  route      : {result.path}")
        print(f"  nodes seen : {result.expanded_nodes}")

        if args.compare_static:
            baseline = static_route(td, args.src, args.dest, float(args.depart) * 60.0)
            delta = baseline.eta_seconds - result.eta_seconds
            print("\n  static-Dijkstra baseline (weights frozen at departure)")
            print(f"    true travel time of its route: {baseline.eta_seconds:.0f} s")
            print(f"    same route as time-dependent : {baseline.path == result.path}")
            print(f"    time lost by freezing weights: {delta:.0f} s")
        return 0

    except (NoRouteError, KeyError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
