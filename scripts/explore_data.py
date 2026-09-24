#!/usr/bin/env python3
"""Exploratory analysis: what the speed data actually looks like.

    python scripts/explore_data.py

Writes figures to reports/figures/ and prints the findings that drove the
modelling decisions. Run this before trusting any model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.cleaning import add_time_bin, clean_speeds  # noqa: E402
from eta_router.config import DEFAULT_CONFIG, WEATHER_LABELS  # noqa: E402
from eta_router.data import load_dataset, load_graph  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def main() -> int:
    cfg = DEFAULT_CONFIG
    figures = cfg.reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(None, cfg)
    graph = load_graph(None, cfg, trusted_pickle=True)

    print("=" * 66)
    print("1. SHAPE AND COVERAGE")
    print("=" * 66)
    print(f"  rows                : {len(raw):,}")
    print(f"  distinct edges      : {raw['edge_id'].nunique():,}")
    print(f"  edges in the graph  : {graph.number_of_edges():,}")
    print(f"  nodes in the graph  : {graph.number_of_nodes():,}")
    print(f"  minute range        : {raw['minute'].min()} - {raw['minute'].max()}")
    if "day" in raw.columns:
        print(f"  days                : {raw['day'].nunique()}")

    observed = set(raw["edge_id"].unique())
    graph_edges = {int(d["id"]) for _, _, d in graph.edges(data=True)}
    missing = graph_edges - observed
    print(f"  graph edges with NO speed history: {len(missing):,} ({100 * len(missing) / max(len(graph_edges), 1):.1f}%)")
    if missing:
        print("    -> these need a fallback; the model uses the city profile")

    cells = raw.groupby(["edge_id", "weather", "is_holiday"]).size()
    print(f"\n  observations per (edge, weather, holiday) cell:")
    print(f"    median {cells.median():.0f} | 10th pct {cells.quantile(0.1):.0f} | min {cells.min():.0f}")
    print("    -> sparse cells exist, so shrinkage towards a parent level is required")

    print("\n" + "=" * 66)
    print("2. OUTLIERS")
    print("=" * 66)
    print(f"  raw speed range     : {raw['speed'].min():.2f} - {raw['speed'].max():.2f} km/h")
    impossible = ((raw["speed"] < cfg.min_speed_kmh) | (raw["speed"] > cfg.max_speed_kmh)).sum()
    print(f"  physically impossible: {impossible:,} ({100 * impossible / len(raw):.2f}%)")
    cleaned, report = clean_speeds(raw, cfg)
    print(f"  after cleaning      : {report.summary()}")
    print(f"  clean speed range   : {cleaned['speed'].min():.2f} - {cleaned['speed'].max():.2f} km/h")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(raw["speed"], bins=120, color="#b03030")
    axes[0].set_title("Raw speeds (note the tail)")
    axes[0].set_xlabel("km/h")
    axes[1].hist(cleaned["speed"], bins=120, color="#2a7f62")
    axes[1].set_title("After cleaning")
    axes[1].set_xlabel("km/h")
    fig.tight_layout()
    fig.savefig(figures / "01_speed_distribution.png")
    plt.close(fig)

    print("\n" + "=" * 66)
    print("3. TIME OF DAY")
    print("=" * 66)
    binned = add_time_bin(cleaned, cfg)
    profile = binned.groupby("time_bin")["speed"].median()
    peak_bin = profile.idxmin()
    quiet_bin = profile.idxmax()
    print(f"  slowest time: {peak_bin * cfg.time_bin_minutes // 60:02d}:{peak_bin * cfg.time_bin_minutes % 60:02d} "
          f"({profile.min():.1f} km/h)")
    print(f"  fastest time: {quiet_bin * cfg.time_bin_minutes // 60:02d}:{quiet_bin * cfg.time_bin_minutes % 60:02d} "
          f"({profile.max():.1f} km/h)")
    print(f"  peak-to-quiet ratio: {profile.max() / profile.min():.2f}x")
    print("    -> a single average speed per edge would be badly wrong at rush hour")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    hours = profile.index.to_numpy() * cfg.time_bin_minutes / 60
    for holiday, label, style in ((0, "working day", "-"), (1, "holiday", "--")):
        subset = binned[binned["is_holiday"] == holiday]
        if subset.empty:
            continue
        series = subset.groupby("time_bin")["speed"].median()
        ax.plot(series.index * cfg.time_bin_minutes / 60, series.to_numpy(), style, label=label)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("median speed (km/h)")
    ax.set_title("Daily speed profile")
    ax.set_xticks(range(0, 25, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "02_daily_profile.png")
    plt.close(fig)

    print("\n" + "=" * 66)
    print("4. WEATHER AND HOLIDAYS")
    print("=" * 66)
    overall = cleaned["speed"].median()
    for weather, label in WEATHER_LABELS.items():
        subset = cleaned[cleaned["weather"] == weather]
        if subset.empty:
            print(f"  {label:<7}: not present in the data")
            continue
        median = subset["speed"].median()
        print(f"  {label:<7}: median {median:6.2f} km/h  ({100 * (median / overall - 1):+.1f}% vs overall, n={len(subset):,})")
    for holiday, label in ((0, "working day"), (1, "holiday")):
        subset = cleaned[cleaned["is_holiday"] == holiday]
        if not subset.empty:
            print(f"  {label:<12}: median {subset['speed'].median():6.2f} km/h (n={len(subset):,})")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    for weather, label in WEATHER_LABELS.items():
        subset = binned[binned["weather"] == weather]
        if subset.empty:
            continue
        series = subset.groupby("time_bin")["speed"].median()
        ax.plot(series.index * cfg.time_bin_minutes / 60, series.to_numpy(), label=label)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("median speed (km/h)")
    ax.set_title("Daily profile by weather")
    ax.set_xticks(range(0, 25, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "03_weather_profiles.png")
    plt.close(fig)

    print("\n" + "=" * 66)
    print("5. VARIATION BETWEEN EDGES")
    print("=" * 66)
    per_edge = cleaned.groupby("edge_id")["speed"].median()
    print(f"  per-edge median speed: {per_edge.min():.1f} - {per_edge.max():.1f} km/h")
    print(f"  interquartile range  : {per_edge.quantile(0.25):.1f} - {per_edge.quantile(0.75):.1f} km/h")
    print("    -> edges differ a lot, so a city-wide profile alone is not enough")

    print(f"\nFigures written to {figures}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
