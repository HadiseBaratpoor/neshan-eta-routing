"""A synthetic city that matches the challenge's data contract exactly.

The real ``network.gpickle`` / ``dataset.csv`` / ``test_cases.csv`` are not
public, so the repository ships a generator. It produces files with the same
names, columns, dtypes and semantics, which means:

* the whole pipeline runs end to end on a clean clone;
* the tests have something deterministic to assert against;
* swapping in the real files is a copy into ``data/raw`` and nothing else.

The generator is not a toy. It reproduces the structure that makes the problem
interesting:

* a grid city with arterials (fast, long) and side streets (slow, short);
* a bimodal rush-hour profile, with the morning peak biting hardest on
  arterials heading into the centre;
* weather multipliers (rain ~0.85x, snow ~0.65x) and a flatter holiday
  profile with no sharp peaks;
* log-normal noise, ~3% missing cells, and ~1% injected outliers
  (near-zero crawl speeds and impossible 150+ km/h readings),

so the cleaning and imputation code has something real to do. A hidden
ground truth is written alongside, letting ``evaluate.py`` score ETAs
honestly instead of only scoring speeds.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG, MINUTES_PER_DAY

WEATHER_MULTIPLIER = {0: 1.00, 1: 0.85, 2: 0.65}


def _rush_hour_factor(
    minute: np.ndarray,
    is_holiday: int,
    arterial: np.ndarray,
    peak_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Speed multiplier in [0.22, 1.05] from time of day.

    Working days get a sharp morning and a broader evening peak; holidays get
    a single shallow midday dip. Arterials are hit harder because that is
    where commuters queue.

    ``peak_offset`` shifts an edge's peak in minutes. This is what makes the
    problem genuinely *time-dependent* rather than merely time-varying: real
    congestion sweeps across a city, inbound corridors jamming before the
    ring roads do. A traveller who leaves at 07:30 can cross one district
    before it peaks and reach the next after it has cleared - which only a
    router that prices edges at their true arrival time can exploit.
    """

    m = minute.astype("float64")
    if is_holiday:
        dip = 0.85 + 0.15 * np.cos(2 * np.pi * (m - 840) / MINUTES_PER_DAY)
        return np.clip(dip, 0.6, 1.0)

    shift = 0.0 if peak_offset is None else peak_offset
    morning = np.exp(-0.5 * ((m - (480 + shift)) / 30.0) ** 2)   # ~08:00
    evening = np.exp(-0.5 * ((m - (1080 + shift)) / 50.0) ** 2)  # ~18:00
    night = np.exp(-0.5 * ((m - 180) / 120.0) ** 2)              # 03:00, free flowing

    # ``arterial`` doubles as the per-edge congestion severity when a float
    # array is passed (see build_grid_graph). Downtown edges saturate at the
    # peak while the ring stays usable, so detouring around the core is a
    # real decision rather than a tie-break between equivalent paths.
    severity = np.asarray(arterial, dtype="float64")
    if severity.dtype == bool or set(np.unique(severity)) <= {0.0, 1.0}:
        severity = np.where(severity > 0, 0.82, 0.30)
    congestion = severity * (0.95 * morning + 0.80 * evening)
    factor = 1.0 - congestion + 0.08 * night
    return np.clip(factor, 0.12, 1.05)


def build_grid_graph(rows: int = 14, cols: int = 14, seed: int = 42) -> nx.DiGraph:
    """A directed grid city with integer node ids and per-edge ``id``/``length``.

    Congestion sweeps across the city from west to east: an edge's peak is
    offset by up to +-45 minutes according to its column. That spatial
    gradient is what a time-dependent router can exploit and a static one
    cannot see.
    """

    rng = np.random.default_rng(seed)
    grid = nx.grid_2d_graph(rows, cols)
    mapping = {node: idx for idx, node in enumerate(sorted(grid.nodes()))}
    grid = nx.relabel_nodes(grid, mapping)

    directed = nx.DiGraph()
    directed.add_nodes_from(grid.nodes())

    edge_id = 0
    for u, v in sorted(grid.edges()):
        ru, cu = divmod(u, cols)
        rv, cv = divmod(v, cols)
        # Every third row/column is an arterial: longer blocks, faster limit.
        arterial = (ru == rv and ru % 3 == 0) or (cu == cv and cu % 3 == 0)
        base_length = 2100.0 if arterial else 950.0
        length = float(np.round(base_length * rng.uniform(0.8, 1.25), 1))
        free_flow = 62.0 if arterial else 38.0
        free_flow = float(np.round(free_flow * rng.uniform(0.9, 1.1), 1))

        # Distance from the city centre, normalised to roughly [0, 1].
        centre_r, centre_c = (rows - 1) / 2.0, (cols - 1) / 2.0
        mean_r, mean_c = (ru + rv) / 2.0, (cu + cv) / 2.0
        radius = np.hypot(mean_r - centre_r, mean_c - centre_c)
        max_radius = np.hypot(centre_r, centre_c)
        centrality = 1.0 - radius / max(max_radius, 1e-6)

        # Downtown chokes at the peak (severity ~0.93), the outer ring barely
        # slows (~0.22). This gradient is what makes a detour worth taking.
        severity = float(np.clip(0.22 + 0.71 * centrality**1.5, 0.15, 0.95))
        if arterial:
            severity = float(min(severity + 0.06, 0.95))

        # The congestion wave crosses the city; mean column sets the phase.
        peak_offset = float(np.round((mean_c / max(cols - 1, 1) - 0.5) * 210.0, 1))
        for a, b in ((u, v), (v, u)):
            directed.add_edge(
                a,
                b,
                id=edge_id,
                length=length,
                _arterial=severity,
                _free_flow_kmh=free_flow,
                _peak_offset=peak_offset,
            )
            edge_id += 1
    return directed


def true_speed(
    free_flow: np.ndarray,
    arterial: np.ndarray,
    minute: np.ndarray,
    weather: int,
    is_holiday: int,
    peak_offset: np.ndarray | None = None,
) -> np.ndarray:
    """The hidden generating function - never exposed to the model."""

    factor = _rush_hour_factor(minute, is_holiday, arterial, peak_offset)
    return free_flow * factor * WEATHER_MULTIPLIER[int(weather)]


def generate_dataset(
    graph: nx.DiGraph,
    n_days: int = 14,
    sample_minutes: int = 10,
    seed: int = 42,
    missing_rate: float = 0.03,
    outlier_rate: float = 0.01,
) -> pd.DataFrame:
    """Aggregated user speeds with realistic noise, gaps and outliers."""

    rng = np.random.default_rng(seed)
    minutes = np.arange(0, MINUTES_PER_DAY, sample_minutes)

    edge_ids, lengths, free_flows, arterials, offsets = [], [], [], [], []
    for _, _, data in graph.edges(data=True):
        edge_ids.append(int(data["id"]))
        lengths.append(float(data["length"]))
        free_flows.append(float(data["_free_flow_kmh"]))
        arterials.append(float(data["_arterial"]))
        offsets.append(float(data["_peak_offset"]))
    edge_ids = np.array(edge_ids)
    lengths = np.array(lengths)
    free_flows = np.array(free_flows)
    arterials = np.array(arterials, dtype="float64")
    offsets = np.array(offsets)

    frames = []
    for day in range(n_days):
        is_holiday = int(day % 7 in (5, 6))
        weather = int(rng.choice([0, 1, 2], p=[0.7, 0.22, 0.08]))
        n_edges = len(edge_ids)

        minute_grid = np.repeat(minutes, n_edges)
        edge_grid = np.tile(edge_ids, len(minutes))
        length_grid = np.tile(lengths, len(minutes))
        ff_grid = np.tile(free_flows, len(minutes))
        art_grid = np.tile(arterials, len(minutes))
        off_grid = np.tile(offsets, len(minutes))

        clean = true_speed(ff_grid, art_grid, minute_grid, weather, is_holiday, off_grid)
        # Multiplicative log-normal noise: aggregated speeds are right-skewed.
        noisy = clean * rng.lognormal(mean=0.0, sigma=0.11, size=clean.shape)

        n = noisy.size
        n_out = int(outlier_rate * n)
        if n_out:
            idx = rng.choice(n, size=n_out, replace=False)
            half = n_out // 2
            noisy[idx[:half]] = rng.uniform(0.2, 2.0, size=half)          # stuck probes
            noisy[idx[half:]] = rng.uniform(140.0, 260.0, size=n_out - half)  # GPS jumps

        keep = rng.random(n) >= missing_rate
        frames.append(
            pd.DataFrame(
                {
                    "day": day,
                    "edge_id": edge_grid[keep],
                    "length": length_grid[keep],
                    "minute": minute_grid[keep],
                    "speed": np.round(noisy[keep], 2),
                    "is_holiday": is_holiday,
                    "weather": weather,
                }
            )
        )

    dataset = pd.concat(frames, ignore_index=True)
    return dataset.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_test_cases(
    graph: nx.DiGraph,
    n_cases: int = 300,
    seed: int = 7,
    min_hops: int = 14,
    rush_hour_fraction: float = 0.70,
) -> pd.DataFrame:
    """Source/destination pairs far enough apart for traffic to evolve en route.

    Short hops are uninteresting: the traveller arrives before conditions
    change. ``min_hops`` enforces a real journey, and a majority of departures
    are placed in or just before a rush-hour window, which is where routing
    decisions actually matter.
    """

    rng = np.random.default_rng(seed)
    nodes = np.array(sorted(graph.nodes()))
    rows = []
    while len(rows) < n_cases:
        src, dest = rng.choice(nodes, size=2, replace=False)
        try:
            if nx.shortest_path_length(graph, int(src), int(dest)) < min_hops:
                continue
        except nx.NetworkXNoPath:
            continue
        if rng.random() < rush_hour_fraction:
            # Departures clustered around the morning and evening builds.
            centre = 420 if rng.random() < 0.5 else 1020
            start = int(np.clip(rng.normal(centre, 60), 0, MINUTES_PER_DAY - 1))
        else:
            start = int(rng.integers(0, MINUTES_PER_DAY))
        rows.append(
            {
                "src": int(src),
                "dest": int(dest),
                "route_start_t": start,
                "is_holiday": int(rng.random() < 0.25),
                "weather": int(rng.choice([0, 1, 2], p=[0.7, 0.22, 0.08])),
            }
        )
    frame = pd.DataFrame(rows)
    frame["eta"] = ""
    frame["route"] = ""
    return frame


def write_synthetic_data(
    config: Config = DEFAULT_CONFIG,
    rows: int = 14,
    cols: int = 14,
    n_days: int = 14,
    n_cases: int = 300,
    seed: int = 42,
) -> dict[str, Path]:
    """Write graph, dataset and test cases into ``data/raw``."""

    raw = Path(config.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    graph = build_grid_graph(rows=rows, cols=cols, seed=seed)
    dataset = generate_dataset(graph, n_days=n_days, seed=seed)
    test_cases = generate_test_cases(graph, n_cases=n_cases, seed=seed + 5)

    # The published graph must not leak the generating parameters.
    public = nx.DiGraph()
    public.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        public.add_edge(u, v, id=int(data["id"]), length=float(data["length"]))

    graph_path = raw / config.graph_file
    with graph_path.open("wb") as handle:
        pickle.dump(public, handle)

    dataset_path = raw / config.dataset_file
    dataset.to_csv(dataset_path, index=False)

    test_path = raw / config.test_cases_file
    test_cases.to_csv(test_path, index=False)

    truth_path = raw / "_synthetic_truth.gpickle"
    with truth_path.open("wb") as handle:
        pickle.dump(graph, handle)

    return {
        "graph": graph_path,
        "dataset": dataset_path,
        "test_cases": test_path,
        "truth": truth_path,
    }
