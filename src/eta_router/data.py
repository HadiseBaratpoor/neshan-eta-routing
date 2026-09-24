"""Loading the road graph, the speed observations and the test cases.

The challenge ships three files in ``data/raw``:

``network.gpickle``
    A ``networkx`` graph. Edges carry ``id`` and ``length`` (metres); nodes
    carry an identifier.
``dataset.csv``
    Aggregated user speeds with columns
    ``edge_id, length, minute, speed, is_holiday, weather`` (and, when the
    organisers include it, a day column).
``test_cases.csv``
    ``src, dest, route_start_t, is_holiday, weather`` plus the two columns we
    must fill in: ``eta`` and ``route``.

Every loader here is defensive: real challenge exports vary in column order,
in dtype, and in whether the graph is directed. Nothing downstream should
have to care.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG

REQUIRED_DATASET_COLUMNS = ("edge_id", "length", "minute", "speed", "is_holiday", "weather")
REQUIRED_TEST_COLUMNS = ("src", "dest", "route_start_t", "is_holiday", "weather")
# Optional column names that different exports use for "which day is this".
DAY_COLUMN_CANDIDATES = ("day", "day_id", "date", "day_of_week", "dow")


class DataError(RuntimeError):
    """Raised when an input file is missing or structurally unusable."""


def load_graph(
    path: Path | str | None = None, config: Config = DEFAULT_CONFIG, *, trusted_pickle: bool = False
) -> nx.DiGraph:
    """Load trusted pickle road network as a directed graph.

    Pickle is executable; only load a file from a trusted local source.
    ``networkx`` dropped ``read_gpickle`` in 3.0, so the file is unpickled
    directly. An undirected input is expanded into both directions, which is
    what a road network means for routing purposes.
    """

    path = Path(path) if path is not None else config.graph_path
    if not path.exists():
        raise DataError(
            f"Road graph not found at {path}. Place network.gpickle in data/raw/ "
            "or generate a synthetic network with scripts/make_synthetic_data.py."
        )

    # Pickle can execute arbitrary code while loading. The challenge graph is
    # distributed in this legacy format; callers must explicitly trust it.
    if not trusted_pickle:
        raise DataError(
            "Refusing to load a pickle graph without trusted_pickle=True. "
            "Pickle can execute arbitrary code; only trust a local file from a known source."
        )
    with path.open("rb") as handle:
        graph = pickle.load(handle)

    if not isinstance(graph, nx.Graph):
        raise DataError(f"{path} did not contain a networkx graph (got {type(graph)!r}).")

    _validate_graph(graph)
    if graph.is_directed():
        directed = nx.DiGraph(graph)
    else:
        directed = nx.DiGraph()
        directed.add_nodes_from(graph.nodes(data=True))
        for u, v, data in graph.edges(data=True):
            directed.add_edge(u, v, **dict(data))
            directed.add_edge(v, u, **dict(data))

    _validate_graph(directed)
    return directed


def _validate_graph(graph: nx.DiGraph) -> None:
    missing_length = [(u, v) for u, v, d in graph.edges(data=True) if "length" not in d]
    if missing_length:
        raise DataError(
            f"{len(missing_length)} edges have no 'length' attribute, e.g. {missing_length[:3]}."
        )
    invalid_length = [
        (u, v, d.get("length"))
        for u, v, d in graph.edges(data=True)
        if "length" in d and (not np.isfinite(float(d["length"])) or float(d["length"]) <= 0)
    ]
    if invalid_length:
        raise DataError(f"edges must have finite positive lengths, e.g. {invalid_length[:3]}.")
    missing_id = [(u, v) for u, v, d in graph.edges(data=True) if "id" not in d]
    if missing_id:
        raise DataError(
            f"{len(missing_id)} edges have no 'id' attribute, e.g. {missing_id[:3]}."
        )
    ids = [int(d["id"]) for _, _, d in graph.edges(data=True)]
    duplicate_ids = sorted({edge_id for edge_id in ids if ids.count(edge_id) > 1})
    if duplicate_ids:
        raise DataError(f"edge IDs must be unique; duplicate IDs include {duplicate_ids[:3]}.")


def load_dataset(path: Path | str | None = None, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load the aggregated speed observations with normalised dtypes."""

    path = Path(path) if path is not None else config.dataset_path
    if not path.exists():
        raise DataError(
            f"Speed dataset not found at {path}. Place dataset.csv in data/raw/ "
            "or generate a synthetic one with scripts/make_synthetic_data.py."
        )

    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    _require_columns(frame, REQUIRED_DATASET_COLUMNS, path)

    frame["edge_id"] = frame["edge_id"].astype("int64")
    frame["minute"] = frame["minute"].astype("int64")
    frame["length"] = frame["length"].astype("float64")
    frame["speed"] = frame["speed"].astype("float64")
    frame["is_holiday"] = frame["is_holiday"].astype("int8")
    frame["weather"] = frame["weather"].astype("int8")

    day_col = next((c for c in DAY_COLUMN_CANDIDATES if c in frame.columns), None)
    if day_col is not None:
        frame = frame.rename(columns={day_col: "day"})
    return frame


def load_test_cases(path: Path | str | None = None, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load ``test_cases.csv`` preserving its original column order."""

    path = Path(path) if path is not None else config.test_cases_path
    if not path.exists():
        raise DataError(
            f"Test cases not found at {path}. Place test_cases.csv in data/raw/ "
            "or generate a synthetic one with scripts/make_synthetic_data.py."
        )

    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    _require_columns(frame, REQUIRED_TEST_COLUMNS, path)
    for column in ("src", "dest", "route_start_t", "is_holiday", "weather"):
        frame[column] = frame[column].astype("int64")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataError(f"{path} is missing required column(s): {missing}. Found {list(frame.columns)}.")


def edge_lengths_from_graph(graph: nx.DiGraph) -> dict[int, float]:
    """Map ``edge_id`` to length in metres, taken from the graph itself."""

    lengths: dict[int, float] = {}
    for _, _, data in graph.edges(data=True):
        lengths[int(data["id"])] = float(data["length"])
    return lengths
