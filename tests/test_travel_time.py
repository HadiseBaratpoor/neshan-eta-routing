"""Unit conversion and edge travel-time correctness.

The brief warns that unit errors are the easiest way to lose points, so the
conversion is pinned by hand-checked cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.config import Config  # noqa: E402
from eta_router.speed_model import SpeedModel  # noqa: E402
from eta_router.travel_time import TimeDependentGraph, seconds_from_speed  # noqa: E402


def test_seconds_from_speed_known_values():
    # 1000 m at 36 km/h = 10 m/s -> 100 s
    assert seconds_from_speed(1000.0, 36.0) == pytest.approx(100.0)
    # 500 m at 60 km/h = 16.667 m/s -> 30 s
    assert seconds_from_speed(500.0, 60.0) == pytest.approx(30.0)
    # 3600 m at 3.6 km/h = 1 m/s -> 3600 s
    assert seconds_from_speed(3600.0, 3.6) == pytest.approx(3600.0)


def test_seconds_from_speed_rejects_zero():
    """A zero speed must not produce inf or a division error."""

    value = seconds_from_speed(100.0, 0.0)
    assert np.isfinite(value)
    assert value > 0


def test_travel_time_is_positive_everywhere(fitted_graph):
    """Dijkstra requires non-negative weights; the brief calls this out."""

    td = fitted_graph
    assert (td._travel_seconds > 0).all()
    assert np.isfinite(td._travel_seconds).all()


def test_fifo_property_holds(fitted_graph):
    """Leaving later must never mean arriving earlier."""

    td = fitted_graph
    bin_seconds = td.config.time_bin_minutes * 60
    departures = np.arange(td.config.n_time_bins) * bin_seconds
    arrivals = departures[None, :] + td._travel_seconds
    differences = np.diff(arrivals, axis=1)
    assert (differences >= -1e-6).all(), "FIFO violated: a later departure arrives earlier"
    # The actual lookup is piecewise constant, so also check exact transitions
    # including the midnight wrap, not only arrivals at sampled bin starts.
    for edge_idx in range(len(td.edges)):
        for boundary in range(td.config.n_time_bins):
            before = (boundary * bin_seconds - 1e-6) % (24 * 3600)
            after = boundary * bin_seconds
            arr_before = before + td.edge_travel_time_by_index(edge_idx, before)
            arr_after = after + td.edge_travel_time_by_index(edge_idx, after)
            if boundary == 0:
                arr_after += 24 * 3600
            assert arr_after >= arr_before - 1e-6
    assert (td._travel_seconds[:, -1] + bin_seconds >= td._travel_seconds[:, 0] - 1e-6).all()


def test_path_travel_time_reprices_each_edge(fitted_graph):
    """A two-edge path must not be priced entirely at the departure time."""

    td = fitted_graph
    path = [0, 1, 2]
    start = 7 * 3600
    total = td.path_travel_time(path, start)
    first = td.edge_travel_time(0, 1, start)
    second_at_start = td.edge_travel_time(1, 2, start)
    second_on_arrival = td.edge_travel_time(1, 2, start + first)
    assert total == pytest.approx(first + second_on_arrival)
    if second_at_start != second_on_arrival:
        assert total != pytest.approx(first + second_at_start)


@pytest.fixture
def fitted_graph():
    import pandas as pd

    from eta_router.cleaning import add_time_bin

    config = Config(time_bin_minutes=10, smoothing_sigma_bins=1.0)
    graph = nx.DiGraph()
    for i in range(3):
        graph.add_edge(i, i + 1, id=i, length=1000.0)
        graph.add_edge(i + 1, i, id=10 + i, length=1000.0)

    rows = []
    for edge_id in list(range(3)) + list(range(10, 13)):
        for minute in range(0, 1440, 10):
            # A clear rush-hour dip so time dependence is measurable.
            speed = 50.0 - 30.0 * np.exp(-0.5 * ((minute - 480) / 60.0) ** 2)
            rows.append(
                {
                    "edge_id": edge_id,
                    "length": 1000.0,
                    "minute": minute,
                    "speed": speed,
                    "is_holiday": 0,
                    "weather": 0,
                }
            )
    frame = add_time_bin(pd.DataFrame(rows), config)
    model = SpeedModel(config=config).fit(frame)
    return TimeDependentGraph(graph, model, weather=0, is_holiday=0, config=config)
