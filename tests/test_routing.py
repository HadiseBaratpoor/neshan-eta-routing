"""Routing correctness: optimality, endpoints, and time dependence."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.cleaning import add_time_bin  # noqa: E402
from eta_router.config import Config  # noqa: E402
from eta_router.routing import NoRouteError, static_route, time_dependent_dijkstra  # noqa: E402
from eta_router.speed_model import SpeedModel  # noqa: E402
from eta_router.travel_time import TimeDependentGraph  # noqa: E402

# Mirror the production resolution: a coarse test config would smooth away
# exactly the rush-hour structure these tests are meant to verify.
CONFIG = Config(time_bin_minutes=10, smoothing_sigma_bins=1.0)


def _diamond_graph():
    """Two routes from 0 to 3: a short one through 1, a long one through 2."""

    graph = nx.DiGraph()
    graph.add_edge(0, 1, id=0, length=1000.0)
    graph.add_edge(1, 3, id=1, length=1000.0)
    graph.add_edge(0, 2, id=2, length=1500.0)
    graph.add_edge(2, 3, id=3, length=1500.0)
    return graph


def _model(speed_by_edge_and_minute):
    rows = []
    for edge_id, speed_fn in speed_by_edge_and_minute.items():
        # Several observations per cell, as a real aggregated export has -
        # shrinkage is adaptive, so one sample per cell would be treated as
        # untrustworthy and pulled towards the city average.
        for minute in range(0, 1440, 5):
            for _ in range(6):
                rows.append(
                    {
                        "edge_id": edge_id,
                        "length": 1000.0,
                        "minute": minute,
                        "speed": speed_fn(minute),
                        "is_holiday": 0,
                        "weather": 0,
                    }
                )
    frame = add_time_bin(pd.DataFrame(rows), CONFIG)
    return SpeedModel(config=CONFIG).fit(frame)


def test_route_endpoints_are_included():
    """The brief is explicit: src and dest both appear in the route array."""

    graph = _diamond_graph()
    model = _model({i: (lambda m: 40.0) for i in range(4)})
    td = TimeDependentGraph(graph, model, 0, 0, CONFIG)
    result = time_dependent_dijkstra(td, 0, 3, 8 * 3600)
    assert result.path[0] == 0
    assert result.path[-1] == 3


def test_picks_the_genuinely_faster_route():
    """With uniform speeds the shorter road must win."""

    graph = _diamond_graph()
    model = _model({i: (lambda m: 40.0) for i in range(4)})
    td = TimeDependentGraph(graph, model, 0, 0, CONFIG)
    result = time_dependent_dijkstra(td, 0, 3, 3 * 3600)
    assert result.path == [0, 1, 3]


def test_time_dependence_changes_the_answer():
    """When the short road jams, the router must switch to the long one."""

    graph = _diamond_graph()
    # Edges 0 and 1 form the short road and collapse to 5 km/h at 08:00.
    def jammed(minute):
        return 5.0 if 420 <= minute <= 540 else 60.0

    model = _model({0: jammed, 1: jammed, 2: lambda m: 45.0, 3: lambda m: 45.0})
    td = TimeDependentGraph(graph, model, 0, 0, CONFIG)

    free_flow = time_dependent_dijkstra(td, 0, 3, 3 * 3600)
    rush_hour = time_dependent_dijkstra(td, 0, 3, 8 * 3600)

    assert free_flow.path == [0, 1, 3], "off-peak should use the short road"
    assert rush_hour.path == [0, 2, 3], "at the peak it should detour"
    assert rush_hour.eta_seconds != free_flow.eta_seconds


def test_matches_networkx_when_weights_are_constant():
    """With a time-invariant profile the answer must equal classical Dijkstra."""

    rng = np.random.default_rng(0)
    graph = nx.DiGraph()
    base = nx.gnm_random_graph(25, 90, seed=3, directed=True)
    for i, (u, v) in enumerate(base.edges()):
        graph.add_edge(u, v, id=i, length=float(rng.uniform(200, 2000)))

    speeds = {int(d["id"]): 40.0 for _, _, d in graph.edges(data=True)}
    model = _model({eid: (lambda m, s=s: s) for eid, s in speeds.items()})
    td = TimeDependentGraph(graph, model, 0, 0, CONFIG)

    weights = td.static_travel_times(0)
    for u, v, data in graph.edges(data=True):
        data["w"] = weights[(u, v)]

    checked = 0
    for source in list(graph.nodes())[:8]:
        for target in list(graph.nodes())[:8]:
            if source == target:
                continue
            try:
                expected = nx.shortest_path_length(graph, source, target, weight="w")
            except nx.NetworkXNoPath:
                continue
            result = time_dependent_dijkstra(td, source, target, 0.0)
            assert result.eta_seconds == pytest.approx(expected, rel=1e-9)
            checked += 1
    assert checked > 5, "the random graph produced too few connected pairs to be a real test"


def test_same_source_and_target_is_zero():
    graph = _diamond_graph()
    model = _model({i: (lambda m: 40.0) for i in range(4)})
    td = TimeDependentGraph(graph, model, 0, 0, CONFIG)
    result = time_dependent_dijkstra(td, 2, 2, 0.0)
    assert result.eta_seconds == 0.0
    assert result.path == [2]


def test_unreachable_target_raises():
    graph = _diamond_graph()
    graph.add_node(99)
    model = _model({i: (lambda m: 40.0) for i in range(4)})
    td = TimeDependentGraph(graph, model, 0, 0, CONFIG)
    with pytest.raises(NoRouteError):
        time_dependent_dijkstra(td, 0, 99, 0.0)


def test_static_baseline_never_beats_time_dependent():
    """Time-dependent Dijkstra is optimal, so it cannot lose to the baseline."""

    graph = _diamond_graph()

    def jammed(minute):
        return 5.0 if 420 <= minute <= 540 else 60.0

    model = _model({0: jammed, 1: jammed, 2: lambda m: 45.0, 3: lambda m: 45.0})
    td = TimeDependentGraph(graph, model, 0, 0, CONFIG)

    for start_minute in range(0, 1440, 30):
        start = start_minute * 60.0
        dynamic = time_dependent_dijkstra(td, 0, 3, start)
        baseline = static_route(td, 0, 3, start)
        assert dynamic.eta_seconds <= baseline.eta_seconds + 1e-6
