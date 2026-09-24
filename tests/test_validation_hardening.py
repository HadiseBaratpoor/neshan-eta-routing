import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.data import DataError, _validate_graph, load_graph
from eta_router.submission import validate_submission


def _submission(route="[1,2]", eta=5, src=1, dest=2):
    return pd.DataFrame({"src": [src], "dest": [dest], "route_start_t": [0],
                         "is_holiday": [0], "weather": [0], "eta": [eta], "route": [route]})


def test_bad_route_text_is_validation_problem():
    problems = validate_submission(_submission("[1, nope]"), 1)
    assert any("malformed route" in problem for problem in problems)


def test_submission_rejects_route_edge_missing_from_graph():
    graph = nx.DiGraph()
    graph.add_edge(1, 3)
    graph.add_edge(3, 2)
    assert any("graph edge" in p for p in validate_submission(_submission(), 1, graph))


def test_source_equals_destination_has_zero_eta_and_single_node_route():
    assert validate_submission(_submission("[4]", eta=0, src=4, dest=4), 1) == []
    assert any("source==destination" in p for p in validate_submission(_submission("[4,4]", eta=1, src=4, dest=4), 1))


@pytest.mark.parametrize("length", [0, -1, float("nan"), float("inf")])
def test_graph_rejects_nonpositive_or_nonfinite_length(length):
    graph = nx.DiGraph()
    graph.add_edge(1, 2, id=1, length=length)
    with pytest.raises(DataError, match="finite positive"):
        _validate_graph(graph)


def test_graph_rejects_duplicate_edge_ids():
    graph = nx.DiGraph()
    graph.add_edge(1, 2, id=7, length=1)
    graph.add_edge(2, 3, id=7, length=1)
    with pytest.raises(DataError, match="unique"):
        _validate_graph(graph)


def test_graph_pickle_requires_explicit_trust(tmp_path):
    path = tmp_path / "graph.gpickle"
    path.write_bytes(b"irrelevant")
    with pytest.raises(DataError, match="trusted_pickle=True"):
        load_graph(path)
