"""Shortest-path search on a time-dependent road network.

The brief's final hint is the important one: classical Dijkstra freezes the
weights at departure time, so a route chosen at 08:00 is priced with 08:00
traffic even for the edge you reach at 08:40. On a congested network that is
exactly when the answer is worst.

``time_dependent_dijkstra`` fixes this. The label of a node is its **arrival
time**, and an edge is priced at the moment the traveller actually arrives at
its tail:

    arrival(v) = arrival(u) + tau_{u,v}(arrival(u))

Because travel times are positive and the profile is FIFO-corrected (see
``travel_time.py``), the label-setting property still holds: the first time a
node leaves the priority queue, its arrival time is optimal. The result is a
provably minimal ETA, not a heuristic.

``static_route`` implements the frozen-weight baseline so the two can be
compared directly - that comparison is the headline evidence in the report.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .travel_time import TimeDependentGraph


class NoRouteError(RuntimeError):
    """Raised when the destination is unreachable from the source."""


@dataclass
class RouteResult:
    """One answered query."""

    path: list
    eta_seconds: float
    expanded_nodes: int

    def as_submission_fields(self) -> tuple[int, str]:
        """``(eta, route)`` formatted exactly as the challenge expects."""

        route_text = "[" + ",".join(str(n) for n in self.path) + "]"
        return int(round(self.eta_seconds)), route_text


def time_dependent_dijkstra(
    td_graph: TimeDependentGraph,
    source,
    target,
    start_second: float,
) -> RouteResult:
    """Minimum-arrival-time path, evaluating each edge at its real entry time."""

    graph = td_graph.graph
    if source not in graph:
        raise NoRouteError(f"source node {source!r} is not in the graph")
    if target not in graph:
        raise NoRouteError(f"target node {target!r} is not in the graph")
    if source == target:
        return RouteResult(path=[source], eta_seconds=0.0, expanded_nodes=0)

    best: dict = {source: float(start_second)}
    previous: dict = {}
    settled: set = set()
    heap: list[tuple[float, int, object]] = [(float(start_second), 0, source)]
    counter = 1
    expanded = 0

    while heap:
        arrival, _, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        expanded += 1

        if node == target:
            return RouteResult(
                path=_reconstruct(previous, source, target),
                eta_seconds=arrival - float(start_second),
                expanded_nodes=expanded,
            )

        for neighbour in graph.successors(node):
            if neighbour in settled:
                continue
            cost = td_graph.edge_travel_time(node, neighbour, arrival)
            candidate = arrival + cost
            if candidate < best.get(neighbour, float("inf")):
                best[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(heap, (candidate, counter, neighbour))
                counter += 1

    raise NoRouteError(f"no path from {source!r} to {target!r}")


def static_route(
    td_graph: TimeDependentGraph,
    source,
    target,
    start_second: float,
) -> RouteResult:
    """Baseline: choose the path with weights frozen at departure time.

    The reported ETA is still measured honestly by replaying the chosen path
    through the time-dependent function, so the baseline is penalised for
    being wrong rather than for being differently scored.
    """

    graph = td_graph.graph
    if source not in graph or target not in graph:
        raise NoRouteError(f"{source!r} or {target!r} missing from the graph")
    if source == target:
        return RouteResult(path=[source], eta_seconds=0.0, expanded_nodes=0)

    start_minute = int((start_second // 60) % 1440)
    weights = td_graph.static_travel_times(start_minute)

    best: dict = {source: 0.0}
    previous: dict = {}
    settled: set = set()
    heap: list[tuple[float, int, object]] = [(0.0, 0, source)]
    counter = 1
    expanded = 0

    while heap:
        cost_so_far, _, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        expanded += 1
        if node == target:
            path = _reconstruct(previous, source, target)
            return RouteResult(
                path=path,
                eta_seconds=td_graph.path_travel_time(path, start_second),
                expanded_nodes=expanded,
            )
        for neighbour in graph.successors(node):
            if neighbour in settled:
                continue
            candidate = cost_so_far + weights[(node, neighbour)]
            if candidate < best.get(neighbour, float("inf")):
                best[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(heap, (candidate, counter, neighbour))
                counter += 1

    raise NoRouteError(f"no path from {source!r} to {target!r}")


def route(
    td_graph: TimeDependentGraph,
    source,
    target,
    start_minute: int,
) -> RouteResult:
    """Convenience wrapper taking the challenge's unit: minutes after midnight."""

    return time_dependent_dijkstra(td_graph, source, target, float(start_minute) * 60.0)


def _reconstruct(previous: dict, source, target) -> list:
    path = [target]
    node = target
    while node != source:
        node = previous[node]
        path.append(node)
    path.reverse()
    return path
