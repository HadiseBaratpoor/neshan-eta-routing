"""Turning predicted speeds into time-dependent edge traversal times.

The routing layer needs one function:

    travel_time(u, v, departure_second) -> seconds

Unit discipline (the brief warns about this explicitly):

    speed is km/h, length is metres, output is seconds

    seconds = length_m / (speed_kmh / 3.6)

Weights are therefore strictly positive, which is what Dijkstra requires.

FIFO / non-overtaking
---------------------
A time-dependent network is *FIFO* when leaving later can never mean arriving
earlier. Real traffic is close to FIFO, but an interpolated speed profile can
violate it at a sharp rush-hour edge - and a violation breaks the
label-setting property that makes time-dependent Dijkstra correct.

``TimeDependentGraph`` precomputes arrival times on a discrete departure grid
and enforces monotonicity with a running maximum, so the routing layer is
provably safe. The correction is tiny in practice and always conservative
(it never invents a faster trip).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .config import Config, DEFAULT_CONFIG, MINUTES_PER_DAY
from .speed_model import SpeedModel

SECONDS_PER_DAY = MINUTES_PER_DAY * 60


def seconds_from_speed(length_m: float, speed_kmh: float) -> float:
    """Traversal time in seconds. Guards against zero/negative speed."""

    speed = max(float(speed_kmh), 1e-3)
    return float(length_m) / (speed / 3.6)


@dataclass
class TimeDependentGraph:
    """Pre-resolved travel-time functions for one (weather, holiday) scenario.

    Building this once per scenario and reusing it across every test case in
    that scenario is what keeps the submission script fast: the expensive
    model lookups happen ``n_edges x n_bins`` times, not once per Dijkstra
    relaxation.
    """

    graph: nx.DiGraph
    model: SpeedModel
    weather: int
    is_holiday: int
    config: Config = DEFAULT_CONFIG

    def __post_init__(self) -> None:
        self._edges = list(self.graph.edges(data=True))
        self._edge_key = {(u, v): i for i, (u, v, _) in enumerate(self._edges)}
        self._lengths = np.array([float(d["length"]) for _, _, d in self._edges])
        self._edge_ids = np.array([int(d["id"]) for _, _, d in self._edges], dtype="int64")

        self._bin_minutes = self.config.time_bin_minutes
        self._n_bins = self.config.n_time_bins
        bin_starts = np.arange(self._n_bins) * self._bin_minutes

        # travel_seconds[edge, bin] for a departure inside that bin
        speeds = np.vstack(
            [
                self.model.predict(self._edge_ids, np.full(len(self._edge_ids), m), self.weather, self.is_holiday)
                for m in bin_starts
            ]
        ).T  # shape (n_edges, n_bins)
        speeds = np.clip(speeds, self.config.min_speed_kmh, self.config.max_speed_kmh)
        self._travel_seconds = self._lengths[:, None] / (speeds / 3.6)

        if self.config.enforce_fifo:
            self._travel_seconds = self._enforce_fifo(self._travel_seconds)

    def _enforce_fifo(self, travel: np.ndarray) -> np.ndarray:
        """Clamp the profile so arrival time is non-decreasing in departure time.

        For departures at bin starts ``t_i`` the arrival is ``t_i + tau_i``.
        Walking forward, an arrival that would precede the previous bin's
        arrival is raised to it. The day wraps, so one extra pass propagates
        a correction across midnight.
        """

        bin_seconds = self._bin_minutes * 60
        corrected = travel.copy()
        # The arrival envelope includes the previous day's departures. Each
        # sample is raised only enough to preserve their arrivals; using a
        # periodic envelope also fixes the last-to-first transition.
        for offset in range(1, self._n_bins):
            corrected = np.maximum(corrected, np.roll(travel, offset, axis=1) - offset * bin_seconds)
        return np.maximum(corrected, 1e-3)

    # ---------------------------------------------------------------- query
    def edge_travel_time(self, u, v, departure_second: float) -> float:
        """Seconds to traverse ``(u, v)`` when leaving ``u`` at that time."""

        idx = self._edge_key.get((u, v))
        if idx is None:
            raise KeyError(f"edge ({u}, {v}) is not in the graph")
        return self.edge_travel_time_by_index(idx, departure_second)

    def edge_travel_time_by_index(self, idx: int, departure_second: float) -> float:
        bin_seconds = self._bin_minutes * 60
        phase = departure_second % SECONDS_PER_DAY
        bin_index = int(phase // bin_seconds)
        fraction = (phase - bin_index * bin_seconds) / bin_seconds
        current = self._travel_seconds[idx, bin_index]
        following = self._travel_seconds[idx, (bin_index + 1) % self._n_bins]
        return float((1 - fraction) * current + fraction * following)

    def static_travel_times(self, minute: int) -> dict[tuple, float]:
        """Frozen weights at one instant - used by the static baseline."""

        bin_index = int((minute % MINUTES_PER_DAY) // self._bin_minutes)
        column = self._travel_seconds[:, bin_index]
        return {(u, v): float(column[i]) for i, (u, v, _) in enumerate(self._edges)}

    def path_travel_time(self, path: list, start_second: float) -> float:
        """Total seconds for a node path, re-evaluating each edge on arrival.

        This is the honest evaluation of any route: the second edge is priced
        at the time the traveller actually reaches it, not at departure.
        """

        t = float(start_second)
        for u, v in zip(path[:-1], path[1:]):
            t += self.edge_travel_time(u, v, t)
        return t - float(start_second)

    @property
    def free_flow_seconds(self) -> np.ndarray:
        """Fastest traversal of each edge over the whole day - the A* bound."""

        return self._travel_seconds.min(axis=1)

    @property
    def edges(self) -> list:
        return self._edges
