"""Answering "when should I leave?" - the question the brief actually poses.

The problem statement ends on this: *"with this system we can easily know,
given today's conditions, what time to leave in order to arrive sooner."*
Choosing the departure minute is worth far more than choosing the path. On a
congested corridor the gap between the best and worst departure inside a
two-hour window is routinely tens of percent, while re-routing around a jam
saves single digits.

``departure_curve`` sweeps a window of candidate departure minutes, runs the
full time-dependent router at each, and returns the travel time for every
one. ``best_departure`` picks the minimum, and ``latest_departure_for_arrival``
answers the inverse question - *the meeting is at 09:00, when must I leave?* -
by scanning backwards for the latest departure that still arrives on time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MINUTES_PER_DAY
from .routing import NoRouteError, time_dependent_dijkstra
from .travel_time import TimeDependentGraph


@dataclass
class DepartureOption:
    """One candidate departure."""

    depart_minute: int
    travel_seconds: float
    arrive_minute: float
    path: list

    @property
    def travel_minutes(self) -> float:
        return self.travel_seconds / 60.0


@dataclass
class DepartureCurve:
    """Travel time as a function of departure minute, plus the best choice."""

    options: list[DepartureOption]

    @property
    def best(self) -> DepartureOption:
        return min(self.options, key=lambda o: o.travel_seconds)

    @property
    def worst(self) -> DepartureOption:
        return max(self.options, key=lambda o: o.travel_seconds)

    @property
    def minutes(self) -> np.ndarray:
        return np.array([o.depart_minute for o in self.options])

    @property
    def travel_times(self) -> np.ndarray:
        return np.array([o.travel_seconds for o in self.options])

    def saving_vs_worst(self) -> float:
        """Seconds saved by leaving at the best minute instead of the worst."""

        return self.worst.travel_seconds - self.best.travel_seconds

    def summary(self) -> str:
        best, worst = self.best, self.worst
        pct = 100.0 * self.saving_vs_worst() / max(worst.travel_seconds, 1e-6)
        return (
            f"best: leave at {_hhmm(best.depart_minute)} -> {best.travel_minutes:.1f} min | "
            f"worst: leave at {_hhmm(worst.depart_minute)} -> {worst.travel_minutes:.1f} min | "
            f"saving {self.saving_vs_worst() / 60:.1f} min ({pct:.1f}%)"
        )


def _hhmm(minute: int) -> str:
    minute = int(minute) % MINUTES_PER_DAY
    return f"{minute // 60:02d}:{minute % 60:02d}"


def departure_curve(
    td_graph: TimeDependentGraph,
    source,
    target,
    window_start: int,
    window_end: int,
    step_minutes: int = 5,
) -> DepartureCurve:
    """Travel time for every candidate departure in ``[window_start, window_end]``.

    Minutes are measured from the start of the day and the window may wrap
    past midnight (e.g. 1380 -> 120).
    """

    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")

    span = (int(window_end) - int(window_start)) % MINUTES_PER_DAY
    if span == 0:
        span = MINUTES_PER_DAY
    candidates = [(int(window_start) + offset) % MINUTES_PER_DAY for offset in range(0, span + 1, step_minutes)]

    options: list[DepartureOption] = []
    for minute in candidates:
        try:
            result = time_dependent_dijkstra(td_graph, source, target, float(minute) * 60.0)
        except (NoRouteError, KeyError):
            continue
        options.append(
            DepartureOption(
                depart_minute=minute,
                travel_seconds=result.eta_seconds,
                arrive_minute=(minute + result.eta_seconds / 60.0) % MINUTES_PER_DAY,
                path=[int(n) for n in result.path],
            )
        )

    if not options:
        raise NoRouteError(f"no route from {source!r} to {target!r} at any departure in the window")
    return DepartureCurve(options=options)


def best_departure(
    td_graph: TimeDependentGraph,
    source,
    target,
    window_start: int,
    window_end: int,
    step_minutes: int = 5,
) -> DepartureOption:
    """The departure minute inside the window with the shortest travel time."""

    return departure_curve(td_graph, source, target, window_start, window_end, step_minutes).best


def latest_departure_for_arrival(
    td_graph: TimeDependentGraph,
    source,
    target,
    arrive_by_minute: int,
    search_back_minutes: int = 180,
    step_minutes: int = 5,
) -> DepartureOption | None:
    """Latest departure that still arrives by ``arrive_by_minute``.

    Scans backwards from the deadline. Returns ``None`` when even leaving
    ``search_back_minutes`` early is not enough.
    """

    for offset in range(0, search_back_minutes + 1, step_minutes):
        depart = (int(arrive_by_minute) - offset) % MINUTES_PER_DAY
        try:
            result = time_dependent_dijkstra(td_graph, source, target, float(depart) * 60.0)
        except (NoRouteError, KeyError):
            return None
        if result.eta_seconds <= offset * 60.0:
            return DepartureOption(
                depart_minute=depart,
                travel_seconds=result.eta_seconds,
                arrive_minute=(depart + result.eta_seconds / 60.0) % MINUTES_PER_DAY,
                path=[int(n) for n in result.path],
            )
    return None
