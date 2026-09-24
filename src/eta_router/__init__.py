"""Time-dependent ETA prediction and routing on an urban road graph.

Public entry points:

    from eta_router import SpeedModel, TimeDependentGraph, route, best_departure

See ``docs/method.md`` for the modelling write-up.
"""

from .config import Config, DEFAULT_CONFIG
from .departure import DepartureCurve, DepartureOption, best_departure, departure_curve
from .routing import route, static_route, time_dependent_dijkstra
from .speed_model import SpeedModel
from .travel_time import TimeDependentGraph

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "DepartureCurve",
    "DepartureOption",
    "SpeedModel",
    "TimeDependentGraph",
    "best_departure",
    "departure_curve",
    "route",
    "static_route",
    "time_dependent_dijkstra",
]

__version__ = "0.1.0"
