"""Validation of the speed model and of whole-journey ETAs.

The organisers score two things: ETA error and route similarity. Both are
implemented here so the same numbers can be produced locally before
submitting anything.

Speed-level validation uses a **day-wise split** when the dataset carries a
day column. Splitting rows at random would leak: the same edge-minute-weather
cell appears on several days, so a random split lets the model see a near
copy of every test row and reports an optimistic error. Holding out whole
days is the honest analogue of "predict tomorrow".

Route similarity uses the Jaccard index over the node sets plus an exact-match
rate, which is what "how close is my route to the best route" means when the
ground-truth route is a node list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .speed_model import SpeedModel


@dataclass
class SpeedMetrics:
    mae_kmh: float
    rmse_kmh: float
    mape_percent: float
    n: int

    def summary(self) -> str:
        return (
            f"n={self.n:,} | MAE={self.mae_kmh:.3f} km/h | "
            f"RMSE={self.rmse_kmh:.3f} km/h | MAPE={self.mape_percent:.2f}%"
        )


@dataclass
class EtaMetrics:
    mae_seconds: float
    rmse_seconds: float
    mape_percent: float
    n: int

    def summary(self) -> str:
        return (
            f"n={self.n:,} | MAE={self.mae_seconds:.1f} s | "
            f"RMSE={self.rmse_seconds:.1f} s | MAPE={self.mape_percent:.2f}%"
        )


def speed_metrics(actual: np.ndarray, predicted: np.ndarray) -> SpeedMetrics:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    actual, predicted = actual[mask], predicted[mask]
    if actual.size == 0:
        return SpeedMetrics(float("nan"), float("nan"), float("nan"), 0)
    error = predicted - actual
    return SpeedMetrics(
        mae_kmh=float(np.mean(np.abs(error))),
        rmse_kmh=float(np.sqrt(np.mean(error**2))),
        mape_percent=float(np.mean(np.abs(error / actual)) * 100.0),
        n=int(actual.size),
    )


def eta_metrics(actual: np.ndarray, predicted: np.ndarray) -> EtaMetrics:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    actual, predicted = actual[mask], predicted[mask]
    if actual.size == 0:
        return EtaMetrics(float("nan"), float("nan"), float("nan"), 0)
    error = predicted - actual
    return EtaMetrics(
        mae_seconds=float(np.mean(np.abs(error))),
        rmse_seconds=float(np.sqrt(np.mean(error**2))),
        mape_percent=float(np.mean(np.abs(error / actual)) * 100.0),
        n=int(actual.size),
    )


def route_similarity(predicted: list, reference: list) -> float:
    """Jaccard index over node sets; 1.0 means identical node coverage."""

    a, b = set(predicted), set(reference)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def split_by_day(
    frame: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole days when possible, otherwise fall back to a random split.

    Returns ``(train, validation)``. A random fallback is clearly worse but
    keeps the pipeline runnable on exports without a day column.
    """

    if "day" in frame.columns:
        days = np.sort(frame["day"].unique())
        if len(days) > config.validation_days:
            holdout = set(days[-config.validation_days :])
            mask = frame["day"].isin(holdout)
            return frame.loc[~mask].copy(), frame.loc[mask].copy()

    rng = np.random.default_rng(config.random_seed)
    mask = rng.random(len(frame)) < 0.2
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def evaluate_speed_model(
    model: SpeedModel,
    validation: pd.DataFrame,
) -> SpeedMetrics:
    """Score a fitted model against held-out speed observations."""

    predictions = np.empty(len(validation), dtype="float64")
    grouped = validation.groupby(["weather", "is_holiday"], sort=False)
    for (weather, holiday), part in grouped:
        values = model.predict(
            part["edge_id"].to_numpy(),
            part["minute"].to_numpy(),
            int(weather),
            int(holiday),
        )
        predictions[validation.index.get_indexer(part.index)] = values
    return speed_metrics(validation["speed"].to_numpy(), predictions)


def constant_speed_baseline(train: pd.DataFrame, validation: pd.DataFrame) -> SpeedMetrics:
    """Reference point: predict each edge's overall median speed, always."""

    medians = train.groupby("edge_id")["speed"].median()
    fallback = float(train["speed"].median())
    predicted = validation["edge_id"].map(medians).fillna(fallback).to_numpy()
    return speed_metrics(validation["speed"].to_numpy(), predicted)
