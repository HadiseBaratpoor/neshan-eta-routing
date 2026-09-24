"""Cleaning the aggregated speed observations.

Two problems dominate this dataset, and the challenge brief names both:

1. **Outliers.** Aggregated probe speeds include vehicles that stopped, GPS
   drift in tunnels, and map-matching errors. A single 200 km/h reading on a
   residential street drags a mean badly.
2. **Missing cells.** Many ``(edge, minute, weather, holiday)`` combinations
   were simply never observed, so the model must fall back rather than
   produce ``NaN`` weights.

This module handles (1). Imputation of (2) lives in ``speed_model.py``, where
it can use the fitted hierarchy.

Filtering strategy, in order:

* drop physically impossible speeds (outside ``[min_speed, max_speed]``);
* within each ``(edge_id, weather, is_holiday)`` group, drop points whose
  modified z-score exceeds ``mad_z_threshold``.

The modified z-score uses the median and the median absolute deviation, so a
handful of extreme values cannot mask each other the way they do with a mean
and a standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG

# Scale factor making the MAD a consistent estimator of sigma for normal data.
_MAD_TO_SIGMA = 1.4826


@dataclass
class CleaningReport:
    """What cleaning actually removed - printed by the pipeline scripts."""

    n_input: int
    n_out_of_bounds: int
    n_mad_outliers: int
    n_output: int

    @property
    def removed_fraction(self) -> float:
        return 0.0 if self.n_input == 0 else 1.0 - self.n_output / self.n_input

    def summary(self) -> str:
        return (
            f"rows in: {self.n_input:,} | "
            f"out of physical bounds: {self.n_out_of_bounds:,} | "
            f"MAD outliers: {self.n_mad_outliers:,} | "
            f"rows kept: {self.n_output:,} ({100 * (1 - self.removed_fraction):.1f}%)"
        )


def modified_zscore(values: np.ndarray) -> np.ndarray:
    """Return |modified z-score| for ``values``.

    Uses the median absolute deviation. When the MAD is zero (a group where
    most readings are identical) it falls back to the mean absolute deviation
    so that constant groups are not reported as all-outliers.
    """

    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return values
    median = np.median(values)
    deviations = np.abs(values - median)
    mad = np.median(deviations)
    if mad > 0:
        scale = _MAD_TO_SIGMA * mad
    else:
        mean_ad = deviations.mean()
        if mean_ad <= 0:
            return np.zeros_like(values)
        scale = 1.253314 * mean_ad
    return deviations / scale


def clean_speeds(
    frame: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove impossible and statistically extreme speed observations."""

    n_input = len(frame)
    working = frame.copy()

    in_bounds = working["speed"].between(config.min_speed_kmh, config.max_speed_kmh)
    n_out_of_bounds = int((~in_bounds).sum())
    working = working.loc[in_bounds].copy()

    if working.empty:
        report = CleaningReport(n_input, n_out_of_bounds, 0, 0)
        return working, report

    group_keys = ["edge_id", "weather", "is_holiday"]
    scores = working.groupby(group_keys, sort=False)["speed"].transform(
        lambda s: pd.Series(modified_zscore(s.to_numpy()), index=s.index)
        if len(s) >= config.min_group_size_for_mad
        else pd.Series(np.zeros(len(s)), index=s.index)
    )
    keep = scores <= config.mad_z_threshold
    n_mad_outliers = int((~keep).sum())
    working = working.loc[keep].copy()

    report = CleaningReport(
        n_input=n_input,
        n_out_of_bounds=n_out_of_bounds,
        n_mad_outliers=n_mad_outliers,
        n_output=len(working),
    )
    return working, report


def add_time_bin(frame: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Attach the coarse time-of-day bin used for profile estimation."""

    out = frame.copy()
    minute = out["minute"].to_numpy() % 1440
    out["minute"] = minute
    out["time_bin"] = (minute // config.time_bin_minutes).astype("int16")
    return out
