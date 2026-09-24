"""Estimating edge speed for every (edge, minute, weather, holiday) cell.

The system must answer: *how fast is this street at 07:42 on a rainy working
day?* - including for the many cells that were never observed.

Why not a single global regressor
---------------------------------
A gradient-boosted model over ``(edge_id, minute, weather, is_holiday)``
treats the edge id as a categorical with thousands of levels. With a few
hundred rows per edge it memorises exactly the way the brief warns about, and
it has no principled answer for an edge-condition pair it never saw.

The estimator used here
-----------------------
A **multiplicative profile decomposition**, which is the standard way traffic
speed profiles are built, and which degrades gracefully as data thins:

    speed(edge, bin, weather, holiday)
        = city(bin, weather, holiday)   # when does the whole city slow down
        x edge_level(edge)              # is this street fast or slow
        x edge_shape(edge, bin)         # does it have its own rush pattern
        x residual(edge, bin, w, h)     # anything condition-specific left

Each factor is estimated on the *residual* left by the factors above it, so
they do not fight each other. Crucially the edge factors are learned from
**condition-normalised** observations: dividing by the city profile first
removes weather and holiday effects, so an edge observed only on sunny
working days still transfers correctly to a snowy holiday.

Robustness comes from three places:

* every factor is a **median**, not a mean, so residual outliers cannot drag it;
* every daily profile is smoothed with a **circular Gaussian kernel**, so
  midnight wraps onto 23:59 and unobserved minutes borrow from neighbours -
  this is the imputation step the brief asks for;
* sparse cells are shrunk towards the level above with a James-Stein weight

      w = n / (n + k),  estimate = w * own + (1 - w) * parent

  so two observations mostly inherit the parent while a hundred stand alone.

This is deliberately a *smoother*, not a *fitter*: there is no capacity to
memorise individual rows, which is what keeps held-out error close to
training error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG, MINUTES_PER_DAY

WEATHER_VALUES = (0, 1, 2)
HOLIDAY_VALUES = (0, 1)

# Multiplicative factors are clipped to this band. A factor outside it means
# the cell had almost no data and the estimate should not be trusted.
_MIN_FACTOR = 0.25
_MAX_FACTOR = 4.0


def circular_gaussian_kernel(n_bins: int, sigma_bins: float) -> np.ndarray:
    """Normalised circular Gaussian smoothing matrix of shape ``(n, n)``."""

    if sigma_bins <= 0:
        return np.eye(n_bins)
    idx = np.arange(n_bins)
    diff = np.abs(idx[:, None] - idx[None, :])
    diff = np.minimum(diff, n_bins - diff)  # the day is a circle
    kernel = np.exp(-0.5 * (diff / sigma_bins) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    return kernel


def _smooth_rows(values: np.ndarray, weights: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted circular smoothing of one or more daily profiles.

    ``values`` may hold NaN for unobserved bins and ``weights`` the
    observation counts. Returns the smoothed profiles together with the
    effective weight behind each smoothed bin, which the shrinkage step needs.
    """

    filled = np.nan_to_num(values, nan=0.0)
    w = np.where(np.isnan(values), 0.0, weights)
    numerator = filled * w @ kernel.T
    denominator = w @ kernel.T
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(denominator > 0, numerator / denominator, np.nan)
    return smoothed, denominator


def _shrink(own: np.ndarray, effective_n: np.ndarray, parent, k: float) -> np.ndarray:
    """Blend an estimate towards its parent in proportion to its evidence."""

    weight = effective_n / (effective_n + k)
    own_filled = np.where(np.isnan(own), parent, own)
    return weight * own_filled + (1.0 - weight) * parent


@dataclass
class SpeedModel:
    """Multiplicative hierarchical speed profiles with smoothing and shrinkage."""

    config: Config = DEFAULT_CONFIG

    _edge_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype="int64"), repr=False)
    _edge_index: dict[int, int] = field(default_factory=dict, repr=False)
    _global_speed: float = field(default=0.0, repr=False)
    _shrinkage_k: float = field(default=0.0, repr=False)
    # city[(weather, holiday)] -> profile over time bins, km/h
    _city: dict[tuple[int, int], np.ndarray] = field(default_factory=dict, repr=False)
    _edge_level: np.ndarray | None = field(default=None, repr=False)      # (n_edges,)
    _edge_shape: np.ndarray | None = field(default=None, repr=False)      # (n_edges, n_bins)
    # Dense lookup cube [weather, holiday, edge_row, bin] -> km/h
    _cube: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame) -> "SpeedModel":
        """Fit every level from a cleaned dataset carrying a ``time_bin`` column."""

        if "time_bin" not in frame.columns:
            raise ValueError("frame must carry 'time_bin'; call cleaning.add_time_bin first")
        if frame.empty:
            raise ValueError("cannot fit a speed model on an empty frame")

        cfg = self.config
        n_bins = cfg.n_time_bins
        kernel = circular_gaussian_kernel(n_bins, cfg.smoothing_sigma_bins)

        # Adaptive shrinkage. A fixed k regularises a 400k-row export very
        # differently from a 20k-row one, so k is expressed as a multiple of
        # the observations behind a typical (edge, bin) cell. Dense data
        # earns trust in its own cells; sparse data leans on the parent.
        n_edges_est = max(frame["edge_id"].nunique(), 1)
        per_cell = len(frame) / (n_edges_est * n_bins)
        k = max(cfg.min_shrinkage_k, cfg.shrinkage_fraction * per_cell)
        self._shrinkage_k = float(k)

        self._edge_ids = np.sort(frame["edge_id"].unique())
        self._edge_index = {int(e): i for i, e in enumerate(self._edge_ids)}
        n_edges = len(self._edge_ids)

        self._global_speed = float(frame["speed"].median())

        # -- level 0: city-wide daily profile for each condition ------------
        # Estimated from every edge at once, so it is dense and stable even
        # for rare conditions such as snow on a holiday.
        for weather in WEATHER_VALUES:
            for holiday in HOLIDAY_VALUES:
                subset = frame[(frame["weather"] == weather) & (frame["is_holiday"] == holiday)]
                values = np.full(n_bins, np.nan)
                counts = np.zeros(n_bins)
                if not subset.empty:
                    grouped = subset.groupby("time_bin")["speed"].agg(["median", "size"])
                    values[grouped.index.to_numpy()] = grouped["median"].to_numpy()
                    counts[grouped.index.to_numpy()] = grouped["size"].to_numpy()
                smoothed, effective = _smooth_rows(values[None, :], counts[None, :], kernel)
                profile = _shrink(smoothed[0], effective[0], self._global_speed, k)
                self._city[(weather, holiday)] = np.clip(
                    np.where(np.isnan(profile), self._global_speed, profile),
                    cfg.min_speed_kmh,
                    cfg.max_speed_kmh,
                )

        # -- condition-normalised observations ------------------------------
        # Dividing each observation by the city profile for its own condition
        # strips weather, holiday and time-of-day out, leaving a "how does
        # this street compare to the city right now" ratio. Everything below
        # is learned from that ratio, which is why an edge seen only in sunny
        # weather still generalises to snow.
        city_lookup = np.stack(
            [self._city[(w, h)] for w in WEATHER_VALUES for h in HOLIDAY_VALUES]
        )  # (6, n_bins)
        cond_index = (
            frame["weather"].to_numpy().astype("int64") * len(HOLIDAY_VALUES)
            + frame["is_holiday"].to_numpy().astype("int64")
        )
        bins = frame["time_bin"].to_numpy().astype("int64")
        reference = city_lookup[cond_index, bins]
        relative = frame["speed"].to_numpy() / np.maximum(reference, 1e-6)

        work = pd.DataFrame(
            {
                "edge_row": [self._edge_index[int(e)] for e in frame["edge_id"].to_numpy()],
                "time_bin": bins,
                "relative": relative,
            }
        )

        # -- level 1: is this street faster or slower than the city ---------
        level1 = work.groupby("edge_row")["relative"].agg(["median", "size"])
        own_level = np.ones(n_edges)
        level_counts = np.zeros(n_edges)
        own_level[level1.index.to_numpy()] = level1["median"].to_numpy()
        level_counts[level1.index.to_numpy()] = level1["size"].to_numpy()
        self._edge_level = np.clip(_shrink(own_level, level_counts, 1.0, k), _MIN_FACTOR, _MAX_FACTOR)

        # -- level 2: does this street have its own daily rhythm ------------
        # Residual after removing the edge's overall level; a value above 1
        # at 08:00 means this street suffers less than the city average then.
        work["residual"] = work["relative"] / self._edge_level[work["edge_row"].to_numpy()]
        shape_values, shape_counts = self._pivot(work, "residual", n_edges, n_bins)
        smoothed_shape, effective_shape = _smooth_rows(shape_values, shape_counts, kernel)
        self._edge_shape = np.clip(
            _shrink(smoothed_shape, effective_shape, 1.0, k), _MIN_FACTOR, _MAX_FACTOR
        )

        # -- level 3: anything genuinely condition-specific per edge --------
        # Fitted last and shrunk hardest, because this is the level with the
        # least data behind it and the most capacity to overfit.
        base = (
            self._edge_level[:, None] * self._edge_shape
        )  # (n_edges, n_bins), condition-free
        work["weather"] = frame["weather"].to_numpy()
        work["is_holiday"] = frame["is_holiday"].to_numpy()
        work["cond_residual"] = work["relative"] / base[work["edge_row"], work["time_bin"]]

        cube = np.empty((len(WEATHER_VALUES), len(HOLIDAY_VALUES), n_edges, n_bins))
        for w_i, weather in enumerate(WEATHER_VALUES):
            for h_i, holiday in enumerate(HOLIDAY_VALUES):
                subset = work[(work["weather"] == weather) & (work["is_holiday"] == holiday)]
                values, counts = self._pivot(subset, "cond_residual", n_edges, n_bins)
                smoothed, effective = _smooth_rows(values, counts, kernel)
                correction = np.clip(_shrink(smoothed, effective, 1.0, 2.0 * k), _MIN_FACTOR, _MAX_FACTOR)
                cube[w_i, h_i] = np.clip(
                    self._city[(weather, holiday)][None, :] * base * correction,
                    self.config.min_speed_kmh,
                    self.config.max_speed_kmh,
                )

        self._cube = cube
        return self

    def _pivot(self, frame: pd.DataFrame, column: str, n_edges: int, n_bins: int):
        """Median of ``column`` and its count per (edge row, time bin)."""

        values = np.full((n_edges, n_bins), np.nan)
        counts = np.zeros((n_edges, n_bins))
        if frame.empty:
            return values, counts
        grouped = frame.groupby(["edge_row", "time_bin"])[column].agg(["median", "size"])
        rows = grouped.index.get_level_values(0).to_numpy()
        cols = grouped.index.get_level_values(1).to_numpy()
        values[rows, cols] = grouped["median"].to_numpy()
        counts[rows, cols] = grouped["size"].to_numpy()
        return values, counts

    # -------------------------------------------------------------- predict
    def predict(
        self,
        edge_ids: np.ndarray | list[int],
        minutes: np.ndarray | list[int],
        weather: int,
        is_holiday: int,
    ) -> np.ndarray:
        """Vectorised speed lookup in km/h."""

        if self._cube is None:
            raise RuntimeError("SpeedModel.predict called before fit")
        edge_ids = np.asarray(edge_ids, dtype="int64")
        minutes = np.asarray(minutes, dtype="int64") % MINUTES_PER_DAY
        bins = (minutes // self.config.time_bin_minutes).astype("int64")

        rows = np.array([self._edge_index.get(int(e), -1) for e in edge_ids])
        w_i = WEATHER_VALUES.index(int(weather))
        h_i = HOLIDAY_VALUES.index(int(is_holiday))

        # An edge with no history at all falls back to the city profile,
        # which is still condition-aware - better than a fixed constant.
        city = self._city[(int(weather), int(is_holiday))]
        out = city[bins].astype("float64")
        known = rows >= 0
        if known.any():
            out[known] = self._cube[w_i, h_i][rows[known], bins[known]]
        return out

    def predict_one(self, edge_id: int, minute: int, weather: int, is_holiday: int) -> float:
        return float(self.predict([edge_id], [minute], weather, is_holiday)[0])

    # ----------------------------------------------------------- inspection
    @property
    def edge_ids(self) -> np.ndarray:
        return self._edge_ids

    def profile(self, edge_id: int, weather: int, is_holiday: int) -> np.ndarray:
        """The full smoothed daily profile of one edge, for plotting."""

        if self._cube is None:
            raise RuntimeError("SpeedModel.profile called before fit")
        row = self._edge_index.get(int(edge_id))
        if row is None:
            return self._city[(int(weather), int(is_holiday))].copy()
        w_i = WEATHER_VALUES.index(int(weather))
        h_i = HOLIDAY_VALUES.index(int(is_holiday))
        return self._cube[w_i, h_i][row].copy()

    def city_profile(self, weather: int, is_holiday: int) -> np.ndarray:
        return self._city[(int(weather), int(is_holiday))].copy()

    def coverage(self) -> dict[str, float]:
        """Sanity check on the fitted lookup cube."""

        if self._cube is None:
            return {"finite_fraction": 0.0}
        return {
            "finite_fraction": float(np.isfinite(self._cube).mean()),
            "n_edges": float(len(self._edge_ids)),
            "n_bins": float(self.config.n_time_bins),
            "global_median_speed_kmh": round(self._global_speed, 3),
            "shrinkage_k": round(self._shrinkage_k, 2),
        }
