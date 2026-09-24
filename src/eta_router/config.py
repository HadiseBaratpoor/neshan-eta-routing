"""Project-wide configuration.

Every tunable constant used by the pipeline lives here so that experiments are
reproducible and reviewers can see the assumptions in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class Config:
    """Tunables for cleaning, modelling and routing."""

    # ---- file names inside data/raw -------------------------------------
    graph_file: str = "network.gpickle"
    dataset_file: str = "dataset.csv"
    test_cases_file: str = "test_cases.csv"

    # ---- physical sanity bounds (km/h) ----------------------------------
    # Aggregated probe speeds contain GPS artefacts: standing vehicles,
    # tunnel drift, and occasional impossible values. Anything outside this
    # band is discarded before any statistic is computed.
    min_speed_kmh: float = 1.0
    max_speed_kmh: float = 130.0

    # ---- outlier filtering ----------------------------------------------
    # Robust per (edge, weather, holiday) filtering with the median absolute
    # deviation. 3.5 is the conventional threshold for the modified z-score.
    mad_z_threshold: float = 3.5
    # A group needs at least this many observations before MAD filtering is
    # trusted; smaller groups are only clipped to the physical bounds.
    min_group_size_for_mad: int = 8

    # ---- temporal smoothing ---------------------------------------------
    # Speed profiles are estimated on a coarse grid then smoothed, because
    # per-minute medians are noisy and many minutes are unobserved.
    # 10 minutes resolves a rush-hour peak without making cells too sparse.
    time_bin_minutes: int = 10
    # Width of the circular Gaussian kernel used to smooth a daily profile,
    # expressed in time bins. Wide enough to fill gaps, narrow enough to keep
    # the morning peak sharp.
    smoothing_sigma_bins: float = 1.5

    # ---- hierarchical fallback ------------------------------------------
    # Shrinkage strength when blending a sparse cell towards its parent level
    # (see speed_model.py): weight = n / (n + k), so a cell needs k
    # observations to be trusted half on its own evidence.
    #
    # k is set ADAPTIVELY at fit time as a fraction of the typical cell's
    # observation count, because a fixed k means different things on a
    # 400k-row export and a 20k-row one. This fraction is the real tunable
    # and it transfers across dataset sizes; 1.5 was selected on held-out
    # days (see docs/method.md).
    shrinkage_fraction: float = 1.5
    # Floor, so a very sparse dataset still gets some regularisation.
    min_shrinkage_k: float = 5.0

    # ---- routing ---------------------------------------------------------
    # Time-dependent Dijkstra evaluates edge costs at the arrival time of the
    # tail node, so weights stay non-negative (time, in seconds) and FIFO
    # enforcement keeps the label-setting property valid.
    enforce_fifo: bool = True
    # Free-flow speed assumed for edges with no observation at all (km/h).
    fallback_speed_kmh: float = 30.0

    # ---- evaluation -------------------------------------------------------
    random_seed: int = 42
    validation_days: int = 2

    raw_dir: Path = field(default=RAW_DIR)
    processed_dir: Path = field(default=PROCESSED_DIR)
    reports_dir: Path = field(default=REPORTS_DIR)

    @property
    def graph_path(self) -> Path:
        return self.raw_dir / self.graph_file

    @property
    def dataset_path(self) -> Path:
        return self.raw_dir / self.dataset_file

    @property
    def test_cases_path(self) -> Path:
        return self.raw_dir / self.test_cases_file

    @property
    def n_time_bins(self) -> int:
        return MINUTES_PER_DAY // self.time_bin_minutes


DEFAULT_CONFIG = Config()

WEATHER_LABELS = {0: "sunny", 1: "rainy", 2: "snowy"}
