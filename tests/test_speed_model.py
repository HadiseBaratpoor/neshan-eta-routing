"""Cleaning, speed-model behaviour, and submission format."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eta_router.cleaning import add_time_bin, clean_speeds, modified_zscore  # noqa: E402
from eta_router.config import Config  # noqa: E402
from eta_router.speed_model import SpeedModel, circular_gaussian_kernel  # noqa: E402
from eta_router.submission import (  # noqa: E402
    build_submission,
    format_route,
    parse_route,
    validate_submission,
)

# Mirror the production resolution rather than a coarse stand-in: with
# 60-minute bins the smoothing kernel spans 90 minutes and erases the very
# rush-hour dip these tests assert on.
CONFIG = Config(time_bin_minutes=10, smoothing_sigma_bins=1.0)


# ----------------------------------------------------------------- cleaning
def test_modified_zscore_flags_the_obvious_outlier():
    values = np.array([40.0, 41.0, 39.0, 40.5, 41.5, 39.5, 40.2, 250.0])
    scores = modified_zscore(values)
    assert scores[-1] > 3.5
    assert (scores[:-1] < 3.5).all()


def test_modified_zscore_constant_group_is_not_all_outliers():
    """A street where everyone drives the same speed is not anomalous."""

    scores = modified_zscore(np.full(10, 42.0))
    assert (scores < 3.5).all()


def test_cleaning_removes_impossible_speeds():
    frame = pd.DataFrame(
        {
            "edge_id": [1] * 12,
            "length": [500.0] * 12,
            "minute": list(range(0, 720, 60)),
            "speed": [40, 41, 39, 42, 38, 40, 41, 39, 0.0, 400.0, -5.0, 40],
            "is_holiday": [0] * 12,
            "weather": [0] * 12,
        }
    )
    cleaned, report = clean_speeds(frame, CONFIG)
    assert report.n_out_of_bounds == 3  # 0.0, 400.0 and -5.0
    assert cleaned["speed"].between(CONFIG.min_speed_kmh, CONFIG.max_speed_kmh).all()


def test_cleaning_keeps_almost_everything_on_clean_data():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "edge_id": np.repeat([1, 2], 100),
            "length": 500.0,
            "minute": np.tile(np.arange(100) * 10, 2),
            "speed": rng.normal(40, 3, 200).clip(5, 90),
            "is_holiday": 0,
            "weather": 0,
        }
    )
    _, report = clean_speeds(frame, CONFIG)
    assert report.removed_fraction < 0.05


def test_add_time_bin_wraps_the_day():
    hourly = Config(time_bin_minutes=60)
    frame = pd.DataFrame({"minute": [0, 59, 60, 1439, 1440, 1500]})
    out = add_time_bin(frame, hourly)
    assert out["time_bin"].tolist() == [0, 0, 1, 23, 0, 1]


# -------------------------------------------------------------- speed model
def test_kernel_rows_sum_to_one_and_wrap():
    kernel = circular_gaussian_kernel(24, 1.5)
    assert kernel.sum(axis=1) == pytest.approx(np.ones(24))
    # bin 0 must borrow from bin 23 as much as from bin 1
    assert kernel[0, 23] == pytest.approx(kernel[0, 1])


def _training_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for edge_id, base in ((1, 50.0), (2, 25.0)):
        for day in range(6):
            for minute in range(0, 1440, 10):
                dip = 1.0 - 0.5 * np.exp(-0.5 * ((minute - 480) / 60.0) ** 2)
                for weather, factor in ((0, 1.0), (1, 0.85)):
                    rows.append(
                        {
                            "day": day,
                            "edge_id": edge_id,
                            "length": 1000.0,
                            "minute": minute,
                            "speed": base * dip * factor * rng.normal(1.0, 0.02),
                            "is_holiday": 0,
                            "weather": weather,
                        }
                    )
    return add_time_bin(pd.DataFrame(rows), CONFIG)


def test_model_recovers_the_rush_hour_dip():
    model = SpeedModel(config=CONFIG).fit(_training_frame())
    at_peak = model.predict_one(1, 480, weather=0, is_holiday=0)
    off_peak = model.predict_one(1, 180, weather=0, is_holiday=0)
    assert at_peak < off_peak * 0.75


def test_model_separates_fast_and_slow_edges():
    model = SpeedModel(config=CONFIG).fit(_training_frame())
    fast = model.predict_one(1, 180, 0, 0)
    slow = model.predict_one(2, 180, 0, 0)
    assert fast > slow * 1.5


def test_model_learns_the_weather_effect():
    model = SpeedModel(config=CONFIG).fit(_training_frame())
    sunny = model.predict_one(1, 180, weather=0, is_holiday=0)
    rainy = model.predict_one(1, 180, weather=1, is_holiday=0)
    assert rainy < sunny


def test_model_generalises_to_an_unseen_condition():
    """Snow never appears in training; the estimate must still be sane."""

    model = SpeedModel(config=CONFIG).fit(_training_frame())
    snowy = model.predict_one(1, 180, weather=2, is_holiday=0)
    assert np.isfinite(snowy)
    assert CONFIG.min_speed_kmh <= snowy <= CONFIG.max_speed_kmh


def test_unknown_edge_falls_back_without_nan():
    model = SpeedModel(config=CONFIG).fit(_training_frame())
    value = model.predict_one(9999, 600, 0, 0)
    assert np.isfinite(value) and value > 0


def test_no_nan_anywhere_in_the_lookup_cube():
    """The imputation requirement: every cell must be answerable."""

    model = SpeedModel(config=CONFIG).fit(_training_frame())
    assert model.coverage()["finite_fraction"] == 1.0


def test_predictions_are_within_physical_bounds():
    model = SpeedModel(config=CONFIG).fit(_training_frame())
    for weather in (0, 1, 2):
        for minute in range(0, 1440, 37):
            value = model.predict_one(1, minute, weather, 0)
            assert CONFIG.min_speed_kmh <= value <= CONFIG.max_speed_kmh


def test_model_does_not_overfit_a_held_out_day():
    """Train/test error must stay close - the brief warns about overfitting."""

    from eta_router.evaluation import evaluate_speed_model, split_by_day

    frame = _training_frame()
    train, validation = split_by_day(frame, Config(time_bin_minutes=10, validation_days=2))
    model = SpeedModel(config=CONFIG).fit(train)
    train_error = evaluate_speed_model(model, train).mae_kmh
    validation_error = evaluate_speed_model(model, validation).mae_kmh
    assert validation_error < train_error * 2.0 + 1.0


# --------------------------------------------------------------- submission
def test_route_format_matches_the_brief():
    assert format_route([15, 45, 78, 35]) == "[15,45,78,35]"


def test_route_round_trip():
    assert parse_route(format_route([1, 2, 3])) == [1, 2, 3]
    assert parse_route("[1, 2, 3]") == [1, 2, 3]


def test_validate_accepts_a_good_submission():
    test_cases = pd.DataFrame(
        {
            "src": [15],
            "dest": [35],
            "route_start_t": [600],
            "is_holiday": [0],
            "weather": [2],
        }
    )
    submission = build_submission(test_cases, [345], [[15, 45, 78, 35]])
    assert validate_submission(submission, expected_rows=1) == []
    assert submission.loc[0, "route"] == "[15,45,78,35]"
    assert submission.loc[0, "eta"] == 345


def test_validate_rejects_mismatched_endpoints():
    test_cases = pd.DataFrame(
        {"src": [15], "dest": [35], "route_start_t": [600], "is_holiday": [0], "weather": [2]}
    )
    submission = build_submission(test_cases, [345], [[15, 45, 78, 99]])
    problems = validate_submission(submission, expected_rows=1)
    assert any("endpoints" in p for p in problems)


def test_validate_rejects_a_changed_row_count():
    test_cases = pd.DataFrame(
        {"src": [15], "dest": [35], "route_start_t": [600], "is_holiday": [0], "weather": [2]}
    )
    submission = build_submission(test_cases, [345], [[15, 35]])
    problems = validate_submission(submission, expected_rows=2)
    assert any("row count" in p for p in problems)


def test_build_submission_rejects_wrong_answer_count():
    test_cases = pd.DataFrame(
        {"src": [1, 2], "dest": [3, 4], "route_start_t": [0, 0], "is_holiday": [0, 0], "weather": [0, 0]}
    )
    with pytest.raises(ValueError):
        build_submission(test_cases, [1], [[1, 3]])
