"""Unit tests for the accuracy metrics."""
from __future__ import annotations

import math

import numpy as np

from app.ftgm import metrics as M


def test_perfect_forecast_has_zero_error() -> None:
    y = np.array([10.0, 20.0, 30.0])
    assert M.mean_absolute_error(y, y) == 0.0
    assert M.root_mean_squared_error(y, y) == 0.0
    assert M.mean_absolute_percentage_error(y, y) == 0.0


def test_mae_and_rmse_known_values() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 5.0])  # single error of 2
    assert math.isclose(M.mean_absolute_error(actual, predicted), 2.0 / 3.0)
    assert math.isclose(M.root_mean_squared_error(actual, predicted), math.sqrt(4.0 / 3.0))


def test_mape_skips_zeros() -> None:
    actual = np.array([0.0, 50.0])
    predicted = np.array([5.0, 55.0])  # only the second point counts: 10%
    assert math.isclose(M.mean_absolute_percentage_error(actual, predicted), 10.0)


def test_mape_all_zero_is_nan() -> None:
    actual = np.zeros(3)
    assert math.isnan(M.mean_absolute_percentage_error(actual, np.ones(3)))


def test_scaled_metrics_against_naive() -> None:
    insample = np.array([10.0, 12.0, 11.0, 13.0])
    actual = np.array([14.0])
    predicted = np.array([13.0])  # abs error 1
    mase = M.mean_absolute_scaled_error(actual, predicted, insample, period=1)
    assert math.isfinite(mase) and mase > 0
