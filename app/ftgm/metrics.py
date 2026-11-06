"""Forecast accuracy metrics.

Two families are provided:

* **Scale-dependent** (``mae``, ``rmse``): expressed in the same units as demand.
  Easy to interpret for a single product and what a store manager cares about, but
  not comparable across products of different magnitudes.
* **Scale-independent** (``mape``, ``mase``, ``rmsse``): normalised errors that can be
  averaged across many products. ``mase`` / ``rmsse`` are the M5-competition metrics
  used in the FTGM paper (Ye et al., 2024) and scale the error by the in-sample naive
  forecast error, which makes them robust even when demand contains zeros.

All functions operate on plain ``float`` arrays and never mutate their inputs.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
    "mean_absolute_scaled_error",
    "root_mean_squared_scaled_error",
]


def _as_pair(actual: FloatArray, predicted: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Validate and align an (actual, predicted) pair."""
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if a.shape != p.shape:
        raise ValueError(f"actual and predicted must share a shape, got {a.shape} vs {p.shape}")
    if a.size == 0:
        raise ValueError("cannot compute a metric on empty arrays")
    return a, p


def mean_absolute_error(actual: FloatArray, predicted: FloatArray) -> float:
    """MAE — average absolute error, in demand units."""
    a, p = _as_pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def root_mean_squared_error(actual: FloatArray, predicted: FloatArray) -> float:
    """RMSE — penalises large misses more than MAE, in demand units."""
    a, p = _as_pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mean_absolute_percentage_error(actual: FloatArray, predicted: FloatArray) -> float:
    """MAPE as a percentage.

    Periods where the actual demand is zero are excluded, because the percentage error
    is undefined there (a common pitfall with intermittent retail demand). Returns
    ``nan`` if every actual value is zero.
    """
    a, p = _as_pair(actual, predicted)
    mask = a != 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


def _naive_scale_mae(insample: FloatArray, period: int) -> float:
    """Mean absolute error of the in-sample one-step (or seasonal) naive forecast.

    This is the denominator shared by MASE/RMSSE. For seasonal data the seasonal naive
    (lag = ``period``) is the natural benchmark; we fall back to the lag-1 naive when
    the in-sample history is shorter than one season.
    """
    y = np.asarray(insample, dtype=np.float64)
    lag = period if (period > 0 and y.size > period) else 1
    diffs = np.abs(y[lag:] - y[:-lag])
    scale = float(np.mean(diffs)) if diffs.size else 0.0
    return scale


def mean_absolute_scaled_error(
    actual: FloatArray, predicted: FloatArray, insample: FloatArray, period: int = 1
) -> float:
    """MASE — MAE scaled by the in-sample naive MAE. Lower is better; 1.0 ties naive."""
    a, p = _as_pair(actual, predicted)
    scale = _naive_scale_mae(insample, period)
    if scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(a - p)) / scale)


def root_mean_squared_scaled_error(
    actual: FloatArray, predicted: FloatArray, insample: FloatArray, period: int = 1
) -> float:
    """RMSSE — the M5 metric: RMSE scaled by the in-sample naive RMSE."""
    a, p = _as_pair(actual, predicted)
    y = np.asarray(insample, dtype=np.float64)
    lag = period if (period > 0 and y.size > period) else 1
    naive_mse = float(np.mean((y[lag:] - y[:-lag]) ** 2)) if y.size > lag else 0.0
    if naive_mse == 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((a - p) ** 2) / naive_mse))
