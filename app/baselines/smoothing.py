"""Exponential-smoothing challengers for the FTGM.

Small retail series are dominated by noise; a model that only follows the *level* (and a
damped trend) is often the hardest benchmark to beat. These are the challengers the
forecast service pits against the FTGM on the rolling-origin hold-out:

* :func:`damped_trend_forecast` — Holt's linear method with a damped trend (Gardner &
  McKenzie). With ``phi = 0`` / ``beta = 0`` it reduces to simple exponential smoothing.
  Parameters are picked from a small grid by one-step-ahead in-sample SSE, which is
  cheap, deterministic and robust on 20–100 points.
* :func:`seasonal_damped_forecast` — classical multiplicative decomposition: seasonal
  indices from the ratio to a centred moving average (averaged over the available
  seasons and shrunk towards 1), the damped-trend model on the deseasonalised series,
  then re-seasonalised. Needs at least two full seasons.

All functions return ``(point_forecast, in_sample_one_step_fit)`` and never go negative.
"""
from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.5)
_BETAS = (0.0, 0.05, 0.15)
_PHIS = (0.8, 0.9, 0.98)
#: Shrink seasonal indices towards 1 (noisy seasons on short histories overfit).
_SEASONAL_SHRINK = 0.5


def _holt_path(x: FloatArray, alpha: float, beta: float, phi: float) -> tuple[FloatArray, float, float]:
    level = float(x[0])
    trend = 0.0
    fitted = np.empty(x.size)
    for t in range(x.size):
        fitted[t] = level + phi * trend
        prev = level
        level = alpha * x[t] + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev) + (1 - beta) * phi * trend
    return fitted, level, trend


def _project(level: float, trend: float, phi: float, horizon: int) -> FloatArray:
    steps = np.arange(1, horizon + 1)
    damp = np.array([sum(phi**k for k in range(1, s + 1)) for s in steps], dtype=np.float64)
    return np.clip(level + damp * trend, 0.0, None)


def damped_trend_forecast(demand: FloatArray, horizon: int) -> tuple[FloatArray, FloatArray]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    x = np.asarray(demand, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot forecast from an empty series")
    if x.size < 3:
        level = float(np.mean(x))
        return np.full(horizon, max(level, 0.0)), np.full(x.size, level)

    # Initialise the level on the first few points, not a single noisy observation.
    x0 = np.concatenate([[float(np.mean(x[: min(4, x.size)]))], x])
    best: tuple[float, float, float, float] | None = None
    for alpha, beta, phi in itertools.product(_ALPHAS, _BETAS, _PHIS):
        if beta == 0.0 and phi != _PHIS[0]:
            continue  # SES: phi is irrelevant
        fitted, _, _ = _holt_path(x0, alpha, beta, phi)
        sse = float(np.sum((x - fitted[1:]) ** 2))
        if best is None or sse < best[0]:
            best = (sse, alpha, beta, phi)
    assert best is not None
    _, alpha, beta, phi = best
    fitted, level, trend = _holt_path(x0, alpha, beta, phi)
    return _project(level, trend, phi, horizon), np.clip(fitted[1:], 0.0, None)


def seasonal_indices(x: FloatArray, period: int) -> FloatArray | None:
    """Multiplicative seasonal indices (mean 1) or None when there are < 2 seasons."""
    n = x.size
    if period < 2 or n < 2 * period or float(np.mean(x)) <= 0:
        return None
    # Centred moving average of length `period` (2xMA for even periods).
    kernel = np.ones(period + (period % 2 == 0), dtype=np.float64)
    if period % 2 == 0:
        kernel[0] = kernel[-1] = 0.5
    kernel /= period
    cma = np.convolve(x, kernel, mode="same")
    half = kernel.size // 2
    valid = np.zeros(n, dtype=bool)
    valid[half : n - half] = True
    ratios: list[list[float]] = [[] for _ in range(period)]
    for t in np.flatnonzero(valid):
        if cma[t] > 0:
            ratios[t % period].append(x[t] / cma[t])
    idx = np.array([float(np.median(r)) if r else 1.0 for r in ratios], dtype=np.float64)
    idx = 1.0 + _SEASONAL_SHRINK * (idx - 1.0)
    idx = np.clip(idx, 0.2, 5.0)
    return idx / float(np.mean(idx))


def seasonal_damped_forecast(demand: FloatArray, horizon: int, period: int) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(demand, dtype=np.float64).ravel()
    idx = seasonal_indices(x, period)
    if idx is None:
        raise ValueError("need at least two full seasons")
    n = x.size
    past = idx[np.arange(n) % period]
    future = idx[np.arange(n, n + horizon) % period]
    point, fitted = damped_trend_forecast(x / past, horizon)
    return np.clip(point * future, 0.0, None), np.clip(fitted * past, 0.0, None)
