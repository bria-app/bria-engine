"""Baselines for demand the FTGM is not designed for.

* :func:`croston_sba` — Croston's method with the Syntetos-Boylan bias correction, the
  standard forecaster for **intermittent** demand (most periods without sales). It
  smooths the size of the non-zero demands and the interval between them separately and
  forecasts a flat *demand rate* ``(1 - alpha/2) * z / p``.
* :func:`moving_average_forecast` — a trailing mean for very short histories, where no
  seasonal model can be identified yet.

Both return ``(point_forecast, in_sample_one_step_fit)``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def croston_sba(demand: FloatArray, horizon: int, alpha: float = 0.1) -> tuple[FloatArray, FloatArray]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    x = np.asarray(demand, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("cannot forecast from an empty series")
    nonzero = np.flatnonzero(x > 0)
    if nonzero.size == 0:
        return np.zeros(horizon), np.zeros(n)

    correction = 1.0 - alpha / 2.0
    first = int(nonzero[0])
    # Initialise from the whole series, not just the first event: the mean demand size
    # and the mean inter-demand interval (n / #events). Starting from the first event
    # alone made a product with one early sale forecast that rate forever (p started
    # at 1 and, with no further events, never updated).
    z = float(np.mean(x[nonzero]))  # smoothed non-zero demand size
    p = float(n / nonzero.size)  # smoothed inter-demand interval
    q = 1  # periods since the last demand
    fitted = np.empty(n)
    fitted[: first + 1] = correction * z / p
    for t in range(first + 1, n):
        fitted[t] = correction * z / p
        if x[t] > 0:
            z += alpha * (x[t] - z)
            p += alpha * (q - p)
            q = 1
        else:
            q += 1
    rate = correction * z / p
    return np.full(horizon, rate), fitted


def moving_average_forecast(demand: FloatArray, horizon: int, window: int) -> tuple[FloatArray, FloatArray]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    x = np.asarray(demand, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot forecast from an empty series")
    w = max(1, min(window, x.size))
    level = float(np.mean(x[-w:]))
    fitted = np.empty(x.size)
    for t in range(x.size):
        past = x[max(0, t - w) : t]
        fitted[t] = float(np.mean(past)) if past.size else x[0]
    return np.full(horizon, level), fitted
