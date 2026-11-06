"""Seasonal naive baseline.

The seasonal naive repeats the value observed one full season ago:
``x_hat(m + h) = x(m + h - T)``. It is the standard, hard-to-beat reference for seasonal
series and the natural sanity check for the FTGM — if the model cannot beat seasonal
naive on validation data, something is wrong. When the history is shorter than one
season it degrades gracefully to the plain naive (repeat the last value).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def seasonal_naive_forecast(demand: FloatArray, horizon: int, period: int) -> FloatArray:
    """Forecast ``horizon`` periods ahead by repeating the last seasonal cycle."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    x = np.asarray(demand, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot forecast from an empty series")

    season = period if x.size >= period > 0 else 1
    last_cycle = x[-season:]
    # Tile the last cycle out to the requested horizon.
    reps = int(np.ceil(horizon / season))
    return np.tile(last_cycle, reps)[:horizon]
