"""Rolling-origin (time-series cross-validation) evaluation.

In-sample fit metrics flatter any model. To report an *honest* accuracy we re-fit the
model at several forecast origins at the end of the history and score the next ``h``
periods it had not seen:

    origin i:  train = x[: n - h - i]      test = x[n - h - i : n - i]      (i = 0..k-1)

The same origins are used for the seasonal-naive benchmark, so the two are directly
comparable (``skill`` = relative RMSE improvement over naive). Per-step RMSE also drives
the empirical prediction interval of the final forecast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from app.ftgm.exceptions import FTGMError

FloatArray = NDArray[np.float64]
Forecaster = Callable[[FloatArray, int], FloatArray]


@dataclass
class HoldoutScore:
    origins: int
    horizon: int
    mae: float
    rmse: float
    mape: float | None
    mase: float | None
    rmse_by_step: list[float] = field(default_factory=list)
    #: Sum |error| / sum actual over every scored period (robust MAPE for small counts).
    wape: float | None = None
    #: Same, on the *total* of each origin's horizon — the number replenishment cares
    #: about ("how many units will I sell next month"), far less noisy than per period.
    total_wape: float | None = None
    #: Units actually sold over every scored period. Tiny totals make percentage errors
    #: meaningless (sell 3, predict 7 → "133% off"), so accuracy_pct hides behind it.
    holdout_units: float = 0.0

    #: Below this many units sold in the holdout, a percentage says nothing.
    MIN_UNITS_TO_SCORE = 10.0

    @property
    def accuracy_pct(self) -> float | None:
        """Plain-language accuracy: 100 - total WAPE, clipped to [0, 100].

        None when the holdout barely had sales — a relative error over a handful of
        units is noise, and reporting "0%" there reads as "broken" instead of
        "not enough history yet".
        """
        if self.total_wape is None or self.holdout_units < self.MIN_UNITS_TO_SCORE:
            return None
        return max(0.0, min(100.0, 100.0 - self.total_wape))


def _naive_scale(train: FloatArray, period: int) -> float:
    lag = period if train.size > period else 1
    if train.size <= lag:
        return 0.0
    return float(np.mean(np.abs(train[lag:] - train[:-lag])))


def rolling_origin(
    x: FloatArray,
    forecaster: Forecaster,
    *,
    horizon: int,
    n_origins: int,
    min_train: int,
    period: int,
) -> HoldoutScore | None:
    """Score ``forecaster`` on the last ``n_origins`` origins; None if none is feasible."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    h = max(1, horizon)
    errors: list[FloatArray] = []
    actuals: list[FloatArray] = []
    scales: list[float] = []
    for i in range(n_origins):
        train_len = n - h - i
        if train_len < min_train:
            break
        train, test = x[:train_len], x[train_len : train_len + h]
        try:
            pred = np.asarray(forecaster(train, h), dtype=np.float64)[:h]
        except (FTGMError, ValueError, np.linalg.LinAlgError):
            continue
        if pred.size != h or not np.all(np.isfinite(pred)):
            continue
        pred = np.clip(pred, 0.0, None)
        errors.append(test - pred)
        actuals.append(test)
        scales.append(_naive_scale(train, period))
    if not errors:
        return None

    err = np.vstack(errors)
    act = np.vstack(actuals)
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    nz = act != 0
    mape = float(np.mean(abs_err[nz] / act[nz]) * 100.0) if nz.any() else None
    scale = float(np.mean(scales)) if scales else 0.0
    mase = mae / scale if scale > 0 else None
    by_step = [float(np.sqrt(np.mean(err[:, s] ** 2))) for s in range(h)]
    act_sum = float(np.sum(act))
    wape = float(np.sum(abs_err) / act_sum * 100.0) if act_sum > 0 else None
    tot_act = act.sum(axis=1)
    tot_err = np.abs(err.sum(axis=1))
    total_wape = float(np.sum(tot_err) / np.sum(tot_act) * 100.0) if float(np.sum(tot_act)) > 0 else None

    def fin(v: float | None) -> float | None:
        return v if v is not None and math.isfinite(v) else None

    return HoldoutScore(
        origins=len(errors),
        horizon=h,
        mae=mae,
        rmse=rmse,
        mape=fin(mape),
        mase=fin(mase),
        rmse_by_step=by_step,
        wape=fin(wape),
        total_wape=fin(total_wape),
        holdout_units=act_sum,
    )
