"""Data-driven Fourier-order selection (Algorithm 1 of Ye et al., 2024).

The Fourier order ``N`` is the only hyper-parameter of the FTGM: too low and the model
cannot describe the seasonal shape; too high and it overfits the noise. The paper picks
``N`` with a validation-set strategy:

    1. Hold out the last ``l`` periods of the in-sample history as a validation set.
    2. For every candidate order, fit on the training part and forecast ``l`` steps.
    3. Keep the order with the lowest **validation RMSE** (RMSE is chosen because it is
       consistent with the least-squares loss used for estimation).

The candidate set is bounded above by the Nyquist-Shannon limit ``N < T / (2h)`` and,
in practice, by how many parameters the (short) training series can identify.

Extension: the same validation also picks the ridge penalty of the estimation from
``ridges`` (0 = the paper's OLS). Clean seasonal series keep 0; short noisy MYPE series
need shrinkage, and the validation set decides which case a product is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from app.ftgm import fourier
from app.ftgm.exceptions import FTGMError, InsufficientDataError
from app.ftgm.metrics import root_mean_squared_error
from app.ftgm.model import FTGM, FTGMConfig

FloatArray = NDArray[np.float64]


@dataclass
class OrderSelection:
    """Result of :func:`select_order`."""

    best_order: int
    scores: dict[int, float]  # validation RMSE per candidate order (nan = failed/skipped)
    validation_size: int
    #: Ridge penalty chosen jointly with the order (see ``FTGMConfig.ridge``).
    best_ridge: float = 0.0


def _nyquist_cap(period: int) -> int:
    """Upper bound on the order from the sampling theorem: ``ceil(T / 2) - 1`` (h = 1)."""
    return max(1, math.ceil(period / 2) - 1)


def _default_validation_size(n_obs: int, period: int) -> int:
    """A modest hold-out: about **half a season**, but never more than a quarter of the data.

    Mirrors the paper's setup (6 validation months for monthly data, T = 12).
    """
    half_season = max(1, period // 2)
    return max(1, min(half_season, n_obs // 4))


def select_order(
    demand: FloatArray,
    config: FTGMConfig | None = None,
    *,
    validation_size: int | None = None,
    max_order: int | None = None,
    ridges: tuple[float, ...] | None = None,
) -> OrderSelection:
    """Choose the best Fourier order for ``demand`` via the validation strategy."""
    cfg = config or FTGMConfig()
    x = np.asarray(demand, dtype=np.float64).ravel()
    n = x.size

    # Even the smallest model (N = 1) must be identifiable on the full series.
    if n - 1 < fourier.n_params(1):
        raise InsufficientDataError(
            f"need at least {fourier.n_params(1) + 1} observations, got {n}"
        )

    val = validation_size if validation_size is not None else _default_validation_size(n, cfg.period)
    train_len = n - val
    # Ensure the training part can still identify an order-1 model; shrink the hold-out
    # if necessary. If even that is impossible, fall back to order 1 without validation.
    min_train = fourier.n_params(1) + 1
    if train_len < min_train:
        val = max(0, n - min_train)
        train_len = n - val
    if val < 1:
        return OrderSelection(best_order=1, scores={1: float("nan")}, validation_size=0)

    train, valid = x[:train_len], x[train_len:]

    nyquist = _nyquist_cap(cfg.period)
    # Identifiability: fitting order N needs (train_len - 1) >= n_params(N) = 2 + 4N,
    # i.e. N <= (train_len - 3) / 4.
    data_cap = (train_len - 3) // 4
    upper = min(nyquist, data_cap, max_order if max_order is not None else nyquist)
    if upper < 1:
        return OrderSelection(best_order=1, scores={1: float("nan")}, validation_size=val)

    grid = ridges if ridges else (cfg.ridge,)
    scores: dict[int, float] = {}
    best_ridge: dict[int, float] = {}
    for order in range(1, upper + 1):
        scores[order] = float("nan")  # unstable for every penalty -> out of the running
        for ridge in grid:
            try:
                model = FTGM(order=order, config=replace(cfg, ridge=ridge)).fit(train)
                rmse = root_mean_squared_error(valid, model.predict(val).point)
            except FTGMError:
                continue
            if math.isfinite(rmse) and not rmse >= scores[order]:
                scores[order], best_ridge[order] = rmse, ridge

    finite = {k: v for k, v in scores.items() if math.isfinite(v)}
    best_order = min(finite, key=lambda k: finite[k]) if finite else 1
    return OrderSelection(
        best_order=best_order,
        scores=scores,
        validation_size=val,
        best_ridge=best_ridge.get(best_order, cfg.ridge),
    )
