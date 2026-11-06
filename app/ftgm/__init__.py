"""Fourier Time-Varying Grey Model — pure numerical core.

This package has no web or serialization dependencies so it can be tested and reused on
its own. Public surface:

* :class:`~app.ftgm.model.FTGM` / :class:`~app.ftgm.model.FTGMConfig` — the model.
* :func:`~app.ftgm.order_selection.select_order` — data-driven Fourier order choice.
* :mod:`~app.ftgm.metrics` — accuracy metrics.
* :mod:`~app.ftgm.preprocessing` — aggregation and stock-out repair.
"""
from app.ftgm.exceptions import (
    FTGMError,
    InsufficientDataError,
    ModelNotFittedError,
    SolverError,
)
from app.ftgm.model import FTGM, FTGMConfig, ForecastResult
from app.ftgm.order_selection import OrderSelection, select_order

__all__ = [
    "FTGM",
    "FTGMConfig",
    "ForecastResult",
    "OrderSelection",
    "select_order",
    "FTGMError",
    "InsufficientDataError",
    "ModelNotFittedError",
    "SolverError",
]
