"""Unit tests for the FTGM model and order selection."""
from __future__ import annotations

import numpy as np
import pytest

from app.baselines import seasonal_naive_forecast
from app.ftgm import FTGM, FTGMConfig, InsufficientDataError, ModelNotFittedError, select_order
from app.ftgm import metrics as M


def _seasonal_series(n: int = 48, seed: int = 7) -> np.ndarray:
    """A trending series with time-varying seasonality (12-month cycle)."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1, dtype=np.float64)
    season = 30.0 * np.sin(2 * np.pi * t / 12) + 12.0 * np.cos(2 * np.pi * t / 12)
    trend = 100.0 + 1.5 * t
    noise = rng.normal(0.0, 3.0, size=n)
    return np.clip(trend + season + noise, 0.0, None)


def test_fit_produces_close_in_sample_fit() -> None:
    x = _seasonal_series()
    model = FTGM(order=2, config=FTGMConfig(period=12)).fit(x)

    assert model.fitted_ is not None and model.fitted_.shape == x.shape
    rmse = M.root_mean_squared_error(x, model.fitted_)
    assert rmse / float(np.mean(x)) < 0.15  # fit within ~15% of the series scale


def test_ftgm_beats_seasonal_naive_in_sample() -> None:
    x = _seasonal_series()
    model = FTGM(order=2, config=FTGMConfig(period=12)).fit(x)
    assert model.fitted_ is not None

    season = 12
    naive_fit = np.concatenate([x[:season], x[:-season]])
    ftgm_rmse = M.root_mean_squared_error(x[season:], model.fitted_[season:])
    naive_rmse = M.root_mean_squared_error(x[season:], naive_fit[season:])
    assert ftgm_rmse < naive_rmse


def test_predict_returns_ordered_nonneg_interval() -> None:
    x = _seasonal_series()
    result = FTGM(order=2, config=FTGMConfig(period=12)).fit(x).predict(horizon=6)

    assert result.point.shape == (6,)
    assert np.all(np.isfinite(result.point))
    assert np.all(result.point >= 0.0)
    assert np.all(result.lower <= result.point + 1e-9)
    assert np.all(result.point <= result.upper + 1e-9)


def test_order_selection_picks_valid_order() -> None:
    x = _seasonal_series(n=60)
    selection = select_order(x, FTGMConfig(period=12))
    assert 1 <= selection.best_order <= 5
    assert selection.best_order in selection.scores


def test_short_series_raises() -> None:
    with pytest.raises(InsufficientDataError):
        FTGM(order=1, config=FTGMConfig(period=12)).fit(np.arange(5, dtype=np.float64))


def test_predict_before_fit_raises() -> None:
    with pytest.raises(ModelNotFittedError):
        FTGM(order=1).predict(3)


def test_invalid_order_rejected() -> None:
    with pytest.raises(ValueError):
        FTGM(order=0)


def test_seasonal_naive_repeats_last_cycle() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = seasonal_naive_forecast(x, horizon=5, period=4)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0, 1.0])
