"""Model tournament, ridge-regularised FTGM and the smoothing challengers."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from app.application.forecast_service import ForecastService
from app.baselines.smoothing import damped_trend_forecast, seasonal_damped_forecast, seasonal_indices
from app.ftgm import FTGM, FTGMConfig, select_order
from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries


def _noisy_retail(months: int = 24, seed: int = 11) -> np.ndarray:
    """Small MYPE-like counts: December peak, growth and Poisson noise."""
    rng = np.random.default_rng(seed)
    curve = np.array([0.8, 0.8, 0.9, 0.95, 1.25, 1.3, 1.15, 0.95, 0.95, 1.0, 1.25, 1.7])
    t = np.arange(months)
    lam = 25 * curve[t % 12] * (0.8 + 0.4 * t / months)
    return rng.poisson(lam).astype(np.float64)


def test_damped_trend_tracks_a_flat_level() -> None:
    point, fitted = damped_trend_forecast(np.full(30, 12.0), 4)
    assert np.allclose(point, 12.0, atol=0.5)
    assert fitted.shape == (30,)


def test_seasonal_indices_average_to_one() -> None:
    x = _noisy_retail(36)
    idx = seasonal_indices(x, 12)
    assert idx is not None and abs(float(np.mean(idx)) - 1.0) < 1e-9
    assert seasonal_indices(x[:20], 12) is None  # < 2 seasons
    point, _ = seasonal_damped_forecast(x, 6, 12)
    assert point.shape == (6,) and np.all(point >= 0)


def test_ridge_is_chosen_by_validation_and_never_singular() -> None:
    sel = select_order(_noisy_retail(), FTGMConfig(period=12), max_order=3, ridges=(0.0, 3.0))
    assert sel.best_ridge in (0.0, 3.0)
    # An all-zero series would make the normal equations singular; the ridge solve copes.
    model = FTGM(order=1, config=FTGMConfig(period=12, ridge=3.0)).fit(np.zeros(20))
    assert np.all(np.isfinite(model.predict(3).point))


def test_tournament_reports_candidates_and_plain_accuracy() -> None:
    x = _noisy_retail()
    start = date(2024, 9, 1)
    points, day = [], start
    for m, units in enumerate(x):
        points.append(ObservationPoint(date=day, demand=float(units)))
        day = (day.replace(day=1) + timedelta(days=32)).replace(day=1)
    fc = ForecastService().forecast(
        ForecastRequest(
            series=[ProductSeries(product_id="p", points=points)],
            frequency="monthly",
            horizon_days=30,
            as_of=day,
        )
    ).forecasts[0]
    d = fc.diagnostics
    assert d is not None
    names = {c.model for c in d.candidates}
    assert {"FTGM", "FTGMCombo", "DampedTrend", "SeasonalNaive"} <= names
    assert sum(c.chosen for c in d.candidates) == 1
    assert d.accuracy_pct is not None and 0 <= d.accuracy_pct <= 100
    assert fc.status == ("ok" if fc.model in ("FTGM", "FTGMCombo") else "fallback")
