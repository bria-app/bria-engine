"""End-to-end tests for the forecast service (the HTTP contract pipeline)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from app.application.forecast_service import ForecastService
from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries


def _daily_points(n_days: int = 1095, seed: int = 3) -> list[ObservationPoint]:
    """~3 years of daily demand with a 12-month seasonal pattern."""
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 1)
    points: list[ObservationPoint] = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        month_phase = 2 * np.pi * (d.month - 1) / 12
        daily = 20 + 8 * np.sin(month_phase) + rng.normal(0, 2)
        points.append(ObservationPoint(date=d, demand=max(0.0, float(daily))))
    return points


def test_forecast_batch_pipeline() -> None:
    request = ForecastRequest(
        period=12,
        horizon=6,
        series=[ProductSeries(product_id="prod-1", points=_daily_points())],
    )
    response = ForecastService().forecast(request)

    assert response.period == 12
    assert len(response.forecasts) == 1
    fc = response.forecasts[0]
    assert fc.product_id == "prod-1"
    assert len(fc.points) == 6
    # Enough data -> the FTGM competes; whichever wins, the FTGM order was identified.
    assert fc.diagnostics is not None and any(c.model == "FTGM" for c in fc.diagnostics.candidates)
    assert fc.diagnostics.order_scores
    assert fc.metrics.rmse is not None

    # Forecast dates advance monthly and the interval is well-ordered.
    months = {p.date.month for p in fc.points}
    assert len(fc.points) == 6 and len(months) == 6
    for p in fc.points:
        assert p.lower_bound <= p.predicted_demand <= p.upper_bound


def test_short_series_falls_back_to_baseline() -> None:
    pts = [ObservationPoint(date=date(2025, 1, 1) , demand=10.0),
           ObservationPoint(date=date(2025, 2, 1), demand=12.0),
           ObservationPoint(date=date(2025, 3, 1), demand=11.0)]
    request = ForecastRequest(period=12, horizon=3, series=[ProductSeries(product_id="p", points=pts)])

    response = ForecastService().forecast(request)
    fc = response.forecasts[0]
    assert fc.order_selected == 0  # baseline fallback
    assert fc.model in ("SeasonalNaive", "DampedTrend")
    assert fc.status == "fallback"
    assert len(fc.points) == 3


def test_stockout_periods_do_not_break_pipeline() -> None:
    points = _daily_points()
    # Flag a whole month as stock-out with zeroed demand.
    for p in points:
        if p.date.year == 2023 and p.date.month == 7:
            p.demand = 0.0
            p.stockout_flag = True
    request = ForecastRequest(period=12, horizon=4, series=[ProductSeries(product_id="p", points=points)])

    response = ForecastService().forecast(request)
    assert len(response.forecasts[0].points) == 4
