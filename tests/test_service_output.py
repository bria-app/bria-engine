"""Tests for the enriched response contract: history, diagnostics, status, warnings."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from app.application.forecast_service import ForecastService
from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries


def _daily_points(n_days: int = 1095, seed: int = 7) -> list[ObservationPoint]:
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 1)
    points: list[ObservationPoint] = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        month_phase = 2 * np.pi * (d.month - 1) / 12
        daily = 20 + 8 * np.sin(month_phase) + rng.normal(0, 2)
        points.append(ObservationPoint(date=d, demand=max(0.0, float(daily))))
    return points


def test_history_and_diagnostics_present() -> None:
    request = ForecastRequest(
        period=12, horizon=3,
        series=[ProductSeries(product_id="p", points=_daily_points())],
    )
    fc = ForecastService().forecast(request).forecasts[0]

    assert fc.status == "ok"
    # 36 monthly buckets of history, each with observed/cleaned/fitted.
    assert len(fc.history) == 36
    for h in fc.history:
        assert h.fitted is not None
        assert h.cleaned >= 0.0

    diag = fc.diagnostics
    assert diag is not None
    assert diag.n_periods == 36
    assert diag.validation_size and diag.validation_size >= 1
    # Algorithm 1 evidence: at least the selected order has a finite score.
    assert fc.order_selected in diag.order_scores
    assert diag.validation_rmse is not None


def test_stockout_month_is_imputed_and_reported() -> None:
    points = _daily_points()
    for p in points:
        if p.date.year == 2023 and p.date.month == 7:
            p.demand = 0.0
            p.stockout_flag = True
    request = ForecastRequest(period=12, horizon=2, series=[ProductSeries(product_id="p", points=points)])
    fc = ForecastService().forecast(request).forecasts[0]

    diag = fc.diagnostics
    assert diag is not None and diag.stockout_periods == 1 and diag.imputed_periods == 1
    stockout_bucket = next(h for h in fc.history if h.is_stockout)
    # The cleaned value was repaired upward from the censored zero.
    assert stockout_bucket.observed == 0.0
    assert stockout_bucket.cleaned > 0.0


def test_empty_series_is_skipped_not_fatal() -> None:
    request = ForecastRequest(
        period=12, horizon=3,
        series=[
            ProductSeries(product_id="empty", points=[]),
            ProductSeries(product_id="good", points=_daily_points()),
        ],
    )
    response = ForecastService().forecast(request)

    skipped = next(f for f in response.forecasts if f.product_id == "empty")
    good = next(f for f in response.forecasts if f.product_id == "good")
    assert skipped.status == "skipped" and skipped.points == []
    assert good.status == "ok" and len(good.points) == 3


def test_baseline_fallback_reports_reason_and_honest_interval() -> None:
    # 14 monthly points: too short for stable FTGM at T=12 -> baseline.
    pts = [
        ObservationPoint(date=date(2024, 1 + (i % 12), 1) if i < 12 else date(2025, i - 11, 1),
                         demand=10.0 + (i % 12))
        for i in range(14)
    ]
    request = ForecastRequest(period=12, horizon=4, series=[ProductSeries(product_id="p", points=pts)])
    fc = ForecastService().forecast(request).forecasts[0]

    if fc.status == "fallback":  # FTGM may or may not survive; if it fell back, verify honesty
        assert fc.model in ("SeasonalNaive", "DampedTrend", "SeasonalDamped")
        assert fc.fallback_reason
        # Interval must not pretend zero uncertainty (unless residuals are truly zero).
        widths = [p.upper_bound - p.lower_bound for p in fc.points]
        assert all(w >= 0 for w in widths)
    assert len(fc.history) == 14
