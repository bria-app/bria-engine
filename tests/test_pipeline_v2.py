"""Tests for the frequency-aware pipeline: auto frequency, cut-off, censored demand,
outliers, intermittent demand, rolling-origin validation and diagnostics."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from app.application.forecast_service import ForecastService
from app.baselines.intermittent import croston_sba
from app.ftgm import cleaning
from app.ftgm import preprocessing as prep
from app.ftgm.validation import rolling_origin
from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries


def _daily(start: date, days: int, base: float = 20.0, amp: float = 8.0, seed: int = 11) -> list[ObservationPoint]:
    rng = np.random.default_rng(seed)
    pts: list[ObservationPoint] = []
    for i in range(days):
        d = start + timedelta(days=i)
        v = base + amp * np.sin(2 * np.pi * (d.month - 1) / 12) + rng.normal(0, 2)
        pts.append(ObservationPoint(date=d, demand=max(0.0, round(float(v)))))
    return pts


def _request(points: list[ObservationPoint], **kw: object) -> ForecastRequest:
    return ForecastRequest(series=[ProductSeries(product_id="p", points=points)], **kw)  # type: ignore[arg-type]


def test_bucketize_excludes_period_in_progress_and_fills_trailing_zeros() -> None:
    dates = [date(2025, 1, 10), date(2025, 3, 5), date(2025, 6, 2)]
    demand = np.array([5.0, 7.0, 9.0])
    out = prep.bucketize(dates, demand, np.zeros(3, dtype=bool), 12, as_of=date(2025, 6, 15))
    # June is in progress -> dropped; Jan..May contiguous (Feb/Apr/May zero-filled).
    assert out.dates == [date(2025, m, 1) for m in range(1, 6)]
    np.testing.assert_allclose(out.demand, [5.0, 0.0, 7.0, 0.0, 0.0])
    assert out.dropped_incomplete


def test_partial_stockout_scales_up_and_full_stockout_interpolates() -> None:
    demand = np.array([100.0, 50.0, 0.0, 110.0])
    share = np.array([0.0, 0.5, 1.0, 0.0])
    repaired, flags, imputed = prep.repair_censored(demand, share)
    assert flags.tolist() == [False, True, True, False]
    assert repaired[1] == 100.0  # 50 sold in half the month -> ~100 demand
    assert 100.0 <= repaired[2] <= 110.0  # interpolated
    assert imputed.sum() == 2


def test_sparse_daily_stockout_share_uses_calendar_days() -> None:
    # Only 2 observations in March (a sale and a 1-day stock-out): still daily data.
    dates = [date(2025, 3, 3), date(2025, 3, 20), date(2025, 4, 1)]
    out = prep.bucketize(dates, np.array([10.0, 0.0, 5.0]), np.array([False, True, False]), 12)
    assert abs(out.stockout_share[0] - 1 / 31) < 1e-9


def test_hampel_removes_spike_but_keeps_seasonal_peak() -> None:
    t = np.arange(48)
    x = 100 + 40 * np.sin(2 * np.pi * t / 12)
    x[30] += 400  # one-off bulk order
    cleaned, mask = cleaning.hampel_clean(x, 12)
    assert mask[30]
    assert mask.sum() <= 2
    assert abs(cleaned[30] - (100 + 40 * np.sin(2 * np.pi * 30 / 12))) < 40
    assert (cleaning.seasonality_strength(cleaned, 12) or 0) > 0.8


def test_croston_forecasts_flat_positive_rate() -> None:
    x = np.array([0, 0, 5, 0, 0, 0, 4, 0, 0, 6, 0, 0], dtype=float)
    point, fitted = croston_sba(x, 4)
    assert np.allclose(point, point[0]) and point[0] > 0
    assert fitted.shape == x.shape


def test_rolling_origin_scores_perfect_forecaster() -> None:
    x = np.arange(1, 31, dtype=float)
    score = rolling_origin(x, lambda tr, h: tr[-1] + np.arange(1, h + 1), horizon=3, n_origins=3, min_train=8, period=12)
    assert score is not None and score.origins == 3 and score.rmse < 1e-9


def test_auto_frequency_monthly_with_long_history() -> None:
    pts = _daily(date(2022, 1, 1), 1200)
    as_of = date(2022, 1, 1) + timedelta(days=1200)
    fc = ForecastService().forecast(_request(pts, frequency="auto", horizon_days=90, as_of=as_of)).forecasts[0]
    assert fc.frequency == "monthly" and fc.period == 12
    assert len(fc.points) == 3
    d = fc.diagnostics
    assert d is not None and d.holdout is not None and d.explanations
    # The FTGM took part in the tournament and the winner is the most accurate contender.
    assert any(c.model == "FTGM" for c in d.candidates)
    chosen = next(c for c in d.candidates if c.chosen)
    assert chosen.mae is not None and chosen.mae <= 1.03 * min(c.mae for c in d.candidates if c.mae is not None)
    assert d.accuracy_pct is not None and 0 <= d.accuracy_pct <= 100
    assert d.seasonality_strength is not None
    assert all(p.predicted_demand >= 0 and p.lower_bound <= p.predicted_demand <= p.upper_bound for p in fc.points)


def test_auto_frequency_weekly_for_medium_history() -> None:
    pts = _daily(date(2025, 1, 6), 300)  # ~10 months -> weekly
    as_of = date(2025, 1, 6) + timedelta(days=300)
    fc = ForecastService().forecast(_request(pts, frequency="auto", horizon_days=30, as_of=as_of)).forecasts[0]
    assert fc.frequency == "weekly" and fc.period == 52
    assert len(fc.points) == 5
    assert fc.status in ("ok", "fallback")


def test_auto_frequency_short_history_uses_moving_average() -> None:
    pts = _daily(date(2025, 1, 6), 60)
    as_of = date(2025, 1, 6) + timedelta(days=60)
    fc = ForecastService().forecast(_request(pts, frequency="auto", horizon_days=30, as_of=as_of)).forecasts[0]
    assert fc.model == "MovingAverage" and fc.status == "fallback"
    assert fc.fallback_reason and "semanas" in fc.fallback_reason


def test_intermittent_demand_uses_croston() -> None:
    start = date(2023, 1, 1)
    pts = [ObservationPoint(date=start + timedelta(days=i), demand=3.0) for i in range(0, 1100, 75)]
    fc = ForecastService().forecast(
        _request(pts, frequency="monthly", horizon_days=90, as_of=start + timedelta(days=1100))
    ).forecasts[0]
    assert fc.model == "CrostonSBA"
    assert fc.diagnostics is not None and fc.diagnostics.intermittent


def test_only_current_period_is_skipped() -> None:
    pts = [ObservationPoint(date=date(2026, 9, 3), demand=4.0)]
    fc = ForecastService().forecast(
        _request(pts, frequency="monthly", horizon_days=30, as_of=date(2026, 9, 12))
    ).forecasts[0]
    assert fc.status == "skipped"
