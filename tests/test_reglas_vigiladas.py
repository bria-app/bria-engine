"""Engine rules the acceptance criteria promise, each pinned at its boundary."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from app.application.forecast_service import ForecastService
from app.ftgm import preprocessing as prep
from app.ftgm.validation import HoldoutScore
from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries


def _months(values: list[float], start: date = date(2023, 1, 1)) -> tuple[list[ObservationPoint], date]:
    points, day = [], start
    for units in values:
        points.append(ObservationPoint(date=day, demand=float(units)))
        day = (day.replace(day=1) + timedelta(days=32)).replace(day=1)
    return points, day


def _forecast(values: list[float]):  # noqa: ANN202
    points, as_of = _months(values)
    return ForecastService().forecast(
        ForecastRequest(
            series=[ProductSeries(product_id="p", points=points)],
            frequency="monthly",
            horizon_days=30,
            as_of=as_of,
        )
    ).forecasts[0]


# --- Sold-out periods (HU0020) ------------------------------------------------------------------
def test_sixty_percent_of_days_sold_out_is_a_full_stockout() -> None:
    repaired, _flags, _imputed = prep.repair_censored(np.array([100.0, 50.0, 100.0]), np.array([0.0, 0.60, 0.0]))
    assert repaired[1] == 100.0  # interpolated between its neighbours, not scaled up from 50


def test_just_under_sixty_percent_is_scaled_from_what_sold() -> None:
    repaired, _flags, _imputed = prep.repair_censored(np.array([100.0, 50.0, 100.0]), np.array([0.0, 0.59, 0.0]))
    assert abs(repaired[1] - 50.0 / 0.41) < 1e-9


def test_scaling_is_capped_at_twice_the_interpolated_value() -> None:
    # 20 sold in half the month would be 40, but its neighbours say about 5: at most 2 x 5, and
    # never less than what really sold.
    repaired, _flags, _imputed = prep.repair_censored(np.array([5.0, 20.0, 5.0]), np.array([0.0, 0.5, 0.0]))
    assert repaired[1] == 20.0
    repaired, _flags, _imputed = prep.repair_censored(np.array([10.0, 12.0, 10.0]), np.array([0.0, 0.5, 0.0]))
    assert repaired[1] == 20.0  # 24 without the cap


def test_a_period_counts_as_sold_out_from_ten_percent_of_its_days() -> None:
    _repaired, flags, _imputed = prep.repair_censored(np.array([100.0, 90.0, 100.0]), np.array([0.0, 0.10, 0.0]))
    assert flags.tolist() == [False, True, False]
    _repaired, flags, _imputed = prep.repair_censored(np.array([100.0, 90.0, 100.0]), np.array([0.0, 0.09, 0.0]))
    assert not flags.any()


# --- Intermittent sales (HU0023) ----------------------------------------------------------------
def test_more_than_half_of_the_periods_at_zero_is_intermittent() -> None:
    values = [0.0 if i % 2 == 0 else 6.0 for i in range(21)]  # 11 of 21 at zero: 52 %
    fc = _forecast(values)
    assert fc.model == "CrostonSBA"
    assert fc.diagnostics is not None and fc.diagnostics.intermittent


def test_a_quarter_of_the_periods_at_zero_is_not_intermittent() -> None:
    values = [0.0 if i % 4 == 0 else 12.0 + (i % 3) for i in range(24)]
    fc = _forecast(values)
    assert fc.model != "CrostonSBA"
    assert fc.diagnostics is not None and not fc.diagnostics.intermittent


# --- The tournament (HU0023) --------------------------------------------------------------------
def _retail() -> list[float]:
    rng = np.random.default_rng(11)
    curve = np.array([0.8, 0.8, 0.9, 0.95, 1.25, 1.3, 1.15, 0.95, 0.95, 1.0, 1.25, 1.7])
    t = np.arange(30)
    return rng.poisson(25 * curve[t % 12] * (0.8 + 0.4 * t / 30)).astype(float).tolist()


def test_every_candidate_reports_its_error() -> None:
    fc = _forecast(_retail())
    assert fc.diagnostics is not None and len(fc.diagnostics.candidates) >= 2
    for candidate in fc.diagnostics.candidates:
        assert candidate.mae is not None
        assert candidate.wape is not None


def test_the_winner_is_the_one_that_erred_least() -> None:
    fc = _forecast(_retail())
    assert fc.diagnostics is not None
    candidates = fc.diagnostics.candidates
    chosen = [c for c in candidates if c.chosen]
    assert len(chosen) == 1
    # The FTGM family may win within 3 % of the best: never by more.
    assert chosen[0].mae <= min(c.mae for c in candidates) * 1.03 + 1e-3


# --- Fallback and accuracy (HU0023, HU0025) -----------------------------------------------------
def test_a_fallback_says_why() -> None:
    points, _as_of = _months([10.0, 12.0, 11.0], date(2025, 1, 1))
    request = ForecastRequest(period=12, horizon=3, series=[ProductSeries(product_id="p", points=points)])
    fc = ForecastService().forecast(request).forecasts[0]
    assert fc.status == "fallback"
    assert fc.fallback_reason


def _score(units: float) -> HoldoutScore:
    return HoldoutScore(origins=1, horizon=1, mae=1.0, rmse=1.0, mape=None, mase=None, total_wape=8.0, holdout_units=units)


def test_no_accuracy_under_ten_units_in_the_test() -> None:
    assert _score(9.9).accuracy_pct is None


def test_accuracy_from_ten_units_on() -> None:
    assert _score(10.0).accuracy_pct == 92.0
