"""Long horizons and lasting level changes (30-month MYPE simulation, Nov 2026).

* Beyond the stretch the tournament can check (6 months / 8 weeks) no model keeps
  extrapolating: the level is held and only the season moves.
* A lasting jump of the level is told apart from steady growth and from the season and
  reported to the user (rescaling the history before a drop is optional: it did not improve
  accuracy in exp. 6).
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from app.application.forecast_service import ForecastService
from app.ftgm import cleaning
from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries

AS_OF = date(2026, 11, 1)


def _daily(rate, months: int = 30, seed: int = 1) -> list[ObservationPoint]:
    """Poisson daily sales for ``months`` complete months ending the day before AS_OF."""
    rng = np.random.default_rng(seed)
    start = date(AS_OF.year - months // 12, AS_OF.month, 1)
    if months % 12:
        m = AS_OF.month - months % 12
        start = date(start.year - (1 if m < 1 else 0), (m - 1) % 12 + 1, 1)
    pts, d, k = [], start, 0
    while d < AS_OF:
        v = int(rng.poisson(max(0.0, rate(d, k))))
        if v:
            pts.append(ObservationPoint(date=d, demand=float(v)))
        d += timedelta(days=1)
        k += 1
    return pts


def _forecast(points: list[ObservationPoint], days: int):
    req = ForecastRequest(
        frequency="auto", horizon_days=days, as_of=AS_OF, series=[ProductSeries(product_id="p", points=points)]
    )
    return ForecastService().forecast(req).forecasts[0]


def test_growth_is_not_extrapolated_beyond_the_checked_stretch() -> None:
    # Steady 3 %/month growth: within 6 months the trend may continue, after that it is held.
    fc = _forecast(_daily(lambda d, k: 4.0 * 1.03 ** (k / 30.4)), 730)
    pts = np.array([p.predicted_demand for p in fc.points])
    assert fc.frequency == "monthly" and pts.size == 24
    # Flat level: per day, only noise and the 28-31 days of each month remain.
    days = np.array([(date(p.date.year + p.date.month // 12, p.date.month % 12 + 1, 1) - p.date).days for p in fc.points])
    tail = (pts / days)[6:]
    assert np.ptp(tail) / tail.mean() < 0.1
    assert tail.mean() < 1.1 * (pts / days)[5]  # no growth after the checked stretch
    assert any("no sigue extrapolando" in e for e in fc.diagnostics.explanations)


def test_held_level_keeps_the_season() -> None:
    season = lambda d: 1.0 + 0.6 * math.cos(2 * math.pi * (d.timetuple().tm_yday - 35) / 365.25)  # noqa: E731
    fc = _forecast(_daily(lambda d, k: 3.0 * season(d), seed=4), 730)
    pts = np.array([p.predicted_demand for p in fc.points])
    expected = np.array([season(p.date + timedelta(days=14)) for p in fc.points])
    # Beyond the checked stretch the shape follows the season (summer high, winter low).
    assert np.corrcoef(pts[6:], expected[6:])[0, 1] > 0.9


def test_short_horizons_are_untouched() -> None:
    pts = _daily(lambda d, k: 4.0 * 1.03 ** (k / 30.4))
    fc = _forecast(pts, 90)
    assert not any("no sigue extrapolando" in e for e in fc.diagnostics.explanations)


def test_lasting_drop_is_reported_to_the_user() -> None:
    drop = date(2026, 4, 1)
    fc = _forecast(_daily(lambda d, k: 2.6 * (0.6 if d >= drop else 1.0), seed=7), 365)
    notes = [e for e in fc.diagnostics.explanations if e.startswith("Cambio de nivel desde 2026-04-01")]
    assert len(notes) == 1 and "revisa el pronóstico" in notes[0]


def test_optional_rescale_starts_the_forecast_from_the_new_level(monkeypatch) -> None:
    from app.application import forecast_service as fs

    monkeypatch.setattr(fs, "_LEVEL_SHIFT_RESCALE_DROPS", True)
    drop = date(2026, 4, 1)
    fc = _forecast(_daily(lambda d, k: 2.6 * (0.6 if d >= drop else 1.0), seed=7), 365)
    assert any("se ajustó al nivel nuevo" in e for e in fc.diagnostics.explanations)
    total = sum(p.predicted_demand for p in fc.points)
    assert total < 0.75 * 12 * 2.6 * 30.4  # far from the old level


def test_lasting_rise_is_reported_but_not_used() -> None:
    rise = date(2026, 4, 1)
    fc = _forecast(_daily(lambda d, k: 2.6 * (1.7 if d >= rise else 1.0), seed=7), 90)
    assert any(e.startswith("Cambio de nivel") and "+" in e for e in fc.diagnostics.explanations)
    assert not any("se ajustó al nivel nuevo" in e for e in fc.diagnostics.explanations)


def test_steady_growth_is_not_a_level_shift() -> None:
    x = np.array([100 * 1.016**i for i in range(30)]) + np.random.default_rng(2).normal(0, 3, 30)
    assert cleaning.detect_level_shift(x, 12) is None


def test_recurring_season_is_not_a_level_shift() -> None:
    # Winter clothing in a weekly series too short (78 weeks) to estimate the season: the
    # winter rise happened the year before too, so it is not a new level.
    w = np.arange(78)
    x = 2 + 14 * np.exp(-(((w % 52) - 50) / 6.0) ** 2) + np.random.default_rng(3).normal(0, 0.5, 78)
    assert cleaning.detect_level_shift(np.clip(x, 0, None), 52) is None


def test_no_shift_without_a_year_before_to_compare() -> None:
    x = np.concatenate([np.full(20, 50.0), np.full(10, 80.0)]) + np.random.default_rng(5).normal(0, 2, 30)
    assert cleaning.detect_level_shift(x, 52) is None
