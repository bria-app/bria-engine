"""Regression tests from the 20-month MYPE simulation (Nov 2026): what the pipeline reports
must belong to the model it actually used, and its sentences never contradict its numbers."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from app.application.forecast_service import ForecastService
from app.ftgm import FTGMError
from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries


def _request(points: list[ObservationPoint], **kw: object) -> ForecastRequest:
    return ForecastRequest(series=[ProductSeries(product_id="p", points=points)], **kw)  # type: ignore[arg-type]


def _weekly_seasonal(weeks: int, seed: int = 3) -> list[ObservationPoint]:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)  # a Monday
    pts = []
    for w in range(weeks):
        level = 40 + 15 * np.sin(2 * np.pi * w / 52) + 0.2 * w
        pts.append(ObservationPoint(date=start + timedelta(weeks=w), demand=max(0.0, round(level + rng.normal(0, 3)))))
    return pts


def test_failed_final_fit_falls_back_to_damped_smoother_and_reports_its_accuracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ForecastService._fit_final

    def flaky(winner: str, *args: object) -> object:
        if winner in ("FTGM", "FTGMCombo"):
            raise FTGMError("integration diverged")
        return original(winner, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(ForecastService, "_fit_final", staticmethod(flaky))
    pts = _weekly_seasonal(120)
    fc = ForecastService().forecast(
        _request(pts, frequency="weekly", horizon_days=56, as_of=date(2026, 4, 20))
    ).forecasts[0]
    diag = fc.diagnostics
    assert diag is not None and fc.model == "DampedTrend"
    chosen = [c for c in diag.candidates if c.chosen]
    assert len(chosen) == 1 and chosen[0].model == fc.model
    # The reported accuracy is the one of the model that produced the forecast.
    assert diag.accuracy_pct == chosen[0].accuracy_pct


def test_fallback_sentence_never_contradicts_its_numbers() -> None:
    pts = _weekly_seasonal(110, seed=9)
    fc = ForecastService().forecast(
        _request(pts, frequency="weekly", horizon_days=56, as_of=date(2026, 3, 9))
    ).forecasts[0]
    if fc.fallback_reason and "se equivocó menos" in fc.fallback_reason:
        diag = fc.diagnostics
        assert diag is not None
        scores = {c.model: c.mae for c in diag.candidates}
        assert scores[fc.model] is not None and scores["FTGM"] is not None
        assert scores[fc.model] < scores["FTGM"]
