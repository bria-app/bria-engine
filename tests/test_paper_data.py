"""Validation against the paper's real M5 data (docs/ftgmmodel/sales_data1.csv).

Two guarantees on genuine retail data (self-skips if the dataset is absent):

* **Robustness** — the service returns finite, non-negative forecasts for every series,
  even the ones where the grey ODE would diverge (those degrade to the baseline).
* **Capability** — on most series the FTGM fits the real seasonal structure better
  in-sample than the seasonal-naive benchmark.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.application.forecast_service import ForecastService
from app.ftgm import FTGM, FTGMConfig, FTGMError, select_order
from app.ftgm import metrics as M
from app.ftgm import preprocessing as prep
from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries

_CSV = Path(__file__).resolve().parents[1] / "docs" / "ftgmmodel" / "sales_data1.csv"
_DAYS_PER_MONTH = 30
_N_ROWS = 15
_SEASON = 12


def _monthly_rows(limit: int) -> list[np.ndarray]:
    """Aggregate the first ``limit`` daily series into monthly totals."""
    rows: list[np.ndarray] = []
    with _CSV.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        first_day = header.index("d_1")
        for i, row in enumerate(reader):
            if i >= limit:
                break
            daily = np.array([float(v) for v in row[first_day:]], dtype=np.float64)
            months = daily.size // _DAYS_PER_MONTH
            rows.append(daily[: months * _DAYS_PER_MONTH].reshape(months, _DAYS_PER_MONTH).sum(axis=1))
    return rows


@pytest.mark.skipif(not _CSV.exists(), reason="paper dataset not available")
def test_service_returns_finite_forecasts_for_every_real_series() -> None:
    service = ForecastService()
    start = date(2020, 1, 1)
    for idx, monthly in enumerate(_monthly_rows(_N_ROWS)):
        month_dates = [start] + prep.future_period_dates(start, monthly.size - 1, 12)
        points = [
            ObservationPoint(date=d, demand=float(v)) for d, v in zip(month_dates, monthly)
        ]
        request = ForecastRequest(
            period=12, horizon=6, series=[ProductSeries(product_id=f"row-{idx}", points=points)]
        )
        forecast = service.forecast(request).forecasts[0]

        assert len(forecast.points) == 6
        for p in forecast.points:
            assert np.isfinite(p.predicted_demand) and p.predicted_demand >= 0.0


@pytest.mark.skipif(not _CSV.exists(), reason="paper dataset not available")
def test_ftgm_beats_seasonal_naive_on_most_real_series() -> None:
    config = FTGMConfig(period=12)
    wins = 0
    fitted_count = 0
    for monthly in _monthly_rows(_N_ROWS):
        if monthly.size < 2 * _SEASON:
            continue
        try:
            order = select_order(monthly, config).best_order
            model = FTGM(order=order, config=config).fit(monthly)
        except FTGMError:
            continue  # diverged -> handled by the baseline in production
        assert model.fitted_ is not None
        fitted_count += 1

        naive_fit = np.concatenate([monthly[:_SEASON], monthly[:-_SEASON]])
        ftgm_rmse = M.root_mean_squared_error(monthly[_SEASON:], model.fitted_[_SEASON:])
        naive_rmse = M.root_mean_squared_error(monthly[_SEASON:], naive_fit[_SEASON:])
        if ftgm_rmse < naive_rmse:
            wins += 1

    assert fitted_count >= 12  # the divergence guard should keep almost all of them
    assert wins >= 8  # FTGM learns the real seasonal structure on the clear majority
