"""HTTP hardening: request bounds, shared-secret auth, honest errors and config (audit)."""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.application import forecast_service as fs
from app.application.forecast_service import ForecastService
from app.config import settings
from app.ftgm import preprocessing as prep
from app.main import app
from app.presentation.schemas import (
    MAX_POINTS,
    MAX_SERIES,
    ForecastRequest,
    ObservationPoint,
    ProductSeries,
)

client = TestClient(app)
URL = "/api/v1/forecast"


def _monthly_points(n: int = 30) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        year, month = divmod(i, 12)
        out.append({"date": date(2022 + year, month + 1, 1).isoformat(), "demand": 10.0 + i % 12})
    return out


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "period": 12,
        "horizon": 3,
        "series": [{"product_id": "p1", "points": _monthly_points()}],
    }
    body.update(overrides)
    return body


@pytest.fixture
def engine_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(settings, "engine_api_key", "s3cr3t")
    return "s3cr3t"


# ----------------------------------------------------------------------- auth
def test_no_key_configured_keeps_forecast_open() -> None:
    assert not settings.engine_api_key
    assert client.post(URL, json=_body()).status_code == 200


def test_missing_key_is_401(engine_key: str) -> None:
    assert client.post(URL, json=_body()).status_code == 401


def test_wrong_key_is_401(engine_key: str) -> None:
    res = client.post(URL, json=_body(), headers={"X-Engine-Key": "nope"})
    assert res.status_code == 401


def test_auth_is_checked_before_body_validation(engine_key: str) -> None:
    res = client.post(URL, json={"series": "garbage"})
    assert res.status_code == 401


def test_right_key_is_accepted(engine_key: str) -> None:
    res = client.post(URL, json=_body(), headers={"X-Engine-Key": engine_key})
    assert res.status_code == 200
    assert res.json()["forecasts"][0]["status"] in ("ok", "fallback")


def test_health_and_config_stay_public(engine_key: str) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/config").status_code == 200


# ---------------------------------------------------------------------- bounds
@pytest.mark.parametrize("demand", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_demand_is_422(demand: str) -> None:
    body = _body()
    raw = json.dumps(body).replace('"demand": 10.0', f'"demand": {demand}', 1)
    assert demand in raw
    res = client.post(URL, content=raw, headers={"Content-Type": "application/json"})
    assert res.status_code == 422
    assert all("input" not in err for err in res.json()["detail"])  # nothing echoed back


@pytest.mark.parametrize("demand", ["nan", "inf", -1.0, "-0.5"])
def test_negative_or_non_finite_string_demand_is_422(demand: Any) -> None:
    body = _body()
    body["series"][0]["points"][0]["demand"] = demand
    assert client.post(URL, json=body).status_code == 422


def test_decimal_string_demand_still_accepted() -> None:
    # The API serialises Decimal demand as a string.
    body = _body()
    for p in body["series"][0]["points"]:
        p["demand"] = str(p["demand"])
    assert client.post(URL, json=body).status_code == 200


@pytest.mark.parametrize(
    "overrides",
    [
        {"period": 367},
        {"period": 0},
        {"max_order": 27},
        {"validation_size": 105},
        {"horizon": 105},
        {"horizon": None, "horizon_days": 731, "frequency": "auto"},
        {"model": "x" * 65},
    ],
)
def test_out_of_range_parameters_are_422(overrides: dict[str, Any]) -> None:
    assert client.post(URL, json=_body(**overrides)).status_code == 422


def test_too_many_points_is_422() -> None:
    start = date(2000, 1, 1)
    pts = [{"date": (start + timedelta(days=i)).isoformat(), "demand": 1.0} for i in range(MAX_POINTS + 1)]
    body = _body(series=[{"product_id": "p1", "points": pts}])
    assert client.post(URL, json=body).status_code == 422


def test_too_many_series_is_422() -> None:
    one = {"product_id": "p", "points": [{"date": "2024-01-01", "demand": 1.0}]}
    assert client.post(URL, json=_body(series=[one] * (MAX_SERIES + 1))).status_code == 422


def test_bounds_admit_the_limits() -> None:
    req = ForecastRequest.model_validate(
        _body(period=366, max_order=26, validation_size=104, horizon=104)
    )
    assert req.period == 366 and req.max_order == 26


# ------------------------------------------------------------- honest outputs
def test_response_model_is_the_engine_label_not_the_client_text() -> None:
    res = client.post(URL, json=_body(model="<script>alert(1)</script>"))
    assert res.status_code == 200
    model = res.json()["forecasts"][0]["model"]
    assert model in {"FTGM", "FTGMCombo", "DampedTrend", "SeasonalDamped", "SeasonalNaive"}


def test_unexpected_error_message_is_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("secret internal detail /srv/app/x.py")

    monkeypatch.setattr(fs.cleaning, "hampel_clean", boom)
    res = client.post(URL, json=_body())
    assert res.status_code == 200
    fc = res.json()["forecasts"][0]
    assert fc["status"] == "skipped"
    assert fc["fallback_reason"] == "Error interno al pronosticar este producto"
    assert "secret" not in json.dumps(fc)


def test_always_out_of_stock_product_is_skipped_not_forecast_zero() -> None:
    pts = [
        # Every day up to the cut-off flagged as a stock-out (how the API reports it).
        ObservationPoint(date=date(2024, 1, 1) + timedelta(days=i), demand=0.0, stockout_flag=True)
        for i in range(152)
    ]
    req = ForecastRequest(
        frequency="auto",
        horizon_days=30,
        as_of=date(2024, 6, 1),
        series=[ProductSeries(product_id="always-out", points=pts)],
    )
    fc = ForecastService().forecast(req).forecasts[0]
    assert fc.status == "skipped"
    assert fc.points == []
    assert fc.fallback_reason is not None and fc.fallback_reason.startswith("Siempre agotado")


def test_mostly_in_stock_product_is_still_repaired() -> None:
    rng = np.random.default_rng(0)
    pts = []
    for i in range(365):
        d = date(2023, 1, 1) + timedelta(days=i)
        out = d.month == 5  # one month without stock
        pts.append(
            ObservationPoint(date=d, demand=0.0 if out else float(rng.poisson(5)), stockout_flag=out)
        )
    req = ForecastRequest(
        frequency="monthly",
        horizon_days=60,
        as_of=date(2024, 1, 1),
        series=[ProductSeries(product_id="p", points=pts)],
    )
    fc = ForecastService().forecast(req).forecasts[0]
    assert fc.status != "skipped"
    may = next(h for h in fc.history if h.date == date(2023, 5, 1))
    assert may.is_stockout and may.cleaned > 0


# ---------------------------------------------------------------------- config
def test_config_exposes_the_thresholds_the_pipeline_uses() -> None:
    res = client.get("/api/v1/config")
    assert res.status_code == 200
    cfg = res.json()
    assert cfg["min_months_for_monthly"] == fs._MIN_MONTHS_FOR_MONTHLY == 24
    assert cfg["min_weeks_for_weekly"] == fs._MIN_WEEKS_FOR_WEEKLY == 26
    assert cfg["intermittent_zero_share"] == fs._INTERMITTENT_ZERO_SHARE == 0.5
    assert cfg["stockout_flag_share"] == prep.STOCKOUT_FLAG_SHARE == 0.10
    assert cfg["full_censor_share"] == prep.FULL_CENSOR_SHARE
    assert cfg["ftgm_tie_margin"] == fs._FTGM_TIE == 1.03
    assert cfg["max_series"] == MAX_SERIES and cfg["max_points_per_series"] == MAX_POINTS
