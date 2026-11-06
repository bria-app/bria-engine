"""HTTP request/response models for the FTGM engine.

This is the **engine side of the contract** shared with ``bria-api``. The
engine accepts a *batch* of product series (so the backend can forecast a whole catalog
in one call) and returns, per product, the point forecast with a prediction interval,
the Fourier order that was selected, the aggregated in-sample history (with the model
fit), accuracy metrics, and diagnostics explaining what the pipeline did.

Two ways to call it:

* **Legacy** — ``period`` (seasonal period T) + ``horizon`` (periods). Every product is
  aggregated with that period.
* **Frequency-aware** — ``frequency`` (``auto`` | ``monthly`` | ``weekly``) +
  ``horizon_days`` + ``as_of``. The engine decides the bucket size per product (monthly
  with >= 24 months of history, weekly with >= 26 weeks, otherwise a short-history
  baseline), drops the period in progress and converts the day horizon to periods.

Backward compatibility: every field added on top of the original contract has a
default, so an older client simply ignores the extras.

Input bounds (audit A-06): the engine is reachable over the network, so the request is
bounded — at most ``MAX_SERIES`` products per batch and ``MAX_POINTS`` observations per
product, finite non-negative demand, and caps on ``period``, ``validation_size`` and
``max_order`` — so a single call cannot saturate the CPU or inject degenerate series.
The caps sit well above anything the forecasting pipeline uses, so valid requests are
processed exactly as before.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ``model`` is a meaningful field name here; opt out of pydantic's ``model_`` guard.
_ALLOW_MODEL_FIELD = ConfigDict(protected_namespaces=())

Frequency = Literal["auto", "monthly", "weekly"]

#: Request bounds (audit A-06).
MAX_SERIES = 2000
MAX_POINTS = 5000
MAX_PERIOD = 366
#: ``validation_size`` is a hold-out length in periods: two years of weeks is plenty
#: (Algorithm 1 shrinks it further when the series is short).
MAX_VALIDATION_SIZE = 104
#: ``max_order`` only *lowers* the engine's own cap (<= 6 for the supported periods,
#: Nyquist ``ceil(T/2) - 1`` = 25 for weekly data).
MAX_ORDER = 26
MAX_HORIZON = 104
MAX_HORIZON_DAYS = 730


class ObservationPoint(BaseModel):
    """A single observed demand value at a given date."""

    date: date
    demand: float = Field(ge=0, allow_inf_nan=False, description="Units demanded (finite, >= 0)")
    stockout_flag: bool = False


class ProductSeries(BaseModel):
    """The historical demand series for one product."""

    product_id: str = Field(max_length=128)
    points: list[ObservationPoint] = Field(max_length=MAX_POINTS)


class ForecastRequest(BaseModel):
    """Batch forecast request."""

    model_config = _ALLOW_MODEL_FIELD

    #: Informative only: the engine always runs its own pipeline and reports the model
    #: it actually used in ``ProductForecast.model`` (the value is never echoed back).
    model: str = Field(default="FTGM", max_length=64)
    period: int = Field(
        default=12, gt=0, le=MAX_PERIOD, description="Seasonal period T (12 = monthly)"
    )
    horizon: int | None = Field(
        default=None, gt=0, le=MAX_HORIZON, description="Number of periods to forecast (legacy)"
    )
    series: list[ProductSeries] = Field(max_length=MAX_SERIES)
    frequency: Frequency | None = Field(
        default=None, description="Bucket size; None keeps the legacy `period` behaviour"
    )
    horizon_days: int | None = Field(
        default=None,
        gt=0,
        le=MAX_HORIZON_DAYS,
        description="Horizon in days (converted per frequency)",
    )
    as_of: date | None = Field(
        default=None, description="Cut-off: the period containing this date is excluded"
    )
    # Optional overrides for Algorithm 1 (defaults follow the paper).
    validation_size: int | None = Field(
        default=None,
        gt=0,
        le=MAX_VALIDATION_SIZE,
        description="Hold-out size for order selection (default: T/2)",
    )
    max_order: int | None = Field(
        default=None, gt=0, le=MAX_ORDER, description="Cap on the candidate Fourier orders"
    )

    @model_validator(mode="after")
    def _horizon_given(self) -> "ForecastRequest":
        if self.horizon is None and self.horizon_days is None:
            raise ValueError("either `horizon` or `horizon_days` is required")
        return self


class ForecastPoint(BaseModel):
    """One forecasted period with its prediction interval."""

    date: date
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class HistoryPoint(BaseModel):
    """One in-sample period: what was observed, what the model consumed and produced.

    ``observed``   – aggregated demand for the bucket (raw sum).
    ``cleaned``    – demand after stock-out repair and outlier treatment (model input).
    ``fitted``     – the model's in-sample fit for the bucket (None for skipped series).
    ``is_stockout``– the bucket was censored by a stock-out (>= 10% of it).
    ``is_outlier`` – the bucket was treated as an outlier (Hampel filter).
    """

    date: date
    observed: float
    cleaned: float
    fitted: float | None = None
    is_stockout: bool = False
    is_outlier: bool = False
    stockout_share: float = 0.0


class ForecastMetrics(BaseModel):
    """In-sample accuracy metrics for the fitted model."""

    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    mase: float | None = None
    rmsse: float | None = None


class HoldoutMetrics(BaseModel):
    """Out-of-sample accuracy from rolling-origin evaluation."""

    origins: int
    horizon: int
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    mase: float | None = None
    wape: float | None = None
    total_wape: float | None = None


class CandidateScore(BaseModel):
    """One contender of the model tournament, scored on the same rolling origins."""

    model: str
    mae: float | None = None
    wape: float | None = None
    accuracy_pct: float | None = None
    chosen: bool = False


class ProductDiagnostics(BaseModel):
    """Evidence of what the pipeline did for one product (Algorithm 1 transparency)."""

    n_input_points: int = Field(description="Raw observations received")
    n_periods: int = Field(description="Aggregated seasonal buckets used for fitting")
    stockout_periods: int = Field(default=0, description="Buckets flagged as stock-out")
    imputed_periods: int = Field(default=0, description="Buckets repaired by interpolation")
    validation_size: int | None = Field(
        default=None, description="Hold-out length used by order selection"
    )
    order_scores: dict[int, float | None] = Field(
        default_factory=dict,
        description="Validation RMSE per candidate Fourier order (null = unstable order)",
    )
    validation_rmse: float | None = Field(
        default=None, description="Validation RMSE of the selected order"
    )
    # --- frequency-aware pipeline evidence ------------------------------------------
    frequency: str | None = None
    period: int | None = None
    frequency_reason: str | None = None
    history_start: date | None = None
    history_end: date | None = None
    dropped_incomplete_period: bool = False
    outliers_cleaned: int = 0
    zero_share: float | None = None
    intermittent: bool = False
    seasonality_strength: float | None = None
    trend_pct_per_period: float | None = None
    max_order_allowed: int | None = None
    holdout: HoldoutMetrics | None = None
    naive_holdout: HoldoutMetrics | None = None
    skill_vs_naive: float | None = Field(
        default=None, description="1 - RMSE(model)/RMSE(seasonal naive) on the same origins"
    )
    interval_level: float | None = None
    accuracy_pct: float | None = Field(
        default=None,
        description="Plain accuracy of the chosen model on past data: 100 - WAPE of the horizon total",
    )
    candidates: list[CandidateScore] = Field(default_factory=list)
    forecast_vs_recent_pct: float | None = Field(
        default=None, description="Forecast total vs the same number of recent periods (%)"
    )
    explanations: list[str] = Field(default_factory=list)


class ProductForecast(BaseModel):
    """Forecast bundle for a single product."""

    model_config = _ALLOW_MODEL_FIELD

    product_id: str
    model: str
    order_selected: int = Field(description="Fourier order chosen; 0 = baseline fallback")
    status: Literal["ok", "fallback", "skipped"] = "ok"
    fallback_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    frequency: str | None = None
    period: int | None = None
    points: list[ForecastPoint]
    history: list[HistoryPoint] = Field(default_factory=list)
    metrics: ForecastMetrics
    diagnostics: ProductDiagnostics | None = None


class ForecastResponse(BaseModel):
    """Batch forecast response."""

    period: int
    forecasts: list[ProductForecast]


class EngineConfig(BaseModel):
    """Read-only thresholds the engine applies (audit B-07).

    Exposed so the backend preview can show the same frequency / intermittency /
    stock-out decisions the engine will take, instead of duplicating the constants.
    """

    min_months_for_monthly: int = Field(description="Complete months needed for monthly buckets")
    min_weeks_for_weekly: int = Field(description="Complete weeks needed for weekly buckets")
    intermittent_zero_share: float = Field(
        description="Share of zero periods above which Croston-SBA replaces the FTGM"
    )
    stockout_flag_share: float = Field(
        description="Censored share of a bucket from which it counts as a stock-out period"
    )
    full_censor_share: float = Field(
        description="Censored share above which demand is interpolated instead of scaled up"
    )
    ftgm_tie_margin: float = Field(
        description="Tournament tie rule: the FTGM family wins within this MAE ratio of the best"
    )
    interval_level: float = Field(description="Nominal level of the prediction interval")
    max_series: int = MAX_SERIES
    max_points_per_series: int = MAX_POINTS
