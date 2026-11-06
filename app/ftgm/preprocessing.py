"""Series preparation before the model sees the data.

The FTGM expects an **equally-spaced, aggregated** series (monthly by default). Raw
retail data arrives daily and noisy, so this module:

* aggregates daily observations into seasonal buckets (month / quarter / week) and
  fills calendar gaps with zeros, producing a contiguous, equally-spaced series; and
* repairs stock-out periods, where an observed zero/low value is *censored demand*
  (we could not sell what we did not have), by scaling up partially censored periods or
  interpolating from neighbouring periods (:func:`repair_censored`).

Both steps matter for forecast quality: gaps break the equal-spacing assumption, and
treating stock-out zeros as genuine demand biases the seasonal shape downwards.

Only the seasonal periods 12 (monthly), 4 (quarterly) and 52 (weekly) trigger calendar
aggregation; any other period assumes the caller already supplies one value per period.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Seasonal period -> "is this a calendar frequency we know how to bucket?"
_CALENDAR_PERIODS = {12, 4, 52}


def _bucket_start(d: date, period: int) -> date:
    """First day of the seasonal bucket that ``d`` falls into."""
    if period == 12:
        return date(d.year, d.month, 1)
    if period == 4:
        quarter_first_month = ((d.month - 1) // 3) * 3 + 1
        return date(d.year, quarter_first_month, 1)
    if period == 52:
        return d - timedelta(days=d.weekday())  # Monday of that week
    return d


def _next_bucket(d: date, period: int) -> date:
    """Start date of the bucket immediately after the one starting at ``d``."""
    if period == 12:
        return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)
    if period == 4:
        month = d.month + 3
        year, month = (d.year + (month - 1) // 12, (month - 1) % 12 + 1)
        return date(year, month, 1)
    if period == 52:
        return d + timedelta(days=7)
    return _add_months(d, 1)  # safe fallback for non-calendar periods


def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year, month = d.year + month // 12, month % 12 + 1
    return date(year, month, 1)


def aggregate(
    dates: list[date], demand: FloatArray, stockout: BoolArray, period: int
) -> tuple[list[date], FloatArray, BoolArray]:
    """Aggregate observations into contiguous seasonal buckets.

    Returns ``(bucket_starts, summed_demand, stockout_flags)`` sorted in time, with
    calendar gaps filled by zero-demand periods. A bucket is flagged as a stock-out if
    *any* observation inside it was flagged.
    """
    demand = np.asarray(demand, dtype=np.float64).ravel()
    stockout = np.asarray(stockout, dtype=bool).ravel()
    if not (len(dates) == demand.size == stockout.size):
        raise ValueError("dates, demand and stockout must have the same length")
    if demand.size == 0:
        return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)

    # Sum demand (and OR the stock-out flag) within each bucket.
    sums: dict[date, float] = {}
    flags: dict[date, bool] = {}
    for d, value, flag in zip(dates, demand, stockout):
        key = _bucket_start(d, period)
        sums[key] = sums.get(key, 0.0) + float(value)
        flags[key] = flags.get(key, False) or bool(flag)

    ordered = sorted(sums)
    if period not in _CALENDAR_PERIODS:
        # Already one value per period: no calendar gap filling possible.
        return ordered, np.array([sums[k] for k in ordered]), np.array([flags[k] for k in ordered])

    # Walk the calendar from first to last bucket, inserting zeros for missing periods.
    out_dates: list[date] = []
    out_vals: list[float] = []
    out_flags: list[bool] = []
    cursor, last = ordered[0], ordered[-1]
    while cursor <= last:
        out_dates.append(cursor)
        out_vals.append(sums.get(cursor, 0.0))
        out_flags.append(flags.get(cursor, False))
        cursor = _next_bucket(cursor, period)
    return out_dates, np.array(out_vals, dtype=np.float64), np.array(out_flags, dtype=bool)


def future_period_dates(last_start: date, horizon: int, period: int) -> list[date]:
    """Bucket start dates for the ``horizon`` periods following ``last_start``."""
    out: list[date] = []
    cursor = last_start
    for _ in range(horizon):
        cursor = _next_bucket(cursor, period)
        out.append(cursor)
    return out


# --------------------------------------------------------------------------------------
# Frequency-aware bucketing with a cut-off and *partial* stock-out censoring.
# --------------------------------------------------------------------------------------

#: Seasonal period for each supported calendar frequency.
FREQUENCY_PERIOD = {"monthly": 12, "weekly": 52}
PERIOD_FREQUENCY = {12: "monthly", 52: "weekly", 4: "quarterly"}
#: Average calendar days per bucket (horizon conversion).
AVG_DAYS_PER_PERIOD = {12: 30.4375, 52: 7.0, 4: 91.3125}

#: A bucket counts as a stock-out period when at least this share of it was censored.
STOCKOUT_FLAG_SHARE = 0.10
#: Above this share (or when nothing was sold) the recorded value is unusable and the
#: demand is interpolated from clean neighbours instead of scaled up.
FULL_CENSOR_SHARE = 0.60


def bucket_start(d: date, period: int) -> date:
    """Public alias of the bucket start (first day of month / Monday of the week)."""
    return _bucket_start(d, period)


def next_bucket(d: date, period: int) -> date:
    return _next_bucket(d, period)


class BucketedSeries:
    """Result of :func:`bucketize`: contiguous buckets up to the last complete one."""

    def __init__(
        self,
        dates: list[date],
        demand: FloatArray,
        stockout_share: FloatArray,
        dropped_incomplete: bool,
    ) -> None:
        self.dates = dates
        self.demand = demand
        self.stockout_share = stockout_share
        self.dropped_incomplete = dropped_incomplete

    @property
    def size(self) -> int:
        return int(self.demand.size)


def bucketize(
    dates: list[date],
    demand: FloatArray,
    stockout: BoolArray,
    period: int,
    as_of: date | None = None,
) -> BucketedSeries:
    """Aggregate observations into buckets, honouring a cut-off date.

    * With ``as_of`` the bucket containing ``as_of`` (the period in progress) and anything
      after it are excluded, and zero-demand buckets are filled **up to the last complete
      bucket** — so a product that stopped selling still shows its trailing zeros.
    * ``stockout_share`` is the censored fraction of each bucket: distinct flagged days
      over the bucket's calendar days for daily data, or flagged/observed when the input
      is already one value per bucket.
    """
    demand = np.asarray(demand, dtype=np.float64).ravel()
    stockout = np.asarray(stockout, dtype=bool).ravel()
    if not (len(dates) == demand.size == stockout.size):
        raise ValueError("dates, demand and stockout must have the same length")

    calendar = period in _CALENDAR_PERIODS
    cutoff_bucket = _bucket_start(as_of, period) if (as_of is not None and calendar) else None
    dropped = bool(as_of is not None and calendar and as_of > cutoff_bucket)  # type: ignore[operator]

    sums: dict[date, float] = {}
    obs: dict[date, int] = {}
    flagged_obs: dict[date, int] = {}
    flagged_days: dict[date, set[date]] = {}
    for d, value, flag in zip(dates, demand, stockout):
        key = _bucket_start(d, period)
        if cutoff_bucket is not None and key >= cutoff_bucket:
            continue
        sums[key] = sums.get(key, 0.0) + float(value)
        obs[key] = obs.get(key, 0) + 1
        if flag:
            flagged_obs[key] = flagged_obs.get(key, 0) + 1
            flagged_days.setdefault(key, set()).add(d)

    if not sums:
        return BucketedSeries([], np.empty(0), np.empty(0), dropped)

    ordered = sorted(sums)
    if not calendar:
        shares = [flagged_obs.get(k, 0) / max(1, obs[k]) for k in ordered]
        return BucketedSeries(
            ordered, np.array([sums[k] for k in ordered]), np.array(shares, dtype=np.float64), dropped
        )

    last = ordered[-1]
    if cutoff_bucket is not None:
        # Last complete bucket = the one right before the bucket in progress.
        cursor = ordered[0]
        while _next_bucket(cursor, period) < cutoff_bucket:
            cursor = _next_bucket(cursor, period)
        last = max(last, cursor)

    # Granularity is decided for the whole series: it is "pre-aggregated" only when every
    # observation sits on a bucket start and no bucket has more than one observation.
    # Sparse daily input (only days with sales or stock-outs) is still daily.
    pre_aggregated = all(obs[k] == 1 for k in obs) and all(
        _bucket_start(d, period) == d for d in dates if cutoff_bucket is None or _bucket_start(d, period) < cutoff_bucket
    )

    out_dates: list[date] = []
    out_vals: list[float] = []
    out_share: list[float] = []
    cursor = ordered[0]
    while cursor <= last:
        out_dates.append(cursor)
        out_vals.append(sums.get(cursor, 0.0))
        n_obs = obs.get(cursor, 0)
        if n_obs == 0:
            share = 0.0
        elif pre_aggregated:
            share = float(flagged_obs.get(cursor, 0))  # already one value per bucket
        else:
            days = max(1, (_next_bucket(cursor, period) - cursor).days)
            share = len(flagged_days.get(cursor, set())) / days
        out_share.append(min(1.0, share))
        cursor = _next_bucket(cursor, period)
    return BucketedSeries(
        out_dates,
        np.array(out_vals, dtype=np.float64),
        np.array(out_share, dtype=np.float64),
        dropped,
    )


def censored_without_evidence(demand: FloatArray, stockout_share: FloatArray) -> bool:
    """True when censored demand cannot be rebuilt (audit M-15).

    :func:`repair_censored` interpolates stock-out buckets from at least two clean
    neighbours. With fewer than two clean buckets a stock-out bucket that sold nothing
    would stay at zero — the product was always out of stock, not undemanded — so the
    caller must not forecast it from the recorded zeros.
    """
    x = np.asarray(demand, dtype=np.float64).ravel()
    share = np.clip(np.asarray(stockout_share, dtype=np.float64).ravel(), 0.0, 1.0)
    flags = share >= STOCKOUT_FLAG_SHARE
    if not flags.any() or int((~flags).sum()) >= 2:
        return False
    return bool(np.any(x[flags] <= 0.0))


def repair_censored(
    demand: FloatArray, stockout_share: FloatArray
) -> tuple[FloatArray, BoolArray, BoolArray]:
    """Repair censored demand in stock-out buckets.

    Returns ``(repaired, stockout_flags, imputed_mask)``:

    * lightly censored buckets (share < 60% and something was sold) are scaled up by the
      uncensored fraction: ``demand / (1 - share)`` — sales happened only on the days the
      product was available — capped at twice the interpolated neighbour level;
    * heavily censored buckets (or zero sales) are interpolated from clean neighbours.
    """
    x = np.asarray(demand, dtype=np.float64).ravel()
    share = np.clip(np.asarray(stockout_share, dtype=np.float64).ravel(), 0.0, 1.0)
    flags = share >= STOCKOUT_FLAG_SHARE
    repaired = x.copy()
    if not flags.any():
        return repaired, flags, np.zeros_like(flags)

    clean = ~flags
    idx = np.arange(x.size, dtype=np.float64)
    interp = np.interp(idx, idx[clean], x[clean]) if clean.sum() >= 2 else None

    for i in np.flatnonzero(flags):
        s = min(float(share[i]), 0.95)
        neighbour = float(interp[i]) if interp is not None else None
        if s >= FULL_CENSOR_SHARE or x[i] <= 0.0:
            if neighbour is not None:
                repaired[i] = max(x[i], neighbour)
            elif x[i] > 0:
                repaired[i] = x[i] / (1.0 - s)
        else:
            scaled = x[i] / (1.0 - s)
            if neighbour is not None and neighbour > 0:
                scaled = min(scaled, max(x[i], 2.0 * neighbour))
            repaired[i] = max(x[i], scaled)
    imputed = flags & ~np.isclose(repaired, x)
    return repaired, flags, imputed
