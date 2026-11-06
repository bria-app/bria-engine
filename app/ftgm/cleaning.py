"""Robust series diagnostics: outlier treatment, seasonality strength and trend.

These helpers run *after* aggregation and censored-demand repair, before the FTGM sees
the data:

* :func:`hampel_clean` — a Hampel filter on the **deseasonalised** series, so a genuine
  seasonal peak (e.g. December) is not mistaken for an outlier, while one-off spikes
  (a bulk order, a data-entry error) are replaced by the local robust level.
* :func:`seasonality_strength` — Hyndman's ``F_s = max(0, 1 - Var(R) / Var(S + R))``
  from a classical additive decomposition (needs two full seasons).
* :func:`trend_per_period` — robust linear slope of the deseasonalised series, as a
  share of the mean level per period.
* :func:`seasonal_index` — multiplicative seasonal factors (all ones when unreliable).
* :func:`detect_level_shift` — a lasting jump of the level near the end of the history
  (e.g. a competitor opens next door), told apart from steady growth and from the season.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_MAD_SCALE = 1.4826  # MAD -> sigma under normality
_MIN_POINTS = 8
#: A point must also deviate by this share of the local level to count as an outlier
#: (avoids flagging tiny wiggles when the local MAD is almost zero).
_MIN_REL_DEVIATION = 0.35


def _centered_moving_average(x: FloatArray, period: int) -> FloatArray:
    """Centered moving average of length ``period`` (2xMA for even periods); NaN at edges."""
    n = x.size
    out = np.full(n, np.nan)
    if period < 2 or n < period + 1:
        return out
    if period % 2 == 1:
        kernel = np.ones(period) / period
        half = period // 2
    else:
        kernel = np.concatenate([[0.5], np.ones(period - 1), [0.5]]) / period
        half = period // 2
    conv = np.convolve(x, kernel, mode="valid")
    out[half : half + conv.size] = conv
    return out


def seasonal_component(x: FloatArray, period: int) -> FloatArray | None:
    """Additive seasonal profile (sums to ~0 over a season) or None if < 2 seasons."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if period < 2 or x.size < 2 * period:
        return None
    trend = _centered_moving_average(x, period)
    detrended = x - trend
    profile = np.zeros(period)
    for pos in range(period):
        vals = detrended[pos::period]
        vals = vals[np.isfinite(vals)]
        profile[pos] = float(np.median(vals)) if vals.size else 0.0
    profile -= profile.mean()
    reps = int(np.ceil(x.size / period))
    return np.tile(profile, reps)[: x.size]


def seasonality_strength(x: FloatArray, period: int) -> float | None:
    """Hyndman's seasonal strength in [0, 1]; None when the history is too short."""
    x = np.asarray(x, dtype=np.float64).ravel()
    season = seasonal_component(x, period)
    if season is None:
        return None
    trend = _centered_moving_average(x, period)
    valid = np.isfinite(trend)
    if valid.sum() < period:
        return None
    detrended = (x - trend)[valid]
    remainder = detrended - season[valid]
    var_sr = float(np.var(detrended))
    if var_sr <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - float(np.var(remainder)) / var_sr)))


def hampel_clean(
    x: FloatArray,
    period: int,
    protected: BoolArray | None = None,
    half_window: int | None = None,
    n_sigmas: float = 3.0,
) -> tuple[FloatArray, BoolArray]:
    """Replace outliers of the deseasonalised series by the local rolling median.

    ``protected`` buckets (e.g. repaired stock-outs) are never modified. Returns the
    cleaned series (non-negative) and the outlier mask.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    mask = np.zeros(n, dtype=bool)
    if n < _MIN_POINTS:
        return x.copy(), mask
    k = half_window if half_window is not None else (4 if period >= 52 else 3)
    season = seasonal_component(x, period)
    base = x - season if season is not None else x.copy()

    cleaned = x.copy()
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        window = np.delete(base[lo:hi], i - lo)  # exclude the point itself
        if window.size < 3:
            continue
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med))) * _MAD_SCALE
        deviation = abs(base[i] - med)
        scale = max(mad, 1e-9)
        level = max(abs(med), 1e-9)
        if deviation > n_sigmas * scale and deviation > _MIN_REL_DEVIATION * level:
            if protected is not None and protected[i]:
                continue
            mask[i] = True
            cleaned[i] = med + (season[i] if season is not None else 0.0)
    return np.clip(cleaned, 0.0, None), mask


def trend_per_period(x: FloatArray, period: int) -> float | None:
    """Least-squares slope of the (deseasonalised) recent history, as % of the mean."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 4:
        return None
    season = seasonal_component(x, period)
    base = x - season if season is not None else x
    window = base[-min(base.size, 2 * period if period > 1 else base.size) :]
    mean = float(np.mean(window))
    if mean <= 1e-9:
        return None
    t = np.arange(window.size, dtype=np.float64)
    slope = float(np.polyfit(t, window, 1)[0])
    return 100.0 * slope / mean


def seasonal_index(x: FloatArray, period: int, min_strength: float = 0.3) -> FloatArray:
    """Multiplicative seasonal profile (length ``period``, mean 1); ones when not reliable.

    Position ``i % period`` of the history (and ``(n + k) % period`` of forecast step ``k``)
    reads its factor here. Needs two full seasons and a seasonality strength of at least
    ``min_strength``; otherwise the series is treated as non-seasonal (all ones).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    ones = np.ones(max(1, period))
    if period < 2 or x.size < 2 * period:
        return ones
    strength = seasonality_strength(x, period)
    if strength is None or strength < min_strength:
        return ones
    trend = _centered_moving_average(x, period)
    profile = np.ones(period)
    for pos in range(period):
        t, v = trend[pos::period], x[pos::period]
        ok = np.isfinite(t) & (t > 1e-9)
        if ok.any():
            profile[pos] = float(np.median(v[ok] / t[ok]))
    profile = np.clip(profile, 0.05, None)
    profile = profile / profile.mean()
    # With two or three seasons the decomposition also reads noise as season. Keep only the
    # part of the shape that repeats: correlate the within-year shape of the last full years
    # (each divided by its own straight trend) and shrink the profile by that agreement.
    agreement = _season_agreement(x, period)
    return 1.0 + agreement * (profile - 1.0)


def _season_agreement(x: FloatArray, period: int) -> float:
    """Mean correlation (clipped to [0, 1]) of the within-year shape between consecutive years."""
    cycles = x.size // period
    shapes = []
    t = np.arange(period, dtype=np.float64)
    for c in range(cycles):
        seg = x[x.size - (cycles - c) * period : x.size - (cycles - c - 1) * period]
        line = np.polyval(np.polyfit(t, seg, 1), t)
        if np.any(line <= 1e-9):
            line = np.full(period, max(float(seg.mean()), 1e-9))
        shapes.append(seg / line)
    corrs = []
    for a, b in zip(shapes, shapes[1:]):
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.clip(np.mean(corrs), 0.0, 1.0)) if corrs else 0.0


@dataclass(frozen=True)
class LevelShift:
    """A lasting jump of the level: from ``index`` on, sales run at ``ratio`` x the level before."""

    index: int
    ratio: float
    before: float
    after: float


def detect_level_shift(x: FloatArray, period: int) -> LevelShift | None:
    """Most significant *lasting* level change near the end of the series, or None.

    Works on the deseasonalised series (when the seasonality is reliable). A candidate break
    ``b`` must:

    * leave at least ``min_after`` periods after it (6 months / 13 weeks) and a full window
      before it (12 months / 26 weeks), inside the last 18 months / 52 weeks;
    * change the mean by at least 30 % with a Welch t of at least 5, and last: both halves
      of the stretch after it stay at least 60 % of the way to the new level;
    * fit clearly better as a step than as a straight trend (SSE at most 60 %, so steady
      growth is not read as a jump);
    * not have happened in the same periods the year before (otherwise it is the season,
      e.g. winter clothing in a series too short to estimate the seasonality). When there is
      not a year of history before the break this cannot be checked and no shift is declared.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    weekly = period >= 52
    min_after = 13 if weekly else 6
    w_before = 26 if weekly else 12
    search = 52 if weekly else 18
    if n < w_before + min_after:
        return None
    s = seasonal_index(x, period)
    base = x / np.array([s[i % s.size] for i in range(n)])
    best: LevelShift | None = None
    best_t = 0.0
    for b in range(max(w_before, n - search), n - min_after + 1):
        before, after = base[b - w_before : b], base[b:]
        mb, ma = float(before.mean()), float(after.mean())
        if mb <= 1e-9:
            continue
        ratio = ma / mb
        if abs(ratio - 1.0) < 0.3:
            continue
        se = math.sqrt(float(before.var(ddof=1)) / before.size + float(after.var(ddof=1)) / after.size)
        t = abs(ma - mb) / max(se, 1e-9)
        if t < 5.0:
            continue
        # It must last: both halves of the stretch after the break sit on the new side.
        half = after.size // 2
        jump = ma - mb
        if any((float(part.mean()) - mb) / jump < 0.6 for part in (after[:half], after[half:])):
            continue
        seg = base[b - w_before :]
        sse_step = float(((before - mb) ** 2).sum() + ((after - ma) ** 2).sum())
        tt = np.arange(seg.size, dtype=np.float64)
        sse_line = float(((seg - np.polyval(np.polyfit(tt, seg, 1), tt)) ** 2).sum())
        if sse_step > 0.6 * sse_line:
            continue
        # Same periods one year earlier: a recurring jump is the season, not a new level.
        lb0, lb1 = max(0, b - period - w_before), b - period
        la1 = min(n - period, lb1 + (n - b))
        if lb1 - lb0 < 4 or la1 - lb1 < min_after:
            continue
        prev_before, prev_after = float(x[lb0:lb1].mean()), float(x[lb1:la1].mean())
        if prev_before > 1e-9:
            prev_ratio = prev_after / prev_before
            if (prev_ratio - 1.0) * (ratio - 1.0) > 0 and abs(prev_ratio - 1.0) >= 0.5 * abs(ratio - 1.0):
                continue
        if t > best_t:
            best_t, best = t, LevelShift(index=b, ratio=ratio, before=mb, after=ma)
    return best
