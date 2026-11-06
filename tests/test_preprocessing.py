"""Unit tests for aggregation and stock-out repair."""
from __future__ import annotations

from datetime import date

import numpy as np

from app.ftgm import preprocessing as prep


def test_monthly_aggregation_sums_and_fills_gaps() -> None:
    dates = [date(2025, 1, 5), date(2025, 1, 20), date(2025, 3, 2)]  # Feb missing
    demand = np.array([10.0, 5.0, 7.0])
    flags = np.array([False, False, False])

    bucket_dates, values, _ = prep.aggregate(dates, demand, flags, period=12)

    assert bucket_dates == [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)]
    np.testing.assert_allclose(values, [15.0, 0.0, 7.0])  # Jan summed, Feb gap-filled


def test_stockout_imputation_interpolates() -> None:
    demand = np.array([10.0, 0.0, 12.0])  # middle period is a censored stock-out
    share = np.array([0.0, 1.0, 0.0])

    repaired, flags, imputed = prep.repair_censored(demand, share)

    assert repaired[1] == 11.0  # interpolated between 10 and 12, not the recorded 0
    assert flags.tolist() == [False, True, False] and imputed.tolist() == [False, True, False]


def test_always_out_of_stock_has_no_evidence_to_repair() -> None:
    # Fewer than two clean buckets and a stock-out bucket that sold nothing.
    assert prep.censored_without_evidence(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert prep.censored_without_evidence(np.array([5.0, 0.0]), np.array([0.0, 0.8]))
    # Two clean neighbours (interpolation possible) or sales in every censored bucket: fine.
    assert not prep.censored_without_evidence(np.array([10.0, 0.0, 12.0]), np.array([0.0, 1.0, 0.0]))
    assert not prep.censored_without_evidence(np.array([3.0, 4.0]), np.array([0.5, 0.5]))
    assert not prep.censored_without_evidence(np.array([0.0, 0.0]), np.array([0.0, 0.0]))


def test_future_period_dates_monthly_rolls_over_year() -> None:
    out = prep.future_period_dates(date(2025, 11, 1), horizon=3, period=12)
    assert out == [date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1)]
