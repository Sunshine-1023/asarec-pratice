"""Shared experiment-window time boundaries."""

from __future__ import annotations

import pandas as pd


def as_naive_utc_day(value: pd.Timestamp) -> pd.Timestamp:
    """Normalize a timestamp to a timezone-naive UTC calendar day."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def week_window_start(max_date: pd.Timestamp, weeks: int) -> pd.Timestamp:
    """Return the first normalized day in an inclusive N-week window."""
    if weeks < 1:
        raise ValueError("weeks must be >= 1")
    max_day = as_naive_utc_day(max_date)
    return max_day - pd.Timedelta(days=weeks * 7 - 1)
