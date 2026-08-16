"""Shared experiment-window time boundaries."""

from __future__ import annotations

import pandas as pd


def week_window_start(max_date: pd.Timestamp, weeks: int) -> pd.Timestamp:
    """Return the first normalized day in an inclusive N-week window."""
    if weeks < 1:
        raise ValueError("weeks must be >= 1")
    max_day = pd.Timestamp(max_date).normalize()
    return max_day - pd.Timedelta(days=weeks * 7 - 1)
