"""Canonical Industrial shopping-basket history adapters."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import pandas as pd

from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id


def _day(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _flatten_baskets(baskets: list[list[str]], max_items: int) -> list[str]:
    if max_items < 1:
        raise ValueError("max_items must be >= 1")
    return [item for basket in baskets for item in basket][-max_items:]


def history_from_events(
    events: pd.DataFrame,
    *,
    max_items: int,
    as_of: pd.Timestamp | str | None = None,
    max_baskets: int | None = None,
) -> dict[str, list[str]]:
    """Build ordered histories from user-day-item events, never row order."""
    required = {"user_id", "item_id", "date"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    frame = events.loc[:, ["user_id", "item_id", "date"]].copy()
    frame["user_id"] = frame["user_id"].map(canonical_user_id)
    frame["item_id"] = frame["item_id"].map(canonical_item_id)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if as_of is not None:
        frame = frame[frame["date"] <= _day(as_of)]
    frame = frame.drop_duplicates(["user_id", "date", "item_id"], keep="first")
    histories: dict[str, list[str]] = {}
    ordered = frame.sort_values(["user_id", "date", "item_id"], kind="mergesort")
    for user_id, user_frame in ordered.groupby("user_id", sort=True):
        baskets = [list(group["item_id"]) for _date, group in user_frame.groupby("date", sort=True)]
        if max_baskets is not None:
            if max_baskets < 1:
                raise ValueError("max_baskets must be >= 1")
            baskets = baskets[-max_baskets:]
        histories[str(user_id)] = _flatten_baskets(baskets, max_items)
    return histories


def history_from_interactions(
    paths: Iterable[str | Path],
    *,
    max_items: int,
    as_of: pd.Timestamp | str | None = None,
    max_baskets: int | None = None,
) -> dict[str, list[str]]:
    """Read .inter files and apply the same daily-basket semantics."""
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        frame = pd.read_csv(
            Path(raw_path),
            sep="\t",
            usecols=["user_id:token", "item_id:token", "timestamp:float"],
            dtype={"user_id:token": "string", "item_id:token": "string"},
        )
        frames.append(
            pd.DataFrame(
                {
                    "user_id": frame["user_id:token"].map(canonical_user_id),
                    "item_id": frame["item_id:token"].map(canonical_item_id),
                    "date": pd.to_datetime(frame["timestamp:float"], unit="s").dt.normalize(),
                }
            )
        )
    if not frames:
        return {}
    return history_from_events(
        pd.concat(frames, ignore_index=True),
        max_items=max_items,
        as_of=as_of,
        max_baskets=max_baskets,
    )
