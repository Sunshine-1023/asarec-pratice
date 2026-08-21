"""Build unordered daily baskets from Industrial user-day-item events."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from fashionrec.industrial.data.events import aggregate_user_day_item_events
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id


BASKET_SCHEMA_VERSION = "hm.daily_basket.v1"
PARTITION_COL = "year_month"
BASKET_COLUMNS = ("user_id", "date", "item_ids", "n_items", "quantity_sum")


def _unique_sorted_item_ids(values) -> str:
    return " ".join(sorted({canonical_item_id(value) for value in values}))


def flatten_recent_baskets(
    baskets: list[list[str]],
    *,
    max_item_list_length: int,
    max_shopping_days: int | None = None,
) -> list[str]:
    if max_item_list_length < 1:
        raise ValueError("max_item_list_length must be >= 1")
    selected = list(baskets)
    if max_shopping_days is not None:
        if max_shopping_days < 1:
            raise ValueError("max_shopping_days must be >= 1")
        selected = selected[-max_shopping_days:]
    while selected and sum(len(day) for day in selected) > max_item_list_length:
        if len(selected) == 1:
            return list(selected[0][:max_item_list_length])
        selected = selected[1:]
    return [item_id for day_items in selected for item_id in day_items]


def baskets_from_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "item_id", "date", "quantity"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("events must not be empty")

    frame = events.copy()
    frame["user_id"] = frame["user_id"].map(canonical_user_id)
    frame["item_id"] = frame["item_id"].map(canonical_item_id)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    baskets = (
        frame.groupby(["user_id", "date"], sort=True, observed=True)
        .agg(
            item_ids=("item_id", _unique_sorted_item_ids),
            n_items=("item_id", "nunique"),
            quantity_sum=("quantity", "sum"),
        )
        .reset_index()
    )
    baskets["n_items"] = baskets["n_items"].astype("int64")
    baskets["quantity_sum"] = baskets["quantity_sum"].astype("int64")
    return baskets.loc[:, list(BASKET_COLUMNS)]


def baskets_from_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id:token", "item_id:token", "timestamp:float"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")
    if interactions.empty:
        raise ValueError("interactions must not be empty")
    frame = interactions.loc[:, list(required)].copy()
    frame["user_id"] = frame["user_id:token"].map(canonical_user_id)
    frame["item_id"] = frame["item_id:token"].map(canonical_item_id)
    frame["date"] = pd.to_datetime(frame["timestamp:float"], unit="s").dt.normalize()
    frame["quantity"] = 1
    return baskets_from_events(frame.loc[:, ["user_id", "item_id", "date", "quantity"]])


def write_baskets_parquet(baskets: pd.DataFrame, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if baskets.empty:
        raise ValueError("cannot write empty baskets table")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = baskets.copy()
    frame[PARTITION_COL] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m")
    frame.to_parquet(output_dir, partition_cols=[PARTITION_COL], index=False, engine="pyarrow")
    return output_dir


def build_baskets(
    *,
    output_dir: Path,
    events_dir: Path | None = None,
    transactions_path: Path | None = None,
) -> Path:
    if events_dir is None and transactions_path is None:
        raise ValueError("build_baskets requires events_dir or transactions_path")
    if events_dir is not None:
        events_dir = Path(events_dir)
        if not events_dir.exists():
            raise FileNotFoundError(f"events dir not found: {events_dir}")
        events = pd.read_parquet(events_dir)
    else:
        transactions_path = Path(transactions_path)
        if not transactions_path.is_file():
            raise FileNotFoundError(f"transactions file not found: {transactions_path}")
        transactions = pd.read_csv(
            transactions_path,
            dtype={"customer_id": "string", "article_id": "string"},
        )
        events = aggregate_user_day_item_events(transactions)
    baskets = baskets_from_events(events)
    written = write_baskets_parquet(baskets, output_dir)
    print(f"saved baskets: {written} ({len(baskets):,} rows, schema {BASKET_SCHEMA_VERSION})")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build monthly-partitioned daily baskets.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, default=None)
    parser.add_argument("--transactions", type=Path, default=None)
    args = parser.parse_args(argv)
    build_baskets(
        output_dir=args.output_dir,
        events_dir=args.events_dir,
        transactions_path=args.transactions,
    )


if __name__ == "__main__":
    main()
