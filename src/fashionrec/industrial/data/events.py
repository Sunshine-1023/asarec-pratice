"""Aggregate raw transaction rows into causal user-day-item events."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id


EVENT_SCHEMA_VERSION = "hm.user_day_item.v1"
PARTITION_COL = "year_month"
EVENT_COLUMNS = (
    "user_id",
    "item_id",
    "date",
    "quantity",
    "mean_price",
    "min_price",
    "max_price",
    "sales_channel_mode",
    "channel_count",
)
REQUIRED_COLUMNS = ("t_dat", "customer_id", "article_id", "price", "sales_channel_id")
GROUP_KEYS = ("user_id", "item_id", "date")


def _missing_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.isna() | text.str.strip().eq("") | text.eq("<NA>")


def _require_present(series: pd.Series, name: str) -> None:
    missing = int(_missing_mask(series).sum())
    if missing:
        raise ValueError(f"{missing} rows have missing {name}")


def aggregate_user_day_item_events(transactions: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in transactions.columns]
    if missing_columns:
        raise ValueError(f"transactions missing columns: {missing_columns}")
    if transactions.empty:
        raise ValueError("transactions must not be empty")

    frame = transactions.loc[:, list(REQUIRED_COLUMNS)].copy()
    _require_present(frame["customer_id"], "customer_id")
    _require_present(frame["article_id"], "article_id")
    frame["date"] = pd.to_datetime(frame["t_dat"], errors="coerce").dt.normalize()
    bad_dates = int(frame["date"].isna().sum())
    if bad_dates:
        raise ValueError(f"{bad_dates} rows have missing or invalid t_dat")
    frame["user_id"] = frame["customer_id"].map(canonical_user_id)
    frame["item_id"] = frame["article_id"].map(canonical_item_id)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["sales_channel_id"] = pd.to_numeric(frame["sales_channel_id"], errors="coerce")
    bad_channels = int(frame["sales_channel_id"].isna().sum())
    if bad_channels:
        raise ValueError(f"{bad_channels} rows have missing or invalid sales_channel_id")
    frame["sales_channel_id"] = frame["sales_channel_id"].astype(int)

    keys = list(GROUP_KEYS)
    quantity = frame.groupby(keys, sort=False).size().rename("quantity")
    priced = frame.loc[frame["price"].notna(), keys + ["price"]]
    if priced.empty:
        price_stats = pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in ("mean_price", "min_price", "max_price")},
            index=quantity.index,
        )
    else:
        price_stats = priced.groupby(keys)["price"].agg(
            mean_price="mean",
            min_price="min",
            max_price="max",
        )

    channel_rows = (
        frame.groupby([*keys, "sales_channel_id"], sort=False).size().rename("n").reset_index()
    )
    channel_rows = channel_rows.sort_values(
        [*keys, "n", "sales_channel_id"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    mode = channel_rows.drop_duplicates(keys, keep="first").rename(
        columns={"sales_channel_id": "sales_channel_mode"}
    )
    channel_count = channel_rows.groupby(keys, sort=False)["sales_channel_id"].nunique().rename("channel_count")

    events = (
        quantity.to_frame()
        .join(price_stats, how="left")
        .join(channel_count, how="left")
        .reset_index()
        .merge(mode[keys + ["sales_channel_mode"]], on=keys, how="left")
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    for column in ("quantity", "channel_count", "sales_channel_mode"):
        events[column] = events[column].astype("int64")
    return events.loc[:, list(EVENT_COLUMNS)]


def write_events_parquet(events: pd.DataFrame, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if events.empty:
        raise ValueError("cannot write empty events table")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = events.copy()
    frame[PARTITION_COL] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m")
    frame.to_parquet(output_dir, partition_cols=[PARTITION_COL], index=False, engine="pyarrow")
    return output_dir


def build_events(*, transactions_path: Path, output_dir: Path) -> Path:
    transactions_path = Path(transactions_path)
    if not transactions_path.is_file():
        raise FileNotFoundError(f"transactions file not found: {transactions_path}")
    transactions = pd.read_csv(
        transactions_path,
        dtype={"customer_id": "string", "article_id": "string"},
    )
    events = aggregate_user_day_item_events(transactions)
    written = write_events_parquet(events, output_dir)
    print(f"saved events: {written} ({len(events):,} rows, schema {EVENT_SCHEMA_VERSION})")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate transactions into monthly-partitioned user-day-item events."
    )
    parser.add_argument("--transactions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_events(transactions_path=args.transactions, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
