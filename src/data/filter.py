"""Filter H&M raw data: last 3 months, top items, active users."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
FILTERED_DIR = Path("data/raw/filtered")

TOP_ITEMS = 30_000
MIN_USER_PURCHASES = 5
MAX_USER_BEHAVIORS = 50
MONTHS = 3
CHUNK_SIZE = 500_000


def _normalize_article_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(10)


def _last_n_months_cutoff(dates: pd.Series, months: int) -> pd.Timestamp:
    max_date = pd.to_datetime(dates).max()
    return max_date - pd.DateOffset(months=months)


def filter_transactions(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    top_items: int = TOP_ITEMS,
    min_user_purchases: int = MIN_USER_PURCHASES,
    max_user_behaviors: int = MAX_USER_BEHAVIORS,
    months: int = MONTHS,
) -> Path:
    input_path = input_path or RAW_DIR / "transactions_train.csv"
    output_dir = output_dir or FILTERED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transactions_train.csv"

    print(f"Reading {input_path} ...")

    # Pass 1: find date cutoff from last N months
    max_date = None
    for chunk in pd.read_csv(input_path, usecols=["t_dat"], chunksize=CHUNK_SIZE):
        chunk_max = pd.to_datetime(chunk["t_dat"]).max()
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)

    cutoff = max_date - pd.DateOffset(months=months)
    print(f"Date range: {cutoff.date()} ~ {max_date.date()} (last {months} months)")

    # Pass 2: count item purchases in the time window
    item_counts: dict[str, int] = {}
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
        chunk = chunk[chunk["t_dat"] >= cutoff]
        chunk["article_id"] = _normalize_article_id(chunk["article_id"])
        counts = chunk["article_id"].value_counts()
        for item_id, count in counts.items():
            item_counts[item_id] = item_counts.get(item_id, 0) + int(count)

    top_item_ids = {
        str(item_id)
        for item_id in pd.Series(item_counts)
        .sort_values(ascending=False)
        .head(top_items)
        .index
    }
    print(f"Top {top_items} items selected (unique items in window: {len(item_counts):,})")

    # Pass 3: count user purchases after item filter
    user_counts: dict[str, int] = {}
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
        chunk = chunk[chunk["t_dat"] >= cutoff]
        chunk["article_id"] = _normalize_article_id(chunk["article_id"])
        chunk = chunk[chunk["article_id"].isin(top_item_ids)]
        counts = chunk["customer_id"].value_counts()
        for user_id, count in counts.items():
            user_counts[user_id] = user_counts.get(user_id, 0) + int(count)

    active_user_ids = {
        user_id for user_id, count in user_counts.items() if count >= min_user_purchases
    }
    print(
        f"Active users (>={min_user_purchases} purchases): "
        f"{len(active_user_ids):,} / {len(user_counts):,}"
    )

    # Pass 4: collect filtered transactions
    filtered_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
        chunk = chunk[chunk["t_dat"] >= cutoff]
        chunk["article_id"] = _normalize_article_id(chunk["article_id"])
        chunk = chunk[chunk["article_id"].isin(top_item_ids)]
        chunk = chunk[chunk["customer_id"].isin(active_user_ids)]
        if not chunk.empty:
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        pd.DataFrame(
            columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
        ).to_csv(output_path, index=False)
        print(f"Saved 0 transactions to {output_path}")
        return output_path

    df = pd.concat(filtered_chunks, ignore_index=True)
    before_truncate = len(df)

    # Pass 5: keep only the most recent N behaviors per user
    df = df.sort_values(["customer_id", "t_dat"])
    df = df.groupby("customer_id", sort=False).tail(max_user_behaviors)
    df["t_dat"] = df["t_dat"].dt.strftime("%Y-%m-%d")
    df.to_csv(output_path, index=False)

    truncated = before_truncate - len(df)
    print(
        f"Truncated to last {max_user_behaviors} behaviors per user "
        f"({truncated:,} rows removed)"
    )
    print(f"Saved {len(df):,} transactions to {output_path}")
    return output_path


def filter_articles(
    transactions_path: Path,
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    input_path = input_path or RAW_DIR / "articles.csv"
    output_dir = output_dir or FILTERED_DIR
    output_path = output_dir / "articles.csv"

    item_ids = _normalize_article_id(
        pd.read_csv(transactions_path, usecols=["article_id"], dtype={"article_id": str})[
            "article_id"
        ]
    ).unique()

    articles = pd.read_csv(input_path, dtype={"article_id": str})
    filtered = articles[articles["article_id"].isin(item_ids)]
    filtered.to_csv(output_path, index=False)

    print(f"Saved {len(filtered):,} articles to {output_path}")
    return output_path


def filter_customers(
    transactions_path: Path,
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    input_path = input_path or RAW_DIR / "customers.csv"
    output_dir = output_dir or FILTERED_DIR
    output_path = output_dir / "customers.csv"

    user_ids = pd.read_csv(transactions_path, usecols=["customer_id"])["customer_id"].unique()

    customers = pd.read_csv(input_path)
    filtered = customers[customers["customer_id"].isin(user_ids)]
    filtered.to_csv(output_path, index=False)

    print(f"Saved {len(filtered):,} customers to {output_path}")
    return output_path


def run_filter(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    top_items: int = TOP_ITEMS,
    min_user_purchases: int = MIN_USER_PURCHASES,
    max_user_behaviors: int = MAX_USER_BEHAVIORS,
    months: int = MONTHS,
) -> Path:
    input_dir = input_dir or RAW_DIR
    output_dir = output_dir or FILTERED_DIR

    tx_path = filter_transactions(
        input_path=input_dir / "transactions_train.csv",
        output_dir=output_dir,
        top_items=top_items,
        min_user_purchases=min_user_purchases,
        max_user_behaviors=max_user_behaviors,
        months=months,
    )
    filter_articles(tx_path, input_path=input_dir / "articles.csv", output_dir=output_dir)
    filter_customers(tx_path, input_path=input_dir / "customers.csv", output_dir=output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter H&M dataset")
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=FILTERED_DIR)
    parser.add_argument("--top-items", type=int, default=TOP_ITEMS)
    parser.add_argument("--min-user-purchases", type=int, default=MIN_USER_PURCHASES)
    parser.add_argument("--max-user-behaviors", type=int, default=MAX_USER_BEHAVIORS)
    parser.add_argument("--months", type=int, default=MONTHS)
    args = parser.parse_args()

    run_filter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        top_items=args.top_items,
        min_user_purchases=args.min_user_purchases,
        max_user_behaviors=args.max_user_behaviors,
        months=args.months,
    )


if __name__ == "__main__":
    main()
