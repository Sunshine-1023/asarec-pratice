"""Convert H&M transactions to RecBole ``hm.inter`` format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
FILTERED_RAW_PATH = RAW_DIR / "filtered/transactions_train.csv"
RAW_PATH = RAW_DIR / "transactions_train.csv"
PROCESSED_DIR = Path("data/processed")
DATASET_DIR = PROCESSED_DIR / "hm"
INTER_FILE = DATASET_DIR / "hm.inter"

MONTHS = 3
MIN_USER_PURCHASES = 5


def _default_input_path() -> Path:
    # Prefer sampled data for quick end-to-end runs.
    if FILTERED_RAW_PATH.exists():
        return FILTERED_RAW_PATH
    return RAW_PATH


def load_transactions(
    path: Path | None = None,
    months: int = MONTHS,
    min_user_purchases: int = MIN_USER_PURCHASES,
) -> pd.DataFrame:
    path = path or _default_input_path()
    df = pd.read_csv(
        path,
        dtype={"customer_id": "string", "article_id": "string"},
        parse_dates=["t_dat"],
    )
    df = df[["customer_id", "article_id", "t_dat"]]

    max_date = df["t_dat"].max()
    min_date = max_date - pd.DateOffset(months=months)
    df = df[df["t_dat"] >= min_date]

    user_cnt = df["customer_id"].value_counts()
    valid_users = user_cnt[user_cnt >= min_user_purchases].index
    df = df[df["customer_id"].isin(valid_users)]

    df = df.sort_values(["customer_id", "t_dat"])
    df["timestamp"] = (df["t_dat"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

    out = pd.DataFrame(
        {
            "user_id:token": df["customer_id"],
            "item_id:token": df["article_id"],
            "timestamp:float": df["timestamp"],
        }
    )
    return out


def build_inter_file(
    transactions_path: Path | None = None,
    output_path: Path | None = None,
    months: int = MONTHS,
    min_user_purchases: int = MIN_USER_PURCHASES,
) -> Path:
    output_path = output_path or INTER_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = load_transactions(
        path=transactions_path,
        months=months,
        min_user_purchases=min_user_purchases,
    )
    out.to_csv(output_path, sep="\t", index=False)
    print(f"saved: {output_path}")
    print(f"rows: {len(out):,}")
    print(out.head())
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RecBole hm.inter from H&M transactions")
    parser.add_argument("--transactions-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=INTER_FILE)
    parser.add_argument("--months", type=int, default=MONTHS)
    parser.add_argument("--min-user-purchases", type=int, default=MIN_USER_PURCHASES)
    args = parser.parse_args()

    build_inter_file(
        transactions_path=args.transactions_path,
        output_path=args.output_path,
        months=args.months,
        min_user_purchases=args.min_user_purchases,
    )


if __name__ == "__main__":
    main()
