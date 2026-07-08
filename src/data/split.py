"""Split interactions into train / valid / test by recent-day windows."""

from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
DATASET_DIR = PROCESSED_DIR / "hm"
INTER_FILE = DATASET_DIR / "hm.inter"
TRAIN_INTER_FILE = DATASET_DIR / "hm.train.inter"
VALID_INTER_FILE = DATASET_DIR / "hm.valid.inter"
TEST_INTER_FILE = DATASET_DIR / "hm.test.inter"


def split_by_time(
    inter_path: Path | None = None,
    valid_days: int = 7,
    test_days: int = 7,
    train_inter_path: Path | None = None,
    valid_inter_path: Path | None = None,
    test_inter_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    inter_path = inter_path or INTER_FILE
    train_inter_path = train_inter_path or TRAIN_INTER_FILE
    valid_inter_path = valid_inter_path or VALID_INTER_FILE
    test_inter_path = test_inter_path or TEST_INTER_FILE
    train_inter_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inter_path, sep="\t")
    df["datetime"] = pd.to_datetime(df["timestamp:float"], unit="s")
    df["date"] = df["datetime"].dt.floor("D")

    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    valid_start = test_start - pd.Timedelta(days=valid_days)

    train_df = df[df["date"] < valid_start]
    valid_df = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
    test_df = df[df["date"] >= test_start]

    for split_df, output_path in (
        (train_df, train_inter_path),
        (valid_df, valid_inter_path),
        (test_df, test_inter_path),
    ):
        split_df = split_df.sort_values(["user_id:token", "timestamp:float"])
        split_df[["user_id:token", "item_id:token", "timestamp:float"]].to_csv(
            output_path, sep="\t", index=False
        )

    print(
        "Date windows: "
        f"train < {valid_start.date()}, "
        f"valid [{valid_start.date()}, {(test_start - pd.Timedelta(days=1)).date()}], "
        f"test [{test_start.date()}, {max_date.date()}]"
    )
    print(
        f"Rows - train: {len(train_df):,}, valid: {len(valid_df):,}, test: {len(test_df):,}"
    )
    print(
        f"Users - train: {train_df['user_id:token'].nunique():,}, "
        f"valid: {valid_df['user_id:token'].nunique():,}, "
        f"test: {test_df['user_id:token'].nunique():,}"
    )
    return train_inter_path, valid_inter_path, test_inter_path


if __name__ == "__main__":
    split_by_time()
