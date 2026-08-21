"""Build the compact RecBole item table required by Baseline SASRecF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from fashionrec.shared.domain.ids import canonical_item_id


RAW_ARTICLES_PATH = Path("data/raw/articles.csv")
SEQ_DATASET_DIR = Path("data/processed/hm_seq")
OUTPUT_ITEM_FILE = SEQ_DATASET_DIR / "hm_seq.item"
SEQ_SPLIT_FILES = (
    SEQ_DATASET_DIR / "hm_seq.train.inter",
    SEQ_DATASET_DIR / "hm_seq.valid.inter",
    SEQ_DATASET_DIR / "hm_seq.test.inter",
)
UNKNOWN_TOKEN = "unknown"
RECB_CATEGORY_COLUMNS = (
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "department_name",
    "index_name",
    "index_group_name",
)
RECB_ITEM_FILE_COLUMNS = ("item_id:token", *(f"{column}:token" for column in RECB_CATEGORY_COLUMNS))
RAW_FEATURE_COLUMNS = list(RECB_CATEGORY_COLUMNS)
ITEM_FILE_COLUMNS = list(RECB_ITEM_FILE_COLUMNS)


def clean_category_token(value: object) -> str:
    if value is None or pd.isna(value):
        return UNKNOWN_TOKEN
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return UNKNOWN_TOKEN
    text = re.sub(r"[\s/\\]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or UNKNOWN_TOKEN


def _collect_inter_items(inter_paths: tuple[Path, ...]) -> set[str]:
    item_ids: set[str] = set()
    for inter_path in inter_paths:
        if not inter_path.exists():
            raise FileNotFoundError(f"Missing split file: {inter_path}")
        frame = pd.read_csv(
            inter_path,
            sep="\t",
            usecols=["item_id:token"],
            dtype={"item_id:token": "string"},
        )
        item_ids.update(frame["item_id:token"].map(canonical_item_id))
    return {item_id for item_id in item_ids if item_id}


def _load_recbole_item_frame(articles_path: Path, item_ids: set[str]) -> pd.DataFrame:
    wanted = {"article_id", *RECB_CATEGORY_COLUMNS}
    articles = pd.read_csv(
        articles_path,
        dtype={"article_id": "string"},
        usecols=lambda column: column in wanted,
    )
    if "article_id" not in articles.columns:
        raise ValueError(f"{articles_path} must contain article_id")
    frame = pd.DataFrame({"item_id:token": articles["article_id"].map(canonical_item_id)})
    for column in RECB_CATEGORY_COLUMNS:
        source = articles[column] if column in articles.columns else pd.Series(pd.NA, index=articles.index)
        frame[f"{column}:token"] = source.map(clean_category_token)
    frame = frame.drop_duplicates("item_id:token", keep="first")
    frame = frame[frame["item_id:token"].isin(item_ids)].copy()
    missing = sorted(item_ids.difference(frame["item_id:token"]))
    if missing:
        unknown_rows = pd.DataFrame(
            [
                {"item_id:token": item_id, **{f"{column}:token": UNKNOWN_TOKEN for column in RECB_CATEGORY_COLUMNS}}
                for item_id in missing
            ]
        )
        frame = pd.concat([frame, unknown_rows], ignore_index=True)
    return frame.loc[:, list(RECB_ITEM_FILE_COLUMNS)].sort_values("item_id:token").reset_index(drop=True)


def build_item_features(
    articles_path: Path = RAW_ARTICLES_PATH,
    output_path: Path = OUTPUT_ITEM_FILE,
    inter_paths: tuple[Path, ...] = SEQ_SPLIT_FILES,
) -> Path:
    item_ids = _collect_inter_items(inter_paths)
    if not item_ids:
        raise ValueError("No item ids found in hm_seq split files.")
    frame = _load_recbole_item_frame(Path(articles_path), item_ids)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep="\t", index=False)
    print(f"saved: {output_path}")
    print(f"rows: {len(frame):,}")
    print(f"covered item ids from inter: {len(item_ids):,}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build hm_seq.item for Baseline SASRecF")
    parser.add_argument("--articles-path", type=Path, default=RAW_ARTICLES_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_ITEM_FILE)
    parser.add_argument("--train-inter-path", type=Path, default=SEQ_SPLIT_FILES[0])
    parser.add_argument("--valid-inter-path", type=Path, default=SEQ_SPLIT_FILES[1])
    parser.add_argument("--test-inter-path", type=Path, default=SEQ_SPLIT_FILES[2])
    args = parser.parse_args(argv)
    build_item_features(
        articles_path=args.articles_path,
        output_path=args.output_path,
        inter_paths=(args.train_inter_path, args.valid_inter_path, args.test_inter_path),
    )


if __name__ == "__main__":
    main()
