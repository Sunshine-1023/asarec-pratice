"""Pre-split preprocessing must preserve the full experiment cohort and history."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import fashionrec.data.preprocess as preprocess
from fashionrec.data.command import select_transactions_input
from fashionrec.data.preprocess import load_transactions


def _write_transactions(path: Path, rows: list[tuple[str, str, str]]) -> None:
    pd.DataFrame(rows, columns=["customer_id", "article_id", "t_dat"]).to_csv(path, index=False)


def test_preprocess_does_not_use_future_activity_to_filter_users(tmp_path: Path) -> None:
    source = tmp_path / "transactions.csv"
    _write_transactions(
        source,
        [
            ("future_threshold", "1", "2020-08-20"),
            ("future_threshold", "2", "2020-09-10"),
            ("future_threshold", "3", "2020-09-11"),
            ("future_threshold", "4", "2020-09-17"),
            ("future_threshold", "5", "2020-09-18"),
            ("anchor", "9", "2020-09-22"),
        ],
    )

    interactions = load_transactions(source, weeks=6, min_user_purchases=5)

    kept = interactions[interactions["user_id:token"] == "future_threshold"]
    assert len(kept) == 5


def test_preprocess_does_not_truncate_history_before_split(tmp_path: Path) -> None:
    source = tmp_path / "transactions.csv"
    rows = [("long_history", str(index), "2020-09-01") for index in range(105)]
    rows.append(("anchor", "999", "2020-09-22"))
    _write_transactions(source, rows)

    interactions = load_transactions(source, weeks=6, max_user_history=100)

    assert (interactions["user_id:token"] == "long_history").sum() == 105


def test_preprocess_does_not_order_same_day_items_by_article_id(tmp_path: Path) -> None:
    source = tmp_path / "transactions.csv"
    _write_transactions(
        source,
        [
            ("u", "3", "2020-09-22"),
            ("u", "1", "2020-09-22"),
            ("u", "2", "2020-09-22"),
        ],
    )
    interactions = load_transactions(source, weeks=1)
    assert interactions["item_id:token"].tolist() == ["3", "1", "2"]


def test_default_preprocess_input_never_reuses_existing_filtered_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = tmp_path / "raw.csv"
    filtered = tmp_path / "filtered.csv"
    _write_transactions(raw, [("raw_user", "1", "2020-09-22")])
    _write_transactions(filtered, [("stale_filtered_user", "2", "2020-09-22")])
    monkeypatch.setattr(preprocess, "RAW_PATH", raw)
    monkeypatch.setattr(preprocess, "FILTERED_RAW_PATH", filtered)

    interactions = preprocess.load_transactions(weeks=1)

    assert interactions["user_id:token"].tolist() == ["raw_user"]


def test_data_prep_selects_filtered_input_only_when_requested(tmp_path: Path) -> None:
    assert select_transactions_input(with_filter=False) == preprocess.RAW_PATH
    with pytest.raises(ValueError, match="refusing to reuse data/raw/filtered"):
        select_transactions_input(with_filter=True)
    fresh = tmp_path / "run" / "filtered" / "transactions_train.csv"
    fresh.parent.mkdir(parents=True)
    _write_transactions(fresh, [("fresh_user", "3", "2020-09-22")])
    assert select_transactions_input(with_filter=True, filtered_path=fresh) == fresh
    stale = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="filtered_path"):
        select_transactions_input(with_filter=True, filtered_path=stale)
