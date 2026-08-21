"""Tests for ranking dataset assembly (Task 4.1)."""  # 候选 + 特征 + 标签

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 落盘

import pandas as pd  # 表

from fashionrec.shared.domain.candidates import Candidate
from fashionrec.industrial.ranking.dataset import (
    RANKING_DATASET_SCHEMA_VERSION,
    build_ranking_dataset,
    write_ranking_dataset,
)
from fashionrec.industrial.ranking.features import lambda_rank_group_sizes, ranking_group_id


def _candidate(user: str, item: str, channel: str, score: float, rank: int, split: str = "train") -> Candidate:
    return Candidate(user, item, channel, score, rank, split)


def test_ranking_dataset_one_row_per_candidate_user_item() -> None:
    candidates = [
        _candidate("u1", "0000000001", "popular", 10.0, 1),
        _candidate("u1", "0000000001", "sequence", 0.9, 2),
        _candidate("u1", "0000000002", "popular", 8.0, 2),
    ]
    dataset = build_ranking_dataset(
        candidates,
        channels=["popular", "sequence"],
        snapshot_dates="2020-09-08",
        history_lengths={"u1": 4},
    )
    assert dataset.schema_version == RANKING_DATASET_SCHEMA_VERSION
    assert dataset.n_rows == 2
    row = dataset.frame[dataset.frame["item_id"] == "0000000001"].iloc[0]
    assert row["snapshot_date"] == pd.Timestamp("2020-09-08")
    assert row["group_id"] == ranking_group_id("u1", "2020-09-08")
    assert row["channel_count"] == 2
    assert row["popular_present"] == 1
    assert row["sequence_present"] == 1
    assert dataset.group_sizes == [2]


def test_ranking_dataset_labels_only_inside_candidate_set() -> None:
    candidates = [
        _candidate("u1", "0000000001", "popular", 1.0, 1),
        _candidate("u1", "0000000002", "popular", 1.0, 2),
    ]
    labels = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["0000000001", "0000000009"],
            "as_of_date": ["2020-09-08", "2020-09-08"],
            "label_purchase": [1, 1],
        }
    )
    dataset = build_ranking_dataset(
        candidates,
        channels=["popular"],
        snapshot_dates="2020-09-08",
        labels=labels,
    )
    assert set(dataset.frame["item_id"]) == {"0000000001", "0000000002"}
    assert "0000000009" not in set(dataset.frame["item_id"])  # 候选外购买不得进训练表
    positives = dataset.frame[dataset.frame["label"] == 1]
    assert list(positives["item_id"]) == ["0000000001"]
    assert int(positives.iloc[0]["relevance"]) == 1
    assert dataset.n_positives == 1
    assert dataset.n_uncovered_labels == 1
    negatives = dataset.frame[dataset.frame["item_id"] == "0000000002"].iloc[0]
    assert int(negatives["label"]) == 0
    assert int(negatives["relevance"]) == 0


def test_ranking_dataset_groups_by_user_snapshot() -> None:
    candidates = [
        _candidate("u1", "0000000001", "popular", 1.0, 1, "train"),
        _candidate("u1", "0000000002", "popular", 1.0, 1, "valid"),
        _candidate("u1", "0000000003", "popular", 1.0, 2, "valid"),
    ]
    dataset = build_ranking_dataset(
        candidates,
        channels=["popular"],
        snapshot_dates={
            ("u1", "train"): "2020-09-08",
            ("u1", "valid"): "2020-09-15",
        },
        history_lengths={"u1": 3},
    )
    assert dataset.group_sizes == [1, 2]
    assert lambda_rank_group_sizes(dataset.frame) == [1, 2]
    assert dataset.frame["group_id"].nunique() == 2


def test_ranking_dataset_missing_features_get_defaults_and_rates() -> None:
    candidates = [
        _candidate("u1", "0000000001", "popular", 1.0, 1),
        _candidate("u2", "0000000002", "popular", 1.0, 1),
    ]
    user_features = pd.DataFrame(
        {
            "user_id": ["u1"],
            "as_of_date": ["2020-09-08"],
            "purchase_count_7d": [4.0],
            "feature_version": ["hm.user_features.v1"],
        }
    )
    item_features = pd.DataFrame(
        {
            "item_id": ["0000000001"],
            "product_code:float": [101.0],
            "product_type_name:token": ["T-shirt"],
        }
    )
    cross_features = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["0000000001"],
            "as_of_date": ["2020-09-08"],
            "user_item_purchase_count": [2.0],
        }
    )
    dataset = build_ranking_dataset(
        candidates,
        channels=["popular"],
        snapshot_dates="2020-09-08",
        user_features=user_features,
        item_features=item_features,
        cross_features=cross_features,
        customer_features=None,
    )
    u1 = dataset.frame[dataset.frame["user_id"] == "u1"].iloc[0]
    u2 = dataset.frame[dataset.frame["user_id"] == "u2"].iloc[0]
    assert float(u1["user__purchase_count_7d"]) == 4.0
    assert float(u1["user_features_missing:float"]) == 0.0
    assert float(u2["user_features_missing:float"]) == 1.0
    assert float(u2["user__purchase_count_7d"]) == 0.0
    assert float(u1["item__product_code:float"]) == 101.0
    assert u2["item__product_type_name:token"] == "unknown"
    assert float(u1["cross__user_item_purchase_count"]) == 2.0
    assert float(u2["cross_features_missing:float"]) == 1.0
    assert dataset.missing_rates["customer"] == 1.0
    assert dataset.missing_rates["user"] == 0.5
    assert dataset.missing_rates["item"] == 0.5
    assert dataset.missing_rates["cross"] == 0.5


def test_ranking_dataset_writes_parquet(tmp_path: Path) -> None:
    candidates = [_candidate("u1", "0000000001", "popular", 1.0, 1)]
    dataset = build_ranking_dataset(
        candidates,
        channels=["popular"],
        snapshot_dates="2020-09-08",
        labels=pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["0000000001"],
                "as_of_date": ["2020-09-08"],
                "label_purchase": [1],
            }
        ),
    )
    path = write_ranking_dataset(dataset, tmp_path / "rank.parquet")
    loaded = pd.read_parquet(path)
    assert list(loaded.columns[:5]) == ["user_id", "item_id", "snapshot_date", "group_id", "split"]
    assert int(loaded.iloc[0]["label"]) == 1
