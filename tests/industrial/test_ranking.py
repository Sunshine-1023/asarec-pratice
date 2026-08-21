"""Tests for ranking implementations and LightGBM feature boundary."""

from __future__ import annotations

from fashionrec.shared.domain.candidates import Candidate
from fashionrec.industrial.ranking.features import build_ranking_features, lambda_rank_group_sizes
from fashionrec.industrial.ranking.weighted_rrf import WeightedRRFRanker


def test_weighted_rrf_ranker_implements_stable_ranking_contract() -> None:
    ranker = WeightedRRFRanker({"popular": 0.4, "sequence": 0.6})
    ranked = ranker.rank(
        user_id="u1",
        user_history=set(),
        channel_candidates={"popular": [("1", 10)], "sequence": [("0000000001", 0.9), ("2", 0.8)]},
        top_k=2,
    )
    assert [item.item_id for item in ranked] == ["0000000001", "0000000002"]
    assert [item.rank for item in ranked] == [1, 2]


def test_lightgbm_feature_table_is_one_row_per_user_item_with_groups() -> None:
    candidates = [
        Candidate("u1", "1", "popular", 10, 1, "valid"),
        Candidate("u1", "0000000001", "sequence", 0.9, 1, "valid"),
        Candidate("u1", "2", "popular", 8, 2, "valid"),
        Candidate("u2", "3", "popular", 7, 1, "valid"),
    ]
    frame = build_ranking_features(
        candidates,
        history_lengths={"u1": 5, "u2": 0},
        channels=["popular", "sequence"],
        targets={"u1": {"1"}, "u2": set()},
    )
    assert len(frame) == 3
    first = frame[(frame["user_id"] == "u1") & (frame["item_id"] == "0000000001")].iloc[0]
    assert first["channel_count"] == 2
    assert first["popular_present"] == 1
    assert first["sequence_present"] == 1
    assert first["label"] == 1
    assert lambda_rank_group_sizes(frame) == [2, 1]


def test_ranking_features_attach_snapshot_and_group_id() -> None:
    candidates = [
        Candidate("u1", "1", "popular", 10, 1, "train"),
        Candidate("u1", "2", "popular", 8, 2, "train"),
    ]
    frame = build_ranking_features(
        candidates,
        history_lengths={"u1": 3},
        channels=["popular"],
        snapshot_dates="2020-09-08",
    )
    assert list(frame["group_id"].unique()) == ["u1@2020-09-08"]
    assert lambda_rank_group_sizes(frame) == [2]

