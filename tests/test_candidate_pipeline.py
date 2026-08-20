"""Integration tests for recall registry boundary and candidate union."""

from __future__ import annotations

from dataclasses import dataclass

from fashionrec.candidates.union import (
    UNION_SCHEMA_VERSION,
    build_union_evidence,
    select_union_items,
    union_candidates,
)
from fashionrec.domain.candidates import Candidate
from fashionrec.recall.generator import generate_candidates


@dataclass
class FakeChannel:
    name: str
    values: list[tuple[str, float]]

    def recall(self, user_id: str, history: list[str], top_k: int) -> list[tuple[str, float]]:
        return self.values[:top_k]


def _candidate(user: str, item: str, channel: str, score: float, rank: int) -> Candidate:
    return Candidate(user, item, channel, score, rank, "valid")


def test_generator_and_union_use_one_schema_and_preserve_channel_evidence() -> None:
    candidates = generate_candidates(
        eval_users=["u1"],
        user_history={"u1": ["9"]},
        channels={
            "popular": FakeChannel("popular", [("1", 10.0), ("2", 8.0)]),
            "sequence": FakeChannel("sequence", [("0000000001", 0.9), ("3", 0.8)]),
        },
        split="valid",
        top_k_by_channel={"popular": 2, "sequence": 2},
    )
    assert {candidate.item_id for candidate in candidates} == {"0000000001", "0000000002", "0000000003"}

    union = union_candidates(candidates, top_k_items_per_user=2)
    assert {candidate.item_id for candidate in union} == {"0000000001", "0000000002"}
    assert [(row.item_id, row.channel) for row in union].count(("0000000001", "popular")) == 1
    assert [(row.item_id, row.channel) for row in union].count(("0000000001", "sequence")) == 1


def test_union_prefers_multi_channel_coverage_over_single_channel_rank() -> None:
    rows = [
        _candidate("u1", "0000000001", "popular", 1.0, 1),
        _candidate("u1", "0000000002", "popular", 1.0, 2),
        _candidate("u1", "0000000003", "popular", 1.0, 3),
        _candidate("u1", "0000000009", "sequence", 1.0, 50),
        _candidate("u1", "0000000009", "item2item", 1.0, 50),
    ]
    selected = select_union_items(rows, top_k_items_per_user=2)
    assert selected == {"0000000001", "0000000009"}  # 双路 0000000009 胜过单路 rank2/3


def test_union_evidence_includes_channel_present_rank_score_and_metadata() -> None:
    rows = [
        _candidate("u1", "0000000001", "popular", 10.0, 1),
        _candidate("u1", "0000000001", "sequence", 0.9, 2),
    ]
    evidence = build_union_evidence(
        rows,
        top_k_items_per_user=1,
        channels=["popular", "sequence"],
        source_timestamp="2020-09-01T00:00:00Z",
        feature_version=UNION_SCHEMA_VERSION,
    )
    assert len(evidence) == 1
    row = evidence[0]
    assert row["user_id"] == "u1"
    assert row["item_id"] == "0000000001"
    assert row["channel_count"] == 2
    assert row["best_channel_rank"] == 1
    assert row["max_channel_score"] == 10.0
    assert row["source_timestamp"] == "2020-09-01T00:00:00Z"
    assert row["feature_version"] == UNION_SCHEMA_VERSION
    assert row["popular_present"] == 1
    assert row["popular_rank"] == 1
    assert row["sequence_present"] == 1
    assert row["sequence_rank"] == 2


def test_union_top_k_caps_unique_items_but_keeps_all_channel_rows() -> None:
    rows = [
        _candidate("u1", "0000000001", "popular", 1.0, 1),
        _candidate("u1", "0000000001", "sequence", 1.0, 1),
        _candidate("u1", "0000000002", "popular", 1.0, 2),
        _candidate("u1", "0000000003", "popular", 1.0, 3),
    ]
    union = union_candidates(rows, top_k_items_per_user=2)
    assert {row.item_id for row in union} == {"0000000001", "0000000002"}
    assert sum(1 for row in union if row.item_id == "0000000001") == 2  # 双通道证据都保留
