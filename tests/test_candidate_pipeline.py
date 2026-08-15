"""Integration tests for recall registry boundary and candidate union."""

from __future__ import annotations

from dataclasses import dataclass

from src.candidates.union import union_candidates
from src.recall.generator import generate_candidates


@dataclass
class FakeChannel:
    name: str
    values: list[tuple[str, float]]

    def recall(self, user_id: str, history: list[str], top_k: int) -> list[tuple[str, float]]:
        return self.values[:top_k]


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

