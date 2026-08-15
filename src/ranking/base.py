"""Ranking boundary shared by heuristic and learned rankers."""  # 排序层接口

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RankedItem:
    item_id: str
    score: float
    rank: int


@runtime_checkable
class Ranker(Protocol):
    name: str

    def rank(
        self,
        *,
        user_id: str,
        user_history: set[str],
        channel_candidates: dict[str, list[tuple[str, float]]],
        top_k: int,
    ) -> list[RankedItem]:
        """Rank a materialized candidate set for one user."""

