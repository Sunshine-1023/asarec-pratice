"""Application-neutral ranking boundary."""

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
    """Minimum contract implemented by heuristic and learned rankers."""

    name: str

    def rank(
        self,
        *,
        user_id: str,
        user_history: set[str],
        channel_candidates: dict[str, list[tuple[str, float]]],
        top_k: int,
    ) -> list[RankedItem]:
        """Rank one user's already-materialized candidate set."""
