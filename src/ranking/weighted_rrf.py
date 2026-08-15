"""Weighted reciprocal-rank-fusion implementation of the ranking boundary."""  # 现有基线排序器

from __future__ import annotations

from dataclasses import dataclass

from src.fusion.weighted_fusion import fuse_candidates
from src.ranking.base import RankedItem


@dataclass(slots=True)
class WeightedRRFRanker:
    channel_weights: dict[str, float]
    exclude_seen: bool = False
    name: str = "weighted_rrf"

    def rank(
        self,
        *,
        user_id: str,
        user_history: set[str],
        channel_candidates: dict[str, list[tuple[str, float]]],
        top_k: int,
    ) -> list[RankedItem]:
        fused = fuse_candidates(
            user_id=user_id,
            user_history=user_history,
            channel_candidates=channel_candidates,
            channel_weights=self.channel_weights,
            top_k=top_k,
            exclude_seen=self.exclude_seen,
        )
        return [RankedItem(item_id, score, rank) for rank, (item_id, score) in enumerate(fused, start=1)]

