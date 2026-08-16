"""Weighted reciprocal-rank-fusion implementation of the ranking boundary."""  # 现有基线排序器

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from fashionrec.domain.ids import canonical_item_id
from fashionrec.ranking.base import RankedItem


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
        _ = user_id  # RRF 当前不使用用户 ID，但保留统一 Ranker 接口
        history = {canonical_item_id(item) for item in user_history} if self.exclude_seen else set()
        merged_scores: dict[str, float] = defaultdict(float)
        for channel, candidates in channel_candidates.items():
            weight = self.channel_weights.get(channel, 0.0)
            if weight <= 0:
                continue
            for channel_rank, (item_id, _score) in enumerate(candidates, start=1):
                canonical = canonical_item_id(item_id)
                if canonical in history:
                    continue
                merged_scores[canonical] += weight / channel_rank
        fused = sorted(merged_scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
        return [RankedItem(item_id, score, rank) for rank, (item_id, score) in enumerate(fused, start=1)]
