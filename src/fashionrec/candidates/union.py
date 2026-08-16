"""Deterministic per-user candidate union with channel evidence preserved."""  # 候选并集

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from fashionrec.domain.candidates import Candidate


def union_candidates(candidates: Iterable[Candidate], top_k_items_per_user: int) -> list[Candidate]:
    """De-duplicate channel rows and cap unique items per user while retaining all selected channel rows."""
    if top_k_items_per_user < 1:
        raise ValueError("top_k_items_per_user must be >= 1")

    by_user: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_user[candidate.user_id].append(candidate)

    result: list[Candidate] = []
    for user_id in sorted(by_user):
        unique_channel_rows: dict[tuple[str, str], Candidate] = {}
        for candidate in by_user[user_id]:
            key = (candidate.item_id, candidate.channel)
            previous = unique_channel_rows.get(key)
            if previous is None or (candidate.rank, -candidate.score) < (previous.rank, -previous.score):
                unique_channel_rows[key] = candidate

        rows = list(unique_channel_rows.values())
        item_priority: dict[str, tuple[int, float, str]] = {}
        for row in rows:
            priority = (row.rank, -row.score, row.item_id)
            item_priority[row.item_id] = min(priority, item_priority.get(row.item_id, priority))
        selected_items = {
            item_id
            for item_id, _ in sorted(item_priority.items(), key=lambda pair: pair[1])[:top_k_items_per_user]
        }
        result.extend(
            sorted(
                (row for row in rows if row.item_id in selected_items),
                key=lambda row: (row.user_id, row.item_id, row.channel, row.rank),
            )
        )
    return result

