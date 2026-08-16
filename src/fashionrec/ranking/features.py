"""Build one-row-per-user-item features for LightGBM LambdaRank."""  # 学习排序特征边界

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from fashionrec.domain.candidates import Candidate
from fashionrec.domain.ids import canonical_item_id, canonical_user_id


BASE_RANKING_COLUMNS = (  # 与具体召回通道无关的稳定字段
    "user_id",
    "item_id",
    "split",
    "history_len",
    "channel_count",
    "best_channel_rank",
    "max_channel_score",
)


def build_ranking_features(
    candidates: Iterable[Candidate],
    *,
    history_lengths: Mapping[str, int],
    channels: Iterable[str],
    targets: Mapping[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Pivot channel evidence into a deterministic LambdaRank training/inference table."""
    channel_names = tuple(dict.fromkeys(str(channel).strip().lower() for channel in channels))
    if not channel_names:
        raise ValueError("channels must not be empty")

    evidence: dict[tuple[str, str, str], dict[str, Candidate]] = {}
    for candidate in candidates:
        if candidate.channel not in channel_names:
            continue
        key = (candidate.user_id, candidate.item_id, candidate.split)
        per_channel = evidence.setdefault(key, {})
        previous = per_channel.get(candidate.channel)
        if previous is None or (candidate.rank, -candidate.score) < (previous.rank, -previous.score):
            per_channel[candidate.channel] = candidate

    rows: list[dict[str, object]] = []
    for (user_id, item_id, split), per_channel in sorted(evidence.items()):
        row: dict[str, object] = {
            "user_id": user_id,
            "item_id": item_id,
            "split": split,
            "history_len": int(history_lengths.get(user_id, 0)),
            "channel_count": len(per_channel),
            "best_channel_rank": min(candidate.rank for candidate in per_channel.values()),
            "max_channel_score": max(candidate.score for candidate in per_channel.values()),
        }
        for channel in channel_names:
            candidate = per_channel.get(channel)
            row[f"{channel}_present"] = int(candidate is not None)
            row[f"{channel}_score"] = float(candidate.score) if candidate is not None else 0.0
            row[f"{channel}_rank"] = int(candidate.rank) if candidate is not None else 0
        if targets is not None:
            actual = {canonical_item_id(value) for value in targets.get(canonical_user_id(user_id), set())}
            row["label"] = int(canonical_item_id(item_id) in actual)
        rows.append(row)

    ordered_columns = list(BASE_RANKING_COLUMNS)
    for channel in channel_names:
        ordered_columns.extend([f"{channel}_present", f"{channel}_score", f"{channel}_rank"])
    if targets is not None:
        ordered_columns.append("label")
    return pd.DataFrame(rows, columns=ordered_columns)


def lambda_rank_group_sizes(frame: pd.DataFrame) -> list[int]:
    """Return LightGBM group sizes after deterministic user grouping."""
    if "user_id" not in frame.columns:
        raise KeyError("ranking feature table missing user_id")
    if frame.empty:
        return []
    return frame.groupby("user_id", sort=False).size().astype(int).tolist()

