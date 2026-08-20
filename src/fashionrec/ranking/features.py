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
SNAPSHOT_RANKING_COLUMNS = (  # 快照级训练表额外主键
    "snapshot_date",
    "group_id",
)


def _as_day(value: pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def resolve_snapshot_date(
    user_id: str,
    split: str,
    snapshot_dates: pd.Timestamp | str | Mapping[str, pd.Timestamp | str] | Mapping[tuple[str, str], pd.Timestamp | str],
) -> pd.Timestamp:
    """Resolve one user-split snapshot date from a scalar or mapping."""
    if not isinstance(snapshot_dates, Mapping):
        return _as_day(snapshot_dates)
    user_key = canonical_user_id(user_id)
    pair_key = (user_key, str(split).strip().lower())
    if pair_key in snapshot_dates:
        return _as_day(snapshot_dates[pair_key])  # type: ignore[index]
    if user_key in snapshot_dates:
        return _as_day(snapshot_dates[user_key])  # type: ignore[index]
    raise KeyError(f"missing snapshot_date for user_id={user_key!r} split={split!r}")


def ranking_group_id(user_id: str, snapshot_date: pd.Timestamp | str) -> str:
    """LightGBM group key: one list per user-snapshot."""
    return f"{canonical_user_id(user_id)}@{_as_day(snapshot_date).strftime('%Y-%m-%d')}"


def build_ranking_features(
    candidates: Iterable[Candidate],
    *,
    history_lengths: Mapping[str, int],
    channels: Iterable[str],
    targets: Mapping[str, set[str]] | None = None,
    snapshot_dates: pd.Timestamp | str | Mapping[str, pd.Timestamp | str] | Mapping[tuple[str, str], pd.Timestamp | str] | None = None,
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
        if snapshot_dates is not None:
            snapshot = resolve_snapshot_date(user_id, split, snapshot_dates)
            row["snapshot_date"] = snapshot
            row["group_id"] = ranking_group_id(user_id, snapshot)
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
    if snapshot_dates is not None:
        ordered_columns[2:2] = list(SNAPSHOT_RANKING_COLUMNS)
    for channel in channel_names:
        ordered_columns.extend([f"{channel}_present", f"{channel}_score", f"{channel}_rank"])
    if targets is not None:
        ordered_columns.append("label")
    return pd.DataFrame(rows, columns=ordered_columns)


def lambda_rank_group_sizes(frame: pd.DataFrame) -> list[int]:
    """Return LightGBM group sizes after deterministic user or user-snapshot grouping."""
    if frame.empty:
        return []
    group_col = "group_id" if "group_id" in frame.columns else "user_id"
    if group_col not in frame.columns:
        raise KeyError("ranking feature table missing user_id")
    return frame.groupby(group_col, sort=False).size().astype(int).tolist()

