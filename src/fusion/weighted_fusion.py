"""Weighted fusion for multi-channel candidate lists."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd


def build_user_history(*inter_paths: str | Path) -> dict[str, list[str]]:
    """Build per-user ordered history from one or more .inter files."""
    frames = []
    for path in inter_paths:
        df = pd.read_csv(path, sep="\t", usecols=["user_id:token", "item_id:token", "timestamp:float"])
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["user_id:token", "timestamp:float"])
    history = (
        merged.groupby("user_id:token")["item_id:token"]
        .apply(lambda s: [str(x) for x in s.tolist()])
        .to_dict()
    )
    return history


def load_channel_recall_csv(
    path: str | Path,
    user_col: str = "user_id",
    item_col: str = "item_id",
    score_col: str = "score",
    rank_col: str = "rank",
) -> dict[str, list[tuple[str, float, int]]]:
    """
    Load a channel recall csv as:
      {user_id: [(item_id, score, rank), ...]}
    """
    path = Path(path)
    if not path.exists():
        return {}

    rows_by_user: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row[user_col])
            iid = str(row[item_col])
            score = float(row.get(score_col, 0.0))
            rank = int(row.get(rank_col, 999999))
            rows_by_user[uid].append((iid, score, rank))

    for uid in rows_by_user:
        rows_by_user[uid].sort(key=lambda x: x[2])
    return dict(rows_by_user)


def fuse_candidates(
    user_id: str,
    user_history: set[str],
    channel_candidates: dict[str, list[tuple[str, float]]],
    channel_weights: dict[str, float],
    top_k: int = 12,
) -> list[tuple[str, float]]:
    """
    Weighted rank fusion.

    For each channel list, candidate contribution:
      weight * (1 / (rank + 1))
    Final score is summed over channels.
    """
    history = {str(x) for x in user_history}
    merged_scores: dict[str, float] = defaultdict(float)

    for channel, candidates in channel_candidates.items():
        w = channel_weights.get(channel, 0.0)
        if w <= 0:
            continue
        for rank, (item_id, _) in enumerate(candidates):
            item_id = str(item_id)
            if item_id in history:
                continue
            merged_scores[item_id] += w * (1.0 / (rank + 1))

    ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

