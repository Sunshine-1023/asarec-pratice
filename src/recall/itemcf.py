"""ItemCF recall channel."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from pathlib import Path

import pandas as pd


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")


def build_itemcf_index(
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_INTER_PATH,
    min_cooccur: int = 2,
    top_sim_k: int = 100,
    max_user_items: int = 50,
) -> dict[str, dict[str, float]]:
    """Build item-to-item similarity index from one or more interaction files."""
    if isinstance(inter_paths, (str, Path)):
        paths = [inter_paths]
    else:
        paths = list(inter_paths)
    if not paths:
        paths = [DEFAULT_INTER_PATH]

    frames = []
    for path in paths:
        frames.append(
            pd.read_csv(
                path,
                sep="\t",
                usecols=["user_id:token", "item_id:token", "timestamp:float"],
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["user_id:token", "timestamp:float"])

    user_items: list[list[str]] = []
    for _, group in df.groupby("user_id:token", sort=False):
        items = group["item_id:token"].astype(str).tolist()
        if len(items) > max_user_items:
            items = items[-max_user_items:]

        seen: set[str] = set()
        unique_items: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            unique_items.append(item)
        if unique_items:
            user_items.append(unique_items)

    item_count: defaultdict[str, int] = defaultdict(int)
    cooccur: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    for items in user_items:
        for item in items:
            item_count[item] += 1
        for i, item_i in enumerate(items):
            for item_j in items[i + 1 :]:
                cooccur[item_i][item_j] += 1
                cooccur[item_j][item_i] += 1

    sim_index: dict[str, dict[str, float]] = {}
    for item_i, neighbors in cooccur.items():
        scored_neighbors: dict[str, float] = {}
        for item_j, cij in neighbors.items():
            if cij < min_cooccur:
                continue
            # Cosine-style normalization by item frequency.
            score = cij / sqrt(item_count[item_i] * item_count[item_j])
            scored_neighbors[item_j] = score
        if scored_neighbors:
            top_neighbors = sorted(
                scored_neighbors.items(), key=lambda x: x[1], reverse=True
            )[:top_sim_k]
            sim_index[item_i] = dict(top_neighbors)
    return sim_index


def recall_itemcf(
    user_history: list[str] | set[str],
    itemcf_index: dict[str, dict[str, float]],
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Recall top-k items by aggregating similar neighbors of history items."""
    history_list = [str(x) for x in user_history]
    history_set = set(history_list)
    scores: defaultdict[str, float] = defaultdict(float)

    for item in history_list:
        for neighbor, sim in itemcf_index.get(item, {}).items():
            if neighbor in history_set:
                continue
            scores[neighbor] += sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    index = build_itemcf_index()
    # Use existing indexed items to avoid format mismatch in demo IDs.
    sample_history = list(index.keys())[:3]
    sample = recall_itemcf(sample_history, index, top_k=10)
    print(f"ItemCF index size: {len(index):,}")
    print("Top-10 sample:", sample)

