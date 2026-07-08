"""Popular-item recall channel."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")


def build_popular_index(*inter_paths: str | Path) -> list[tuple[str, float]]:
    """Build a global popularity index from one or more interaction files."""
    if not inter_paths:
        inter_paths = (DEFAULT_INTER_PATH,)
    item_ids: list[str] = []
    for path in inter_paths:
        df = pd.read_csv(path, sep="\t", usecols=["item_id:token"])
        item_ids.extend(df["item_id:token"].astype(str).tolist())
    counts = Counter(item_ids)
    return [(item_id, float(count)) for item_id, count in counts.most_common()]


def recall_popular(
    popular_index: list[tuple[str, float]],
    user_history: set[str] | None = None,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Recall top-k popular items excluding the user's history."""
    history = {str(x) for x in user_history} if user_history else set()
    results: list[tuple[str, float]] = []
    for item_id, score in popular_index:
        if item_id in history:
            continue
        results.append((item_id, score))
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    index = build_popular_index()
    sample = recall_popular(index, user_history=set(), top_k=10)
    print(f"Popular index size: {len(index):,}")
    print("Top-10 sample:", sample)

