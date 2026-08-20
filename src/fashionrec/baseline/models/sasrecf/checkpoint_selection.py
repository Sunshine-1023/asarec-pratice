"""Select a sequence checkpoint using one prediction per valid user-week."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id
from fashionrec.shared.metrics.ranking import map_at_k, mean_metric


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    checkpoint: str
    user_week_map_at_k: float


def load_valid_user_week_targets(valid_inter_path: str | Path) -> dict[str, set[str]]:
    """Load the complete valid-week item set for every user; no test path is accepted."""
    frame = pd.read_csv(
        valid_inter_path,
        sep="\t",
        usecols=["user_id:token", "item_id:token"],
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    targets: dict[str, set[str]] = {}
    for user_id, item_id in frame.itertuples(index=False, name=None):
        targets.setdefault(canonical_user_id(user_id), set()).add(canonical_item_id(item_id))
    if not targets:
        raise ValueError(f"Validation targets are empty: {valid_inter_path}")
    return targets


def load_ranked_recommendations(recall_csv: str | Path, *, k: int) -> dict[str, list[str]]:
    if k < 1:
        raise ValueError("k must be >= 1")
    frame = pd.read_csv(recall_csv, dtype={"user_id": "string", "item_id": "string"})
    required = {"user_id", "item_id", "rank"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Recall CSV missing columns: {sorted(missing)}")
    frame["user_id"] = frame["user_id"].map(canonical_user_id)
    frame["item_id"] = frame["item_id"].map(canonical_item_id)
    frame = frame.sort_values(["user_id", "rank", "item_id"], kind="mergesort")
    recommendations: dict[str, list[str]] = {}
    for user_id, group in frame.groupby("user_id", sort=True):
        recommendations[user_id] = group["item_id"].tolist()[:k]
    return recommendations


def user_week_map_at_k(
    targets: dict[str, set[str]],
    recommendations: dict[str, Sequence[str]],
    *,
    k: int,
) -> float:
    return mean_metric(
        [map_at_k(actual, recommendations.get(user_id, ()), k) for user_id, actual in targets.items()]
    )


def score_recall_csv(valid_inter_path: str | Path, recall_csv: str | Path, *, k: int) -> float:
    targets = load_valid_user_week_targets(valid_inter_path)
    recommendations = load_ranked_recommendations(recall_csv, k=k)
    return user_week_map_at_k(targets, recommendations, k=k)


def select_checkpoint_by_score(
    checkpoint_paths: Iterable[str | Path],
    score_fn: Callable[[Path], float],
    *,
    output_json: str | Path,
    selected_model_path: str | Path,
    k: int = 12,
) -> Path:
    """Score candidate files with a valid-only callback and materialize one stable winner."""
    candidates = sorted({Path(path).resolve() for path in checkpoint_paths}, key=str)
    if not candidates:
        raise ValueError("At least one checkpoint candidate is required")
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint candidates: {missing}")

    scores = [CheckpointScore(str(path), float(score_fn(path))) for path in candidates]
    winner = sorted(scores, key=lambda row: (-row.user_week_map_at_k, row.checkpoint))[0]
    winner_path = Path(winner.checkpoint)
    stable_path = Path(selected_model_path)
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    if winner_path != stable_path.resolve():
        shutil.copy2(winner_path, stable_path)

    payload = {
        "selection_split": "valid",
        "selection_unit": "user_week",
        "metric": f"MAP@{k}",
        "selected_source_checkpoint": str(winner_path),
        "selected_model_path": str(stable_path.resolve()),
        "scores": [asdict(row) for row in scores],
    }
    report_path = Path(output_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return stable_path
