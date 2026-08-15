"""Generate and materialize candidates through the shared recall contract."""  # 候选生成唯一入口

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.domain.candidates import Candidate
from src.domain.ids import canonical_user_id
from src.recall.base import RecallChannel


CANDIDATE_COLUMNS = ("user_id", "item_id", "channel", "score", "rank", "split")  # 稳定表结构


def generate_candidates(
    *,
    eval_users: Iterable[str],
    user_history: Mapping[str, list[str]],
    channels: Mapping[str, RecallChannel],
    split: str,
    top_k_by_channel: Mapping[str, int],
) -> list[Candidate]:
    """Generate normalized candidate rows for every user and channel."""
    rows: list[Candidate] = []
    for raw_user_id in eval_users:
        user_id = canonical_user_id(raw_user_id)
        history = list(user_history.get(user_id, []))
        for channel_name, channel in channels.items():
            top_k = int(top_k_by_channel.get(channel_name, 0))
            if top_k < 1:
                raise ValueError(f"top_k for channel {channel_name!r} must be >= 1")
            for rank, (item_id, score) in enumerate(channel.recall(user_id, history, top_k), start=1):
                if rank > top_k:  # 防御不遵守 top_k 的外部通道
                    break
                rows.append(Candidate(user_id, item_id, channel_name, score, rank, split))
    return rows


def write_candidate_csv(candidates: Iterable[Candidate], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANDIDATE_COLUMNS))
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate.as_dict())
    return path


def read_candidate_csv(path: str | Path) -> list[Candidate]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Candidate file not found: {source}")
    with source.open("r", newline="", encoding="utf-8") as handle:
        return [Candidate(**row) for row in csv.DictReader(handle)]

