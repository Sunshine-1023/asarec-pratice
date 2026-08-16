"""Build configured recall channels without coupling callers to index details."""  # 召回注册表

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

from fashionrec.recall.base import RecallChannel, RecallResult
from fashionrec.recall.category_popular import (
    CategoryPopularIndex,
    SEED_ITEMS as CATEGORY_SEED_ITEMS,
    build_category_popular_index,
    recall_category_popular,
)
from fashionrec.recall.item2item import (
    COOCCUR_WEEKS,
    SEED_ITEMS as ITEM2ITEM_SEED_ITEMS,
    TOP_SIM_K,
    build_item2item_index,
    recall_item2item,
)
from fashionrec.recall.popular import build_popular_index, recall_popular


@dataclass(slots=True)
class PopularChannel:
    index: list[tuple[str, float]]
    name: str = "popular"

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        return recall_popular(self.index, user_history=set(history), top_k=top_k)


@dataclass(slots=True)
class CategoryPopularChannel:
    index: CategoryPopularIndex
    seed_items: int = CATEGORY_SEED_ITEMS
    name: str = "category_popular"

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        return recall_category_popular(history, self.index, seed_items=self.seed_items, top_k=top_k)


@dataclass(slots=True)
class Item2ItemChannel:
    index: dict[str, dict[str, float]]
    seed_items: int = ITEM2ITEM_SEED_ITEMS
    name: str = "item2item"

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        return recall_item2item(history, self.index, seed_items=self.seed_items, top_k=top_k)


@dataclass(slots=True)
class PrecomputedChannel:  # 模型召回等已落盘通道适配器
    name: str
    candidates_by_user: Mapping[str, list[tuple[str, float]]]

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        return list(self.candidates_by_user.get(user_id, []))[:top_k]


def build_rule_channel_registry(  # 一次构建索引，供全部用户复用
    history_paths: list[str | Path],
    *,
    item2item_cooccur_weeks: int = COOCCUR_WEEKS,
    item2item_top_sim_k: int = TOP_SIM_K,
    item2item_seed_items: int = ITEM2ITEM_SEED_ITEMS,
    category_seed_items: int = CATEGORY_SEED_ITEMS,
) -> dict[str, RecallChannel]:
    paths = [Path(path) for path in history_paths]
    if not paths:
        raise ValueError("history_paths must not be empty")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing history file: {path}")

    channels: list[RecallChannel] = [
        PopularChannel(build_popular_index(*paths)),
        CategoryPopularChannel(build_category_popular_index(paths), seed_items=category_seed_items),
        Item2ItemChannel(
            build_item2item_index(
                paths,
                cooccur_weeks=item2item_cooccur_weeks,
                top_sim_k=item2item_top_sim_k,
            ),
            seed_items=item2item_seed_items,
        ),
    ]
    return {channel.name: channel for channel in channels}


def select_channels(
    registry: dict[str, RecallChannel],
    channel_names: tuple[str, ...] | list[str],
) -> dict[str, RecallChannel]:
    missing = [name for name in channel_names if name not in registry]
    if missing:
        raise KeyError(f"Unknown recall channels: {missing}; available={sorted(registry)}")
    return {name: registry[name] for name in channel_names}
