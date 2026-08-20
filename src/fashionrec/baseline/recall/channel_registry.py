"""Build configured recall channels without coupling callers to index details."""  # 召回注册表

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import pandas as pd

from fashionrec.baseline.recall.base import RecallChannel, RecallResult
from fashionrec.baseline.recall.category_popular import (
    CategoryPopularIndex,
    SEED_ITEMS as CATEGORY_SEED_ITEMS,
    build_category_popular_index,
    recall_category_popular,
)
from fashionrec.baseline.recall.item2item import (
    COOCCUR_WEEKS,
    DEFAULT_SIMILARITY_MODE,
    Item2ItemSimilarityMode,
    SEED_ITEMS as ITEM2ITEM_SEED_ITEMS,
    TOP_SIM_K,
    build_item2item_index,
    recall_item2item,
)
from fashionrec.baseline.recall.content import (  # 内容召回
    CONTENT_SEED_ITEMS,
    ContentIndex,
    build_content_index,
    recall_content,
)
from fashionrec.baseline.recall.repurchase import (  # 复购召回
    RepurchaseIndex,
    build_repurchase_index,
    recall_repurchase,
)
from fashionrec.baseline.recall.style import (  # 款式召回
    STYLE_SEED_ITEMS,
    StyleIndex,
    build_style_index,
    recall_style,
)
from fashionrec.baseline.recall.popular import (  # 热门召回
    PopularIndex,
    build_popular_index,
    build_user_cohort_lookup,
    recall_popular,
)


@dataclass(slots=True)
class PopularChannel:
    index: PopularIndex
    cohort_lookup: dict[str, dict[str, str]]
    name: str = "popular"

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        return recall_popular(
            self.index,
            user_history=set(history),
            top_k=top_k,
            user_id=user_id,
            cohort_lookup=self.cohort_lookup,
        )


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
class RepurchaseChannel:
    index: RepurchaseIndex
    name: str = "repurchase"

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        return recall_repurchase(user_id, history, self.index, top_k=top_k)


@dataclass(slots=True)
class StyleChannel:
    index: StyleIndex
    seed_items: int = STYLE_SEED_ITEMS
    name: str = "style"

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        return recall_style(history, self.index, seed_items=self.seed_items, top_k=top_k)


@dataclass(slots=True)
class ContentChannel:
    index: ContentIndex
    seed_items: int = CONTENT_SEED_ITEMS
    name: str = "content"

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        return recall_content(history, self.index, seed_items=self.seed_items, top_k=top_k)


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
    item2item_similarity_mode: Item2ItemSimilarityMode = DEFAULT_SIMILARITY_MODE,
    category_seed_items: int = CATEGORY_SEED_ITEMS,
    item_file: Path | None = None,
    customers_path: Path | None = None,
    articles_path: Path | None = None,
    as_of: pd.Timestamp | str | None = None,
    channel_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, RecallChannel]:
    paths = [Path(path) for path in history_paths]
    if not paths:
        raise ValueError("history_paths must not be empty")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing history file: {path}")

    supported = {"popular", "category_popular", "item2item", "repurchase", "style", "content"}
    requested = supported if channel_names is None else set(channel_names)
    unknown = sorted(requested.difference(supported))
    if unknown:
        raise ValueError(f"Unknown rule recall channels: {unknown}")
    channels: list[RecallChannel] = []
    if "popular" in requested:
        channels.append(
            PopularChannel(
                build_popular_index(*paths, customers_path=customers_path, articles_path=articles_path, as_of=as_of),
                build_user_cohort_lookup(customers_path),
            )
        )
    if "category_popular" in requested:
        channels.append(
            CategoryPopularChannel(
                build_category_popular_index(paths, item_file=item_file, articles_path=articles_path, as_of=as_of),
                seed_items=category_seed_items,
            )
        )
    if "item2item" in requested:
        channels.append(
            Item2ItemChannel(
                build_item2item_index(
                    paths,
                    cooccur_weeks=item2item_cooccur_weeks,
                    top_sim_k=item2item_top_sim_k,
                    similarity_mode=item2item_similarity_mode,
                    as_of=as_of,
                ),
                seed_items=item2item_seed_items,
            )
        )
    if "repurchase" in requested:
        channels.append(RepurchaseChannel(build_repurchase_index(paths, as_of=as_of)))
    has_articles = articles_path is not None and Path(articles_path).is_file()
    if "style" in requested and has_articles:
        channels.append(StyleChannel(build_style_index(paths, articles_path=articles_path, as_of=as_of)))
    if "content" in requested and has_articles:
        channels.append(ContentChannel(build_content_index(articles_path, inter_paths=paths, as_of=as_of)))
    return {channel.name: channel for channel in channels}


def select_channels(
    registry: dict[str, RecallChannel],
    channel_names: tuple[str, ...] | list[str],
) -> dict[str, RecallChannel]:
    missing = [name for name in channel_names if name not in registry]
    if missing:
        raise KeyError(f"Unknown recall channels: {missing}; available={sorted(registry)}")
    return {name: registry[name] for name in channel_names}
