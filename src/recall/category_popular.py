"""Category-aware popular-item recall channel."""  # 类别热门商品召回通道

from __future__ import annotations  # 启用延迟注解评估

from collections import defaultdict  # 导入默认字典
from dataclasses import dataclass  # 导入数据类
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas

from src.data.build_item_features import (  # 复用商品特征清洗逻辑
    RAW_ARTICLES_PATH,
    _clean_category_token,
    _normalize_item_id,
)


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集交互文件
DEFAULT_ITEM_FILE = Path("data/processed/hm_seq/hm_seq.item")  # 默认 RecBole 商品特征文件
FILTERED_ARTICLES_PATH = Path("data/raw/filtered/articles.csv")  # 过滤后商品元数据

CATEGORY_FIELDS = (  # 第一版使用的类别字段
    "product_type_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "colour_group_name",
)
WINDOW_WEEKS = (1, 2, 4)  # 时间衰减窗口（周）
WINDOW_WEIGHTS = (0.5, 0.3, 0.15)  # 各窗口融合权重
PER_BUCKET_TOP_K = 100  # 每个 (类别字段, 类别值) 桶保留的热门商品数
SEED_ITEMS = 10  # 用用户最近 N 个购买推断类别
CATEGORY_POPULAR_RECALL_TOP_K = 50  # 召回输出 Top-K


def _week_window_start(max_date: pd.Timestamp, weeks: int) -> pd.Timestamp:  # 计算含 max_date 的 N 周窗口起始日
    max_day = pd.Timestamp(max_date).normalize()  # 归一化到自然日
    return max_day - pd.Timedelta(days=weeks * 7 - 1)  # 含首尾共 weeks*7 天


def _load_interactions(*inter_paths: str | Path) -> pd.DataFrame:  # 读取交互并合并
    if not inter_paths:  # 未指定路径
        inter_paths = (DEFAULT_INTER_PATH,)  # 使用默认训练集

    frames: list[pd.DataFrame] = []  # 各文件 DataFrame
    for path in inter_paths:  # 遍历交互文件
        df = pd.read_csv(  # 读取商品、时间戳
            path,
            sep="\t",
            usecols=["item_id:token", "timestamp:float"],
        )
        df["item_id:token"] = df["item_id:token"].map(_normalize_item_id)  # 统一为 10 位商品 ID
        df["date"] = pd.to_datetime(df["timestamp:float"], unit="s").dt.normalize()  # 转自然日
        frames.append(df[["item_id:token", "date"]])  # 只保留所需列

    return pd.concat(frames, ignore_index=True)  # 合并全部交互


def _load_item_categories(  # 加载商品到类别字段的映射
    item_file: Path | None = None,
    articles_path: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Return {item_id: {field_name: category_value}} for CATEGORY_FIELDS."""
    item_file = item_file or DEFAULT_ITEM_FILE  # 默认 hm_seq.item
    token_columns = [f"{field}:token" for field in CATEGORY_FIELDS]  # RecBole 列名

    if item_file.exists():  # 优先读取已构建的 item 特征文件
        df = pd.read_csv(item_file, sep="\t", dtype={"item_id:token": "string"})
        missing = {"item_id:token", *token_columns}.difference(df.columns)
        if missing:
            raise ValueError(f"{item_file} missing required columns: {sorted(missing)}")
        source_df = df[["item_id:token", *token_columns]].copy()
    else:  # 回退到 articles.csv
        articles_path = articles_path or (
            FILTERED_ARTICLES_PATH if FILTERED_ARTICLES_PATH.exists() else RAW_ARTICLES_PATH
        )
        if not articles_path.exists():
            raise FileNotFoundError(
                f"Item feature file not found ({item_file}) and articles.csv missing ({articles_path})"
            )

        articles_df = pd.read_csv(articles_path, dtype={"article_id": "string"})
        missing = {"article_id", *CATEGORY_FIELDS}.difference(articles_df.columns)
        if missing:
            raise ValueError(f"{articles_path} missing required columns: {sorted(missing)}")

        source_df = articles_df[["article_id", *CATEGORY_FIELDS]].copy()
        source_df["item_id:token"] = source_df["article_id"].map(_normalize_item_id)
        source_df = source_df.drop(columns=["article_id"])
        source_df = source_df.rename(columns={field: f"{field}:token" for field in CATEGORY_FIELDS})
        for col in token_columns:
            source_df[col] = source_df[col].map(_clean_category_token)

    item_categories: dict[str, dict[str, str]] = {}  # 商品到类别映射
    for _, row in source_df.iterrows():
        item_id = _normalize_item_id(row["item_id:token"])
        item_categories[item_id] = {
            field: str(row[f"{field}:token"]) for field in CATEGORY_FIELDS
        }
    return item_categories


@dataclass(frozen=True)
class CategoryPopularIndex:  # 类别热门索引
    buckets: dict[str, dict[str, list[tuple[str, float]]]]  # field -> value -> ranked items
    item_categories: dict[str, dict[str, str]]  # item_id -> {field: value}


def build_category_popular_index(  # 构建类别热门索引
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_INTER_PATH,
    item_file: Path | None = None,
    articles_path: Path | None = None,
    category_fields: tuple[str, ...] = CATEGORY_FIELDS,
    window_weeks: tuple[int, ...] = WINDOW_WEEKS,
    window_weights: tuple[float, ...] = WINDOW_WEIGHTS,
    per_bucket_top_k: int = PER_BUCKET_TOP_K,
) -> CategoryPopularIndex:
    """
    Build per-category popularity buckets from recent-week train interactions.

    For each category field/value, rank items by:
      score = 0.5 * heat_1w + 0.3 * heat_2w + 0.15 * heat_4w
    """
    if len(window_weeks) != len(window_weights):
        raise ValueError("window_weeks and window_weights must have the same length")

    if isinstance(inter_paths, (str, Path)):
        paths = [inter_paths]
    else:
        paths = list(inter_paths)
    if not paths:
        paths = [DEFAULT_INTER_PATH]

    item_categories = _load_item_categories(item_file=item_file, articles_path=articles_path)
    if not item_categories:
        return CategoryPopularIndex(buckets={}, item_categories={})

    interactions = _load_interactions(*paths)
    if interactions.empty:
        return CategoryPopularIndex(buckets={}, item_categories=item_categories)

    feature_rows = [{"item_id:token": item_id, **categories} for item_id, categories in item_categories.items()]
    feature_df = pd.DataFrame(feature_rows)
    merged = interactions.merge(feature_df, on="item_id:token", how="inner")
    if merged.empty:
        return CategoryPopularIndex(buckets={}, item_categories=item_categories)

    max_date = merged["date"].max()
    scores: dict[tuple[str, str, str], float] = defaultdict(float)  # (field, value, item) -> score

    for weeks, weight in zip(window_weeks, window_weights):
        start_date = _week_window_start(max_date, weeks)
        window_df = merged[merged["date"] >= start_date]
        for field in category_fields:
            counts = window_df.groupby([field, "item_id:token"], observed=True).size()
            for (category_value, item_id), count in counts.items():
                scores[(field, str(category_value), str(item_id))] += weight * float(count)

    bucket_items: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for (field, category_value, item_id), score in scores.items():
        bucket_items[(field, category_value)].append((item_id, score))

    buckets: dict[str, dict[str, list[tuple[str, float]]]] = {field: {} for field in category_fields}
    for (field, category_value), items in bucket_items.items():
        ranked = sorted(items, key=lambda x: (-x[1], x[0]))[:per_bucket_top_k]
        buckets[field][category_value] = ranked

    return CategoryPopularIndex(buckets=buckets, item_categories=item_categories)


def recall_category_popular(  # 基于用户最近购买类别召回热门商品
    user_history: list[str],
    index: CategoryPopularIndex,
    seed_items: int = SEED_ITEMS,
    top_k: int = CATEGORY_POPULAR_RECALL_TOP_K,
) -> list[tuple[str, float]]:
    """Recall top-k items from categories inferred from the user's recent purchases."""
    if not user_history or not index.buckets:
        return []

    history_set = {_normalize_item_id(x) for x in user_history}
    seeds = [_normalize_item_id(x) for x in user_history[-seed_items:]]

    merged_scores: dict[str, float] = defaultdict(float)
    seen_buckets: set[tuple[str, str]] = set()

    for item_id in seeds:
        categories = index.item_categories.get(item_id, {})
        for field, category_value in categories.items():
            bucket_key = (field, category_value)
            if bucket_key in seen_buckets:
                continue
            seen_buckets.add(bucket_key)

            for cand_id, cand_score in index.buckets.get(field, {}).get(category_value, []):
                if cand_id in history_set:
                    continue
                merged_scores[cand_id] += cand_score

    ranked = sorted(merged_scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


if __name__ == "__main__":  # 脚本直接运行入口
    index = build_category_popular_index()
    sample_history = list(index.item_categories.keys())[:3]
    sample = recall_category_popular(sample_history, index, top_k=10)
    bucket_count = sum(len(values) for values in index.buckets.values())
    print(f"Category buckets: {bucket_count:,}")
    print(f"Items with categories: {len(index.item_categories):,}")
    print("Sample recall:", sample)
