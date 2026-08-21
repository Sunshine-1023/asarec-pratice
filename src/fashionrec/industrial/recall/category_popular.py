"""Category-aware popular-item recall channel."""  # 类别热门商品召回通道

from __future__ import annotations  # 启用延迟注解评估

import sys  # 导入系统模块用于路径注入
from collections import defaultdict  # 聚合分
from dataclasses import dataclass  # 导入数据类
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas

if __package__ is None or __package__ == "":  # 若以脚本方式直接运行
    project_root = Path(__file__).resolve().parents[2]  # 定位项目根目录
    if str(project_root) not in sys.path:  # 若根目录不在搜索路径中
        sys.path.insert(0, str(project_root))  # 注入项目根目录到 sys.path

from fashionrec.industrial.data.build_item_features import (  # 复用商品特征清洗逻辑
    RAW_ARTICLES_PATH,  # 原始商品元数据路径
    clean_category_token,  # 清洗类别 token
)  # 商品特征工具导入结束
from fashionrec.industrial.recall.window_scores import (  # 共享多窗口 rank 归一化
    DEFAULT_WINDOW_WEEKS,
    DEFAULT_WINDOW_WEIGHTS,
    build_window_popularity,
    filter_interactions_as_of,
    rank_score_items,
)
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集交互文件
DEFAULT_ITEM_FILE = Path("data/processed/hm_seq/hm_seq.item")  # 默认 RecBole 商品特征文件
FILTERED_ARTICLES_PATH = Path("data/raw/filtered/articles.csv")  # 过滤后商品元数据

CATEGORY_FIELDS = (  # 第一版使用的类别字段
    "product_type_name",  # 商品类型名称
    "department_name",  # 部门名称
    "section_name",  # 分区名称
    "garment_group_name",  # 服装组名称
    "colour_group_name",  # 颜色组名称
)  # 类别字段元组结束
WINDOW_WEEKS = DEFAULT_WINDOW_WEEKS  # 1/2/4/12 周
WINDOW_WEIGHTS = DEFAULT_WINDOW_WEIGHTS  # rank 归一化后加权
PER_BUCKET_TOP_K = 100  # 每个 (类别字段, 类别值) 桶保留的热门商品数
SEED_ITEMS = 10  # 用用户最近 N 个购买推断类别
CATEGORY_POPULAR_RECALL_TOP_K = 50  # 召回输出 Top-K


def _load_interactions(*inter_paths: str | Path) -> pd.DataFrame:  # 读取交互并合并
    if not inter_paths:  # 未指定路径
        inter_paths = (DEFAULT_INTER_PATH,)  # 使用默认训练集

    frames: list[pd.DataFrame] = []  # 各文件 DataFrame
    for path in inter_paths:  # 遍历交互文件
        df = pd.read_csv(  # 读取商品、时间戳
            path,  # 交互文件路径
            sep="\t",  # 制表符分隔
            usecols=["user_id:token", "item_id:token", "timestamp:float"],
            dtype={"user_id:token": "string", "item_id:token": "string"},
        )  # 结束 read_csv 调用
        df["user_id:token"] = df["user_id:token"].map(canonical_user_id)
        df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 统一为 10 位商品 ID
        df["date"] = pd.to_datetime(df["timestamp:float"], unit="s").dt.normalize()  # 转自然日
        frames.append(df[["user_id:token", "item_id:token", "date"]])

    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["user_id:token", "item_id:token", "date"], keep="first"
    )  # 同日同 SKU 只计一次购买事件


def _load_item_categories(  # 加载商品到类别字段的映射
    item_file: Path | None = None,  # RecBole 商品特征文件路径
    articles_path: Path | None = None,  # 原始 articles.csv 路径
) -> dict[str, dict[str, str]]:  # 返回 {item_id: {field: value}}
    """Return {item_id: {field_name: category_value}} for CATEGORY_FIELDS."""  # 返回商品到类别字段值的映射
    item_file = item_file or DEFAULT_ITEM_FILE  # 默认 hm_seq.item
    token_columns = [f"{field}:token" for field in CATEGORY_FIELDS]  # RecBole 列名

    if item_file.exists():  # 优先读取已构建的 item 特征文件
        df = pd.read_csv(item_file, sep="\t", dtype={"item_id:token": "string"})  # 读取商品特征文件
        missing = {"item_id:token", *token_columns}.difference(df.columns)  # 检查缺失列
        if missing:  # 若存在缺失列
            raise ValueError(f"{item_file} missing required columns: {sorted(missing)}")  # 抛出列缺失异常
        source_df = df[["item_id:token", *token_columns]].copy()  # 复制所需列
    else:  # 回退到 articles.csv
        articles_path = articles_path or (  # 选择 articles 路径
            FILTERED_ARTICLES_PATH if FILTERED_ARTICLES_PATH.exists() else RAW_ARTICLES_PATH  # 优先过滤后文件
        )  # 路径选择结束
        if not articles_path.exists():  # 若 articles 文件也不存在
            raise FileNotFoundError(  # 抛出文件未找到异常
                f"Item feature file not found ({item_file}) and articles.csv missing ({articles_path})"  # 提示两个文件均缺失
            )  # 异常消息结束

        articles_df = pd.read_csv(articles_path, dtype={"article_id": "string"})  # 读取 articles 元数据
        missing = {"article_id", *CATEGORY_FIELDS}.difference(articles_df.columns)  # 检查缺失列
        if missing:  # 若存在缺失列
            raise ValueError(f"{articles_path} missing required columns: {sorted(missing)}")  # 抛出列缺失异常

        source_df = articles_df[["article_id", *CATEGORY_FIELDS]].copy()  # 复制所需列
        source_df["item_id:token"] = source_df["article_id"].map(canonical_item_id)  # 规范化商品 ID
        source_df = source_df.drop(columns=["article_id"])  # 删除原始 article_id 列
        source_df = source_df.rename(columns={field: f"{field}:token" for field in CATEGORY_FIELDS})  # 重命名为 token 列
        for col in token_columns:  # 遍历类别 token 列
            source_df[col] = source_df[col].map(clean_category_token)  # 清洗类别 token

    item_categories: dict[str, dict[str, str]] = {}  # 商品到类别映射
    for _, row in source_df.iterrows():  # 逐行构建映射
        item_id = canonical_item_id(row["item_id:token"])  # 规范化商品 ID
        item_categories[item_id] = {  # 写入该商品的类别字典
            field: str(row[f"{field}:token"]) for field in CATEGORY_FIELDS  # 各字段的类别值
        }  # 类别字典结束
    return item_categories  # 返回商品类别映射


@dataclass(frozen=True)  # 不可变数据类
class CategoryPopularIndex:  # 类别热门索引
    buckets: dict[str, dict[str, list[tuple[str, float]]]]  # field -> value -> ranked items
    item_categories: dict[str, dict[str, str]]  # item_id -> {field: value}


def build_category_popular_index(  # 构建类别热门索引
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_INTER_PATH,  # 交互文件路径
    item_file: Path | None = None,  # 商品特征文件路径
    articles_path: Path | None = None,  # articles 元数据路径
    category_fields: tuple[str, ...] = CATEGORY_FIELDS,  # 类别字段列表
    window_weeks: tuple[int, ...] = WINDOW_WEEKS,  # 时间窗口周数
    window_weights: tuple[float, ...] = WINDOW_WEIGHTS,  # 各窗口融合权重
    per_bucket_top_k: int = PER_BUCKET_TOP_K,  # 每个桶保留的热门商品数
    as_of: pd.Timestamp | str | None = None,  # as-of 预测日
) -> CategoryPopularIndex:  # 返回类别热门索引对象
    """Build per-category buckets using rank-normalized multi-window heat."""  # 类别桶内多窗口 rank 归一化
    if len(window_weeks) != len(window_weights):  # 校验窗口与权重长度
        raise ValueError("window_weeks and window_weights must have the same length")  # 长度不一致则报错

    if isinstance(inter_paths, (str, Path)):  # 单路径
        paths = [inter_paths]  # 包装成列表
    else:  # 多路径
        paths = list(inter_paths)  # 转为列表
    if not paths:  # 若路径列表为空
        paths = [DEFAULT_INTER_PATH]  # 使用默认训练集路径

    item_categories = _load_item_categories(item_file=item_file, articles_path=articles_path)  # 加载商品类别映射
    if not item_categories:  # 若无商品类别
        return CategoryPopularIndex(buckets={}, item_categories={})  # 返回空索引

    interactions = _load_interactions(*paths)  # 加载交互数据
    interactions = filter_interactions_as_of(interactions, as_of)  # as-of 截断
    if interactions.empty:  # 若无交互记录
        return CategoryPopularIndex(buckets={}, item_categories=item_categories)  # 返回空桶但保留类别映射

    feature_rows = [{"item_id:token": item_id, **categories} for item_id, categories in item_categories.items()]  # 构建特征行
    feature_df = pd.DataFrame(feature_rows)  # 转为 DataFrame
    merged = interactions.merge(feature_df, on="item_id:token", how="inner")  # 内连接交互与类别
    if merged.empty:  # 若合并后无数据
        return CategoryPopularIndex(buckets={}, item_categories=item_categories)  # 返回空桶

    buckets: dict[str, dict[str, list[tuple[str, float]]]] = {field: {} for field in category_fields}  # 初始化桶结构
    for field in category_fields:  # 每个类别字段
        for category_value, bucket_df in merged.groupby(field, observed=True):  # 每个类别值
            scores = build_window_popularity(  # 窗口 rank 融合
                bucket_df,
                item_col="item_id:token",
                window_weeks=window_weeks,
                window_weights=window_weights,
                as_of=as_of,
            )
            if not scores:  # 空桶
                continue  # 跳过
            ranked = rank_score_items(scores)[:per_bucket_top_k]  # Top-K
            buckets[field][str(category_value)] = ranked  # 写入

    return CategoryPopularIndex(buckets=buckets, item_categories=item_categories)  # 返回类别热门索引


def recall_category_popular(  # 基于用户最近购买类别召回热门商品
    user_history: list[str],  # 用户历史交互商品列表
    index: CategoryPopularIndex,  # 类别热门索引
    seed_items: int = SEED_ITEMS,  # 种子商品数量
    top_k: int = CATEGORY_POPULAR_RECALL_TOP_K,  # 召回数量上限
) -> list[tuple[str, float]]:  # 返回商品 ID 与聚合分数列表
    """Recall top-k items from categories inferred from the user's recent purchases."""  # 根据最近购买推断类别并召回热门商品
    if not user_history or not index.buckets:  # 若无历史或桶为空
        return []  # 返回空列表

    seeds = [canonical_item_id(x) for x in user_history[-seed_items:]]  # 最近 seed_items 个商品作种子

    merged_scores: dict[str, float] = defaultdict(float)  # 候选商品聚合分
    seen_buckets: set[tuple[str, str]] = set()  # 已访问的 (字段, 类别值) 桶

    for item_id in seeds:  # 遍历种子商品
        categories = index.item_categories.get(item_id, {})  # 获取商品类别
        for field, category_value in categories.items():  # 遍历各类别字段
            bucket_key = (field, category_value)  # 桶键
            if bucket_key in seen_buckets:  # 若桶已访问
                continue  # 跳过重复桶
            seen_buckets.add(bucket_key)  # 标记桶已访问

            for cand_id, cand_score in index.buckets.get(field, {}).get(category_value, []):  # 遍历桶内候选
                merged_scores[cand_id] += cand_score  # 累加类别热门分（含复购同款）

    ranked = sorted(merged_scores.items(), key=lambda x: (-x[1], x[0]))  # 按分数降序排序
    return ranked[:top_k]  # 返回 Top-K


if __name__ == "__main__":  # 脚本直接运行入口
    index = build_category_popular_index()  # 构建类别热门索引
    sample_history = list(index.item_categories.keys())[:3]  # 取 3 个商品作为示例历史
    sample = recall_category_popular(sample_history, index, top_k=10)  # 召回 Top-10 示例
    bucket_count = sum(len(values) for values in index.buckets.values())  # 统计类别桶数量
    print(f"Category buckets: {bucket_count:,}")  # 打印类别桶数量
    print(f"Items with categories: {len(index.item_categories):,}")  # 打印有类别商品数
    print("Sample recall:", sample)  # 打印示例召回结果
