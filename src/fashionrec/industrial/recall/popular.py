"""Popular-item recall channel with multi-window rank-normalized popularity."""  # 多窗口 rank 归一化热门召回

from __future__ import annotations  # 延迟注解

from dataclasses import dataclass, field  # 索引结构
from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.industrial.data.customer_features import (  # 冷启动 age_bucket
    DEFAULT_CUSTOMERS,
    load_customers_table,
    parse_age,
)
from fashionrec.industrial.data.item_features import UNKNOWN_TOKEN, clean_category_token  # 类别 token
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # ID
from fashionrec.industrial.recall.window_scores import (  # 共享窗口逻辑
    DEFAULT_WINDOW_WEEKS,
    DEFAULT_WINDOW_WEIGHTS,
    build_window_popularity,
    filter_interactions_as_of,
    rank_score_items,
)


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集
POPULAR_RECALL_TOP_K = 50  # 全局热门召回 Top-K
WINDOW_WEEKS = DEFAULT_WINDOW_WEEKS  # 1/2/4/12 周
WINDOW_WEIGHTS = DEFAULT_WINDOW_WEIGHTS  # rank 归一化后加权
COHORT_DIMENSIONS = ("age_bucket", "index_group_name", "product_group_name", "channel")  # 人群/分段键
SEGMENT_ITEM_FIELDS = ("index_group_name", "product_group_name")  # 商品属性分段


@dataclass(frozen=True)  # 热门索引
class PopularIndex:  # 全局 + 人群/分段
    global_ranked: list[tuple[str, float]]  # 全体热门
    cohort_ranked: dict[tuple[str, str], list[tuple[str, float]]] = field(default_factory=dict)  # (dim,value)

    def __iter__(self):  # 兼容旧代码直接迭代 global
        return iter(self.global_ranked)


def _load_interactions(*inter_paths: str | Path) -> pd.DataFrame:  # 读交互（含 user）
    if not inter_paths:  # 默认路径
        inter_paths = (DEFAULT_INTER_PATH,)  # 训练集
    frames: list[pd.DataFrame] = []  # 收集
    for path in inter_paths:  # 每个文件
        df = pd.read_csv(  # 读 TSV
            path,
            sep="\t",
            usecols=["user_id:token", "item_id:token", "timestamp:float"],
            dtype={"user_id:token": "string", "item_id:token": "string"},
        )
        df["user_id:token"] = df["user_id:token"].map(canonical_user_id)  # 用户
        df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 商品
        df["date"] = pd.to_datetime(df["timestamp:float"], unit="s").dt.normalize()  # 自然日
        frames.append(df[["user_id:token", "item_id:token", "date"]])  # 保留列
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["user_id:token", "item_id:token", "date"], keep="first"
    )  # 同日同 SKU 只计一次购买事件


def _load_user_age_buckets(customers_path: Path | None) -> dict[str, str]:  # user -> age_bucket
    if customers_path is None:  # 未提供
        return {}  # 空映射
    customers_path = Path(customers_path)  # 规范化
    if not customers_path.is_file():  # 缺文件
        return {}  # 空
    customers = load_customers_table(customers_path)  # 读表
    mapping: dict[str, str] = {}  # 结果
    for row in customers.to_dict(orient="records"):  # 逐用户
        user_id = canonical_user_id(row["user_id"])  # 规范
        _age, bucket, _missing = parse_age(row.get("age"))  # 分桶
        if bucket != UNKNOWN_TOKEN:  # 有效桶
            mapping[user_id] = bucket  # 记录
    return mapping  # 返回


def _load_item_segments(articles_path: Path | None) -> pd.DataFrame:  # item_id -> 分段字段
    if articles_path is None:  # 未提供
        return pd.DataFrame(columns=["item_id:token", *SEGMENT_ITEM_FIELDS])  # 空表
    articles_path = Path(articles_path)  # 规范化
    if not articles_path.is_file():  # 缺文件
        return pd.DataFrame(columns=["item_id:token", *SEGMENT_ITEM_FIELDS])  # 空表
    usecols = ["article_id", *SEGMENT_ITEM_FIELDS]  # 所需列
    articles = pd.read_csv(  # 读 CSV
        articles_path,
        dtype={"article_id": "string"},
        usecols=lambda name: name in usecols,
    )
    if "article_id" not in articles.columns:  # 主键
        raise ValueError("articles must contain article_id")  # 报错
    frame = pd.DataFrame({"item_id:token": articles["article_id"].map(canonical_item_id)})  # SKU
    for col in SEGMENT_ITEM_FIELDS:  # 各分段
        if col not in articles.columns:  # 缺列
            frame[col] = UNKNOWN_TOKEN  # unknown
        else:  # 有列
            frame[col] = articles[col].map(clean_category_token)  # 清洗
    return frame  # 返回


def _build_cohort_ranked(  # 人群/分段热门
    frame: pd.DataFrame,  # 交互（含 user/item/date）
    *,
    dimension: str,  # age_bucket / index_group_name / ...
    values: pd.Series,  # 与 frame 等长的分组值
    window_weeks: tuple[int, ...],
    window_weights: tuple[float, ...],
    as_of: pd.Timestamp | str | None,
) -> dict[str, list[tuple[str, float]]]:  # value -> ranked
    enriched = frame.copy()  # 拷贝
    enriched["_cohort_value"] = values.astype(str)  # 分组键
    enriched = enriched[enriched["_cohort_value"].ne("") & enriched["_cohort_value"].ne(UNKNOWN_TOKEN)]  # 有效值
    if enriched.empty:  # 无可分组行
        return {}  # 空
    ranked_by_value: dict[str, list[tuple[str, float]]] = {}  # 收集
    for cohort_value, group in enriched.groupby("_cohort_value", sort=True):  # 各 cohort
        scores = build_window_popularity(  # 窗口融合
            group,
            item_col="item_id:token",
            window_weeks=window_weeks,
            window_weights=window_weights,
            as_of=as_of,
        )
        if scores:  # 非空
            ranked_by_value[str(cohort_value)] = rank_score_items(scores)  # 排序
    return ranked_by_value  # 返回


def build_popular_index(  # 构建全局 + 人群热门索引
    *inter_paths: str | Path,  # 交互文件
    window_weeks: tuple[int, ...] = WINDOW_WEEKS,  # 窗口
    window_weights: tuple[float, ...] = WINDOW_WEIGHTS,  # 权重
    as_of: pd.Timestamp | str | None = None,  # as-of 预测日
    customers_path: Path | None = DEFAULT_CUSTOMERS,  # 用户 age_bucket
    articles_path: Path | None = Path("data/raw/articles.csv"),  # 商品分段
) -> PopularIndex:  # 热门索引
    """Build rank-normalized multi-window popularity with optional cohort buckets."""  # 多窗口 rank 归一化 + 人群桶
    if len(window_weeks) != len(window_weights):  # 校验
        raise ValueError("window_weeks and window_weights must have the same length")  # 报错

    frame = _load_interactions(*inter_paths)  # 读交互
    frame = filter_interactions_as_of(frame, as_of)  # as-of 截断
    if frame.empty:  # 无数据
        return PopularIndex(global_ranked=[], cohort_ranked={})  # 空索引

    global_scores = build_window_popularity(  # 全体
        frame,
        item_col="item_id:token",
        window_weeks=window_weeks,
        window_weights=window_weights,
        as_of=as_of,
    )
    global_ranked = rank_score_items(global_scores)  # 排序

    cohort_ranked: dict[tuple[str, str], list[tuple[str, float]]] = {}  # 人群/分段

    age_buckets = _load_user_age_buckets(customers_path)  # user -> bucket
    if age_buckets:  # 有用户分桶
        bucket_series = frame["user_id:token"].map(lambda user: age_buckets.get(canonical_user_id(user), ""))  # 映射
        for value, ranked in _build_cohort_ranked(  # age_bucket 层
            frame,
            dimension="age_bucket",
            values=bucket_series,
            window_weeks=window_weeks,
            window_weights=window_weights,
            as_of=as_of,
        ).items():
            cohort_ranked[("age_bucket", value)] = ranked  # 写入

    segments = _load_item_segments(articles_path)  # 商品分段
    if not segments.empty:  # 有 articles
        merged = frame.merge(segments, on="item_id:token", how="inner")  # 带属性
        for field in SEGMENT_ITEM_FIELDS:  # index_group / product_group
            if field not in merged.columns:  # 缺列
                continue  # 跳过
            for value, ranked in _build_cohort_ranked(  # 分段层
                merged,
                dimension=field,
                values=merged[field],
                window_weeks=window_weeks,
                window_weights=window_weights,
                as_of=as_of,
            ).items():
                cohort_ranked[(field, value)] = ranked  # 写入

    return PopularIndex(global_ranked=global_ranked, cohort_ranked=cohort_ranked)  # 返回


def build_user_cohort_lookup(customers_path: Path | None = DEFAULT_CUSTOMERS) -> dict[str, dict[str, str]]:  # 冷启动查表
    age_buckets = _load_user_age_buckets(customers_path)  # age
    return {user_id: {"age_bucket": bucket} for user_id, bucket in age_buckets.items()}  # 每用户 cohort


def _select_ranked_for_user(  # 冷启动优先 cohort，否则 global
    index: PopularIndex,
    *,
    user_id: str | None,
    history: set[str],
    cohort_lookup: dict[str, dict[str, str]] | None,
) -> list[tuple[str, float]]:  # ranked list
    if history:  # 有历史走全局（与 category/item2item 互补）
        return index.global_ranked  # 全局
    if user_id and cohort_lookup:  # 冷启动
        cohorts = cohort_lookup.get(canonical_user_id(user_id), {})  # 该用户
        for dimension in COHORT_DIMENSIONS:  # 按优先级
            value = cohorts.get(dimension)  # 取值
            if not value or value == UNKNOWN_TOKEN:  # 无效
                continue  # 下一个
            ranked = index.cohort_ranked.get((dimension, value))  # 桶
            if ranked:  # 有数据
                return ranked  # 人群热门
    return index.global_ranked  # 回退全局


def recall_popular(  # 召回 Top-K 热门
    popular_index: PopularIndex | list[tuple[str, float]],  # 索引或兼容旧 list
    user_history: set[str] | None = None,  # 历史集合
    top_k: int = POPULAR_RECALL_TOP_K,  # K
    *,
    user_id: str | None = None,  # 冷启动查 cohort
    cohort_lookup: dict[str, dict[str, str]] | None = None,  # user -> cohort values
) -> list[tuple[str, float]]:  # 候选
    history = user_history or set()  # 默认空
    if isinstance(popular_index, list):  # 旧 list 索引
        ranked = popular_index  # 直接用
    else:  # PopularIndex
        ranked = _select_ranked_for_user(  # 选择列表
            popular_index,
            user_id=user_id,
            history=history,
            cohort_lookup=cohort_lookup,
        )
    return ranked[:top_k]  # 截断


if __name__ == "__main__":  # 脚本入口
    index = build_popular_index()  # 构建索引
    sample = recall_popular(index, user_history=set(), user_id=None, top_k=10)  # 示例
    print(f"Popular index size: {len(index.global_ranked):,}; cohort buckets: {len(index.cohort_ranked):,}")  # 规模
    print("Top-10 sample:", sample)  # 样例
