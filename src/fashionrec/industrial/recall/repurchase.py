"""Repurchase recall: user-SKU frequency with recency decay."""  # 复购召回

from __future__ import annotations  # 延迟注解

import math  # 衰减
from dataclasses import dataclass, field  # 索引
from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # ID
from fashionrec.industrial.recall.window_scores import filter_interactions_as_of  # as-of


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集
REPURCHASE_RECALL_TOP_K = 50  # 召回 Top-K
REPURCHASE_HALF_LIFE_DAYS = 28.0  # 复购近因衰减半衰期
REPURCHASE_SCHEMA_VERSION = "hm.repurchase.v1"  # 索引语义


@dataclass(frozen=True)  # 用户复购统计
class RepurchaseIndex:
    user_stats: dict[str, dict[str, tuple[int, pd.Timestamp]]] = field(default_factory=dict)  # user -> item -> (count, last_date)


def _load_interactions(*inter_paths: str | Path) -> pd.DataFrame:  # 读交互
    if not inter_paths:  # 默认
        inter_paths = (DEFAULT_INTER_PATH,)  # 训练集
    frames: list[pd.DataFrame] = []  # 收集
    for path in inter_paths:  # 每个文件
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=["user_id:token", "item_id:token", "timestamp:float"],
            dtype={"user_id:token": "string", "item_id:token": "string"},
        )
        df["user_id:token"] = df["user_id:token"].map(canonical_user_id)  # 用户
        df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 商品
        df["date"] = pd.to_datetime(df["timestamp:float"], unit="s").dt.normalize()  # 自然日
        frames.append(df[["user_id:token", "item_id:token", "date"]])  # 保留列
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()  # 合并


def _recency_weight(last_date: pd.Timestamp, *, as_of: pd.Timestamp, half_life_days: float) -> float:  # 近因衰减
    age_days = max(0.0, float((as_of - pd.Timestamp(last_date).normalize()).days))  # 距 as_of
    if half_life_days <= 0.0:  # 无衰减
        return 1.0  # 常数
    return math.exp(-math.log(2.0) * age_days / half_life_days)  # 半衰期


def build_repurchase_index(  # 构建复购索引
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_INTER_PATH,
    *,
    as_of: pd.Timestamp | str | None = None,
) -> RepurchaseIndex:
    if isinstance(inter_paths, (str, Path)):  # 单路径
        paths = [inter_paths]  # 包装
    else:  # 多路径
        paths = list(inter_paths)  # 列表
    frame = _load_interactions(*paths)  # 读交互
    if frame.empty:  # 空
        return RepurchaseIndex()  # 空索引
    frame = filter_interactions_as_of(frame, as_of)  # as-of 截断
    if frame.empty:  # 截断后空
        return RepurchaseIndex()  # 空索引

    user_stats: dict[str, dict[str, tuple[int, pd.Timestamp]]] = {}  # 聚合
    grouped = frame.groupby(["user_id:token", "item_id:token"], sort=False)  # 用户-SKU
    for (user_id, item_id), group in grouped:  # 每组
        count = int(len(group))  # 购买次数
        last_date = pd.Timestamp(group["date"].max()).normalize()  # 最近购买日
        user_stats.setdefault(str(user_id), {})[str(item_id)] = (count, last_date)  # 写入
    return RepurchaseIndex(user_stats=user_stats)  # 返回


def recall_repurchase(  # 复购召回
    user_id: str,
    user_history: list[str] | set[str],
    index: RepurchaseIndex,
    *,
    top_k: int = REPURCHASE_RECALL_TOP_K,
    half_life_days: float = REPURCHASE_HALF_LIFE_DAYS,
    as_of: pd.Timestamp | str | None = None,
) -> list[tuple[str, float]]:
    user_key = canonical_user_id(user_id)  # 规范用户
    stats = index.user_stats.get(user_key, {})  # 用户统计
    if not stats:  # 冷启动
        return []  # 空，由 popular/content fallback
    anchor = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp("today").normalize()  # 锚点
    _ = user_history  # 完整购买历史正是复购信号，不能把它误当成当前购物篮排除
    scored: list[tuple[str, float]] = []  # 候选
    for item_id, (count, last_date) in stats.items():  # 每个买过 SKU
        score = float(count) * _recency_weight(last_date, as_of=anchor, half_life_days=half_life_days)  # 次数×衰减
        scored.append((item_id, score))  # 记录
    scored.sort(key=lambda pair: (-pair[1], pair[0]))  # 降序
    return scored[:top_k]  # Top-K
