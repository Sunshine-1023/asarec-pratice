"""Shared multi-window rank-normalized popularity scoring for recall channels."""  # 多窗口 rank 归一化热度

from __future__ import annotations  # 延迟注解

from collections import Counter  # 计数
from collections.abc import Iterable  # 类型

import pandas as pd  # 日期

from fashionrec.industrial.data.time import week_window_start  # 窗口起


DEFAULT_WINDOW_WEEKS = (1, 2, 4, 12)  # 计划窗口
DEFAULT_WINDOW_WEIGHTS = (0.45, 0.30, 0.15, 0.10)  # 短窗更重，总和 1.0


def filter_interactions_as_of(frame: pd.DataFrame, as_of: pd.Timestamp | str | None) -> pd.DataFrame:  # 只保留 as_of 及以前
    if as_of is None:  # 未指定
        return frame.copy()  # 原样
    if frame.empty:  # 空表
        return frame.copy()  # 直接返回
    cutoff = pd.Timestamp(as_of).normalize()  # 自然日
    if "date" not in frame.columns:  # 无日期列
        raise ValueError("interactions must contain date column for as_of filtering")  # 报错
    return frame[frame["date"] <= cutoff].copy()  # 截断


def rank_normalize_counts(counts: Counter[str]) -> dict[str, float]:  # 窗口内 rank -> [0,1]
    if not counts:  # 空窗
        return {}  # 无分
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))  # 热度降序
    total = float(len(ranked))  # 商品数
    return {item_id: (total - rank + 1.0) / total for rank, (item_id, _count) in enumerate(ranked, start=1)}  # rank 归一化


def blend_window_scores(  # 多窗口 rank 分加权融合
    window_counts: Iterable[Counter[str]],  # 各窗口原始计数
    window_weights: Iterable[float],  # 权重
) -> dict[str, float]:  # item -> score
    weights = tuple(window_weights)  # 固定顺序
    counts_list = list(window_counts)  # 物化
    if len(counts_list) != len(weights):  # 长度不一致
        raise ValueError("window_counts and window_weights must have the same length")  # 报错
    blended: dict[str, float] = {}  # 融合分
    for weight, counts in zip(weights, counts_list):  # 逐窗
        if weight <= 0.0:  # 跳过零权重
            continue  # 下一窗
        normalized = rank_normalize_counts(counts)  # rank 归一化
        for item_id, score in normalized.items():  # 累加
            blended[item_id] = blended.get(item_id, 0.0) + weight * score  # 加权
    return blended  # 返回


def window_count_series(  # 单窗口 item 计数
    frame: pd.DataFrame,  # 需含 date 与 item_col
    *,
    max_date: pd.Timestamp,  # 锚点日
    weeks: int,  # 窗口周数
    item_col: str,  # 商品列
) -> Counter[str]:  # 计数
    start = week_window_start(max_date, weeks)  # 窗口起
    window = frame[frame["date"] >= start]  # 窗口内
    if window.empty:  # 无交互
        return Counter()  # 空
    return Counter(window[item_col].tolist())  # 计数


def build_window_popularity(  # 从交互表构建融合热度
    frame: pd.DataFrame,  # 需含 date 与 item_col
    *,
    item_col: str,  # 商品列名
    window_weeks: tuple[int, ...] = DEFAULT_WINDOW_WEEKS,  # 窗口
    window_weights: tuple[float, ...] = DEFAULT_WINDOW_WEIGHTS,  # 权重
    as_of: pd.Timestamp | str | None = None,  # 预测日
) -> dict[str, float]:  # item -> score
    if len(window_weeks) != len(window_weights):  # 校验
        raise ValueError("window_weeks and window_weights must have the same length")  # 报错
    filtered = filter_interactions_as_of(frame, as_of)  # as-of 截断
    if filtered.empty:  # 无历史
        return {}  # 空索引
    max_date = filtered["date"].max()  # 锚点
    counts = [window_count_series(filtered, max_date=max_date, weeks=weeks, item_col=item_col) for weeks in window_weeks]  # 各窗
    return blend_window_scores(counts, window_weights)  # 融合


def rank_score_items(scores: dict[str, float]) -> list[tuple[str, float]]:  # 排序输出
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))  # 分降序、ID 稳定
