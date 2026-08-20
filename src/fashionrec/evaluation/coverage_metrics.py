"""Coverage and complementarity metrics for multi-channel candidate sets."""  # 多通道候选覆盖与互补指标

from __future__ import annotations  # 延迟注解

import statistics  # 分位数
from collections.abc import Iterable, Mapping, Sequence  # 类型

from fashionrec.domain.ids import canonical_item_id  # ID 规范


def top_item_ids(candidates: Sequence[tuple[str, float]] | Sequence[str], k: int) -> list[str]:  # 前 K 商品 ID
    if k < 1:  # 非法 K
        raise ValueError("k must be >= 1")  # 报错
    if not candidates:  # 无候选
        return []  # 空列表
    if isinstance(candidates[0], tuple):  # (item, score)
        return [canonical_item_id(item_id) for item_id, _score in candidates[:k]]  # 截断
    return [canonical_item_id(item_id) for item_id in candidates[:k]]  # 纯 ID 列表


def top_item_set(candidates: Sequence[tuple[str, float]] | Sequence[str], k: int) -> set[str]:  # 前 K 集合
    return set(top_item_ids(candidates, k))  # 去重集合


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:  # |A∩B|/|A∪B|
    a = {canonical_item_id(item) for item in left}  # 左集
    b = {canonical_item_id(item) for item in right}  # 右集
    if not a and not b:  # 双空
        return 1.0  # 视为完全相同
    union = a | b  # 并集
    if not union:  # 兜底
        return 0.0  # 零
    return len(a & b) / len(union)  # Jaccard


def union_top_items(  # 多通道并集，按跨通道最大分截断到 K
    channel_candidates: Mapping[str, Sequence[tuple[str, float]]],
    k: int,
) -> list[str]:  # 并集 Top-K
    if k < 1:  # 非法
        raise ValueError("k must be >= 1")  # 报错
    best: dict[str, float] = {}  # item -> max score
    for candidates in channel_candidates.values():  # 各通道
        for item_id, score in candidates:  # 全部候选参与并集
            canonical = canonical_item_id(item_id)  # 规范
            best[canonical] = max(best.get(canonical, float("-inf")), float(score))  # 取最大分
    ordered = sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))  # 分降序、ID 稳定
    return [item_id for item_id, _score in ordered[:k]]  # 截断


def candidate_count_summary(counts: Sequence[int]) -> dict[str, float]:  # 候选数分布
    if not counts:  # 无用户
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}  # 全零
    ordered = sorted(int(value) for value in counts)  # 排序

    def _percentile(p: float) -> float:  # 线性插值分位
        if len(ordered) == 1:  # 单点
            return float(ordered[0])  # 直接返回
        rank = (len(ordered) - 1) * p  # 0-based 位置
        lo = int(rank)  # 下界
        hi = min(lo + 1, len(ordered) - 1)  # 上界
        weight = rank - lo  # 插值权重
        return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)  # 插值

    return {  # 汇总
        "mean": float(statistics.fmean(ordered)),  # 均值
        "p50": _percentile(0.50),  # 中位
        "p90": _percentile(0.90),  # 90 分位
        "p95": _percentile(0.95),  # 95 分位
    }  # 返回


def user_coverage(users_with_candidates: int, total_users: int) -> float:  # 有候选用户占比
    if total_users <= 0:  # 无用户
        return 0.0  # 零覆盖
    return float(users_with_candidates) / float(total_users)  # 比例


def exclusive_hit_items(  # 仅单通道命中的标签商品
    actual: Iterable[str],
    channel_topk: Mapping[str, set[str]],
) -> dict[str, set[str]]:  # channel -> 独占命中 SKU
    actual_set = {canonical_item_id(item) for item in actual}  # 标签
    hits = {channel: {canonical_item_id(x) for x in preds} & actual_set for channel, preds in channel_topk.items()}  # 各通道命中
    exclusive: dict[str, set[str]] = {}  # 独占
    for channel, hit_items in hits.items():  # 逐通道
        others = set().union(*(items for name, items in hits.items() if name != channel))  # 其它通道命中
        exclusive[channel] = hit_items - others  # 去掉被其它通道覆盖的
    return exclusive  # 返回


def exclusive_hit_rate(  # 全部命中里由单通道独占的比例
    actual: Iterable[str],
    channel_topk: Mapping[str, set[str]],
) -> float:  # 0..1
    actual_set = {canonical_item_id(item) for item in actual}  # 标签
    hits = set().union(*({canonical_item_id(x) for x in preds} & actual_set for preds in channel_topk.values()))  # 至少一通道命中
    if not hits:  # 无命中
        return 0.0  # 零
    exclusive = exclusive_hit_items(actual, channel_topk)  # 独占
    exclusive_all = set().union(*exclusive.values())  # 所有独占命中
    return len(exclusive_all) / len(hits)  # 独占占比
