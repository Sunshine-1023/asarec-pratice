"""Canonical ranking metrics for offline candidate evaluation."""  # 离线候选评估的唯一指标实现

from __future__ import annotations  # 启用延迟注解

import math  # log2
from collections.abc import Iterable, Sequence  # 类型

from fashionrec.shared.domain.ids import canonical_item_id  # 统一商品 ID 契约


def canonicalize_item_id(item_id: object) -> str:  # 商品 ID 规范化，去掉数字 ID 前导零
    return canonical_item_id(item_id)  # 保留指标层兼容函数名，委托领域契约


def canonicalize_item_set(items: Iterable[object]) -> set[str]:  # 将标签集合规范化
    return {canonicalize_item_id(item) for item in items}  # 逐个规范化后去重


def _unique_prefix(pred: Sequence[object], k: int) -> list[str]:  # 取前 K 个预测并去掉重复商品
    seen: set[str] = set()  # 已出现的规范化 ID
    unique: list[str] = []  # 保序去重结果
    for item in pred[:k]:  # 只看前 K 位
        canonical = canonicalize_item_id(item)  # 规范化
        if canonical in seen:  # 重复预测
            continue  # 忽略后续重复
        seen.add(canonical)  # 记录已出现
        unique.append(canonical)  # 保留第一次出现
    return unique  # 返回去重后的前缀


def recall_at_k(actual: Iterable[object], pred: Sequence[object], k: int) -> float:  # Recall@K
    actual_set = canonicalize_item_set(actual)  # 规范化真实标签
    if not actual_set:  # 无真实标签
        return 0.0  # 约定为 0
    pred_unique = _unique_prefix(pred, k)  # 前 K 去重预测
    return len(set(pred_unique) & actual_set) / len(actual_set)  # 命中数 / 标签数


def hit_at_k(actual: Iterable[object], pred: Sequence[object], k: int) -> float:  # Hit@K
    actual_set = canonicalize_item_set(actual)  # 规范化真实标签
    if not actual_set:  # 无真实标签
        return 0.0  # 约定为 0
    pred_unique = _unique_prefix(pred, k)  # 前 K 去重预测
    return 1.0 if set(pred_unique) & actual_set else 0.0  # 有命中为 1


def ndcg_at_k(actual: Iterable[object], pred: Sequence[object], k: int) -> float:  # NDCG@K
    actual_set = canonicalize_item_set(actual)  # 规范化真实标签
    if not actual_set:  # 无真实标签
        return 0.0  # 约定为 0
    pred_unique = _unique_prefix(pred, k)  # 前 K 去重预测
    dcg = 0.0  # 折损累积增益
    for i, item in enumerate(pred_unique):  # 按去重后位置计分
        if item in actual_set:  # 命中
            dcg += 1.0 / math.log2(i + 2)  # 位置从 1 开始，分母为 log2(rank+1)
    ideal_hits = min(len(actual_set), k)  # 理想命中数
    if ideal_hits == 0:  # 无理想命中
        return 0.0  # 返回 0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))  # 理想 DCG
    return dcg / idcg if idcg > 0 else 0.0  # NDCG


def map_at_k(actual: Iterable[object], pred: Sequence[object], k: int) -> float:  # MAP@K / AP@K
    """Average Precision at K: sum(P@i * rel_i) / min(|actual|, K)."""  # 与现有离线评估口径一致
    actual_set = canonicalize_item_set(actual)  # 规范化真实标签
    if not actual_set:  # 无真实标签
        return 0.0  # 约定为 0

    seen: set[str] = set()  # 已出现的预测商品，避免重复计分
    hits = 0  # 累计命中
    ap_sum = 0.0  # 精确率累加
    for i, item in enumerate(pred[:k], start=1):  # 遍历前 K 位，排名从 1 开始
        canonical = canonicalize_item_id(item)  # 规范化预测 ID
        if canonical in seen:  # 重复预测不再次计分
            continue  # 跳过
        seen.add(canonical)  # 记录已出现
        if canonical in actual_set:  # 命中真实标签
            hits += 1  # 命中数加一
            ap_sum += hits / i  # 累加当前位置精确率

    denom = min(len(actual_set), k)  # 分母取标签数与 K 的较小值
    return ap_sum / denom if denom > 0 else 0.0  # 返回 AP@K


def mean_metric(values: Sequence[float]) -> float:  # 对用户指标取平均
    return float(sum(values) / len(values)) if values else 0.0  # 空列表为 0
