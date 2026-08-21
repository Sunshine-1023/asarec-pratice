"""Tests for the single MAP/Recall/NDCG/Hit implementation."""  # 统一指标测试

from __future__ import annotations  # 延迟注解

import pytest  # 近似比较

from fashionrec.shared.metrics.ranking import hit_at_k, map_at_k, ndcg_at_k, recall_at_k  # 统一指标
from fashionrec.industrial.ranking.fusion import normalize_item_id  # 融合侧 ID 规范化


def test_map_at_12_respects_rank_and_unique_targets() -> None:  # 计划中的核心样例
    actual = {"1", "2"}  # 两个真实商品
    pred = ["1", "9", "2"]  # 第 1、3 位命中
    assert map_at_k(actual, pred, 12) == pytest.approx((1.0 + 2 / 3) / 2)  # AP = (P@1 + P@3) / 2


def test_map_at_k_empty_target_is_zero() -> None:  # 空标签为 0
    assert map_at_k(set(), ["1"], 12) == 0.0  # 无真实购买


def test_map_ignores_duplicate_predictions() -> None:  # 重复预测只计第一次
    actual = {"1"}  # 只有一个真实商品
    pred = ["1", "1", "1"]  # 重复三次
    assert map_at_k(actual, pred, 12) == pytest.approx(1.0)  # 不应把重复命中再加一遍


def test_map_when_actual_larger_than_k() -> None:  # 标签数大于 K
    actual = {str(i) for i in range(20)}  # 20 个真实商品
    pred = [str(i) for i in range(12)]  # 前 12 全中
    expected = sum((i + 1) / (i + 1) for i in range(12)) / 12  # 每次命中 P@i=1，分母=min(20,12)=12
    assert map_at_k(actual, pred, 12) == pytest.approx(expected)  # 分母取 K


def test_map_when_prediction_shorter_than_k() -> None:  # 预测不足 K
    actual = {"1", "2"}  # 两个标签
    pred = ["1"]  # 只预测 1 个
    assert map_at_k(actual, pred, 12) == pytest.approx(1.0 / 2)  # 只在 rank1 命中一次


def test_map_no_hit_is_zero() -> None:  # 无命中
    assert map_at_k({"1"}, ["9", "8"], 12) == 0.0  # 全部未中


def test_metrics_are_consistent_with_leading_zeros() -> None:  # 前导零规范化前后一致
    actual_padded = {"0000000001", "0000000002"}  # 10 位商品 ID
    actual_plain = {"1", "2"}  # 去零后的 ID
    pred_padded = ["0000000001", "0000000009", "0000000002"]  # 带前导零的预测
    pred_plain = ["1", "9", "2"]  # 去零预测
    k = 12  # Top-12
    assert map_at_k(actual_padded, pred_padded, k) == pytest.approx(map_at_k(actual_plain, pred_plain, k))  # MAP 一致
    assert recall_at_k(actual_padded, pred_padded, k) == pytest.approx(recall_at_k(actual_plain, pred_plain, k))  # Recall 一致
    assert ndcg_at_k(actual_padded, pred_padded, k) == pytest.approx(ndcg_at_k(actual_plain, pred_plain, k))  # NDCG 一致
    assert hit_at_k(actual_padded, pred_padded, k) == hit_at_k(actual_plain, pred_plain, k)  # Hit 一致
    assert normalize_item_id("0000000001") == "0000000001"  # 与融合侧十位规范一致


def test_fusion_merges_padded_and_unpadded_same_item_once() -> None:  # 跨通道同商品只保留一次
    from fashionrec.industrial.ranking.fusion import fuse_candidates  # 局部导入保持测试依赖清晰

    fused = fuse_candidates(  # 两个通道使用不同 ID 表示
        user_id="u1",
        user_history=set(),
        channel_candidates={
            "popular": [("706016001", 10.0)],
            "sasrecf": [("0706016001", 0.9)],
        },
        channel_weights={"popular": 0.4, "sasrecf": 0.6},
        top_k=12,
    )
    assert fused == [("0706016001", pytest.approx(1.0))]  # 分数合并且候选唯一


def test_recall_hit_ndcg_basic() -> None:  # 其余指标基本行为
    actual = {"1", "2"}  # 标签
    pred = ["1", "9", "2"]  # 预测
    assert recall_at_k(actual, pred, 12) == pytest.approx(1.0)  # 两个都中
    assert hit_at_k(actual, pred, 12) == 1.0  # 有命中
    assert ndcg_at_k(actual, pred, 12) > 0.0  # NDCG 为正
    assert hit_at_k(actual, ["9"], 12) == 0.0  # 未命中
