"""Tests for candidate coverage metrics."""  # 候选覆盖指标单测

from __future__ import annotations  # 延迟注解

import pytest  # 断言

from fashionrec.evaluation.coverage_metrics import (  # 覆盖指标
    candidate_count_summary,
    exclusive_hit_items,
    exclusive_hit_rate,
    jaccard_similarity,
    top_item_ids,
    union_top_items,
    user_coverage,
)


def test_jaccard_and_union_top_items() -> None:  # Jaccard 与并集
    left = [("0000000001", 1.0), ("0000000002", 0.5)]  # 通道 A
    right = [("0000000002", 0.8), ("0000000003", 0.1)]  # 通道 B
    assert jaccard_similarity(top_item_ids(left, 2), top_item_ids(right, 2)) == pytest.approx(1 / 3)  # 交集 1 / 并集 3
    union = union_top_items({"a": left, "b": right}, 2)  # 并集 Top-2
    assert union == ["0000000001", "0000000002"]  # item 1 分数 1.0 更高


def test_candidate_count_summary_percentiles() -> None:  # 候选数分位
    summary = candidate_count_summary([10, 20, 100, 200])  # 四个用户
    assert summary["mean"] == pytest.approx(82.5)  # 均值
    assert summary["p50"] == pytest.approx(60.0)  # 中位插值
    assert summary["p90"] >= summary["p50"]  # 单调


def test_user_coverage() -> None:  # 用户覆盖率
    assert user_coverage(3, 4) == pytest.approx(0.75)  # 3/4
    assert user_coverage(0, 0) == 0.0  # 空


def test_exclusive_hit_items() -> None:  # 独占命中
    actual = {"0000000001", "0000000002", "0000000003"}  # 标签
    channel_topk = {  # 各通道 Top
        "a": {"0000000001", "0000000009"},  # 只命中 1
        "b": {"0000000002", "0000000003"},  # 命中 2、3
    }
    exclusive = exclusive_hit_items(actual, channel_topk)  # 独占
    assert exclusive["a"] == {"0000000001"}  # 1 仅 A 命中
    assert exclusive["b"] == {"0000000002", "0000000003"}  # 2/3 仅 B 命中
    assert exclusive_hit_rate(actual, channel_topk) == pytest.approx(1.0)  # 全部命中都独占


def test_top_item_ids_canonicalizes() -> None:  # ID 规范化
    assert top_item_ids([("0000000001", 1.0)], 1) == ["0000000001"]  # 保前导零
