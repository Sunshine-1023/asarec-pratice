"""Tests for candidate diagnostics report builder."""  # 候选诊断报告

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pytest  # 断言

from fashionrec.industrial.evaluation.candidate_diagnostics import (  # 诊断
    DIAGNOSTICS_SCHEMA_VERSION,
    assert_candidate_diagnostics_present,
    classify_purchase_stratum,
    diagnose_users,
)
from fashionrec.industrial.evaluation.experiment_report import save_candidate_diagnostics  # 落盘
from fashionrec.shared.experiment.config import load_experiment_config  # 协议


def _users() -> list[dict]:  # 两用户、两通道
    return [
        {
            "user_id": "warm-repeat",
            "actual_items": {"1"},
            "history": ["1", "9"],
            "history_set": {"1", "9"},
            "history_len": 2,
            "channel_candidates": {
                "popular": [("1", 1.0), ("8", 0.5)],
                "item2item": [("2", 1.0), ("1", 0.2)],
            },
        },
        {
            "user_id": "cold-new",
            "actual_items": {"3"},
            "history": [],
            "history_set": set(),
            "history_len": 0,
            "channel_candidates": {
                "popular": [("3", 1.0), ("4", 0.5)],
                "item2item": [("5", 1.0)],
            },
        },
    ]


def test_classify_purchase_stratum() -> None:  # 复购结构
    assert classify_purchase_stratum({"1"}, {"1", "2"}) == "repeat_only"  # 纯复购
    assert classify_purchase_stratum({"3"}, {"1"}) == "new_only"  # 纯新购
    assert classify_purchase_stratum({"1", "3"}, {"1"}) == "mixed"  # 混合


def test_diagnose_users_reports_channel_union_and_strata() -> None:  # 通道/并集/分层
    config = load_experiment_config("configs/industrial/experiment.yaml")
    report = diagnose_users(_users(), channels=["popular", "item2item"], activity_tiers=config.evaluation.activity_tiers)  # 诊断
    assert report["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION  # 版本
    assert report["coverage"]["users_evaluated"] == 2  # 两用户
    assert report["coverage"]["user_coverage"] == 1.0  # 都有候选
    overall_pop = next(  # popular overall Recall@50
        row for row in report["per_channel_metrics"] if row["channel"] == "popular" and row["scope"] == "overall" and row["k"] == 50
    )
    assert overall_pop["Recall@50"] == pytest.approx(1.0)  # 两用户都命中
    union_overall = next(row for row in report["union_metrics"] if row["scope"] == "overall" and row["k"] == 100)  # 并集
    assert union_overall["Hit@100"] == pytest.approx(1.0)  # 并集全命中
    cold_rows = [row for row in report["union_metrics"] if row["scope"] == "warm_cold:cold"]  # 冷启动层
    assert cold_rows and cold_rows[0]["n_users"] == 1  # 一个冷用户
    assert len(report["channel_pair_jaccard"]) == 1  # 一对通道


def test_save_candidate_diagnostics_writes_json_and_csv(tmp_path: Path) -> None:  # 落盘
    config = load_experiment_config("configs/industrial/experiment.yaml")
    payload = diagnose_users(_users(), channels=["popular", "item2item"], activity_tiers=config.evaluation.activity_tiers)  # 报告
    paths = save_candidate_diagnostics(tmp_path / "run", payload)  # 写出
    assert paths["candidate_diagnostics"].exists()  # JSON
    assert paths["candidate_channel_metrics"].exists()  # 单通道 CSV
    assert paths["candidate_union_metrics"].exists()  # 并集 CSV
    assert paths["candidate_pair_jaccard"].exists()  # Jaccard CSV
    assert paths["candidate_exclusive_hits"].exists()  # 独占 CSV


def test_assert_candidate_diagnostics_present(tmp_path: Path) -> None:  # 缺失报告应失败
    with pytest.raises(FileNotFoundError, match="candidate coverage report"):
        assert_candidate_diagnostics_present(tmp_path)  # 无文件
    config = load_experiment_config("configs/industrial/experiment.yaml")
    payload = diagnose_users(_users(), channels=["popular", "item2item"], activity_tiers=config.evaluation.activity_tiers)  # 报告
    save_candidate_diagnostics(tmp_path, payload)  # 写出
    assert assert_candidate_diagnostics_present(tmp_path).name == "candidate_diagnostics.json"  # 通过
