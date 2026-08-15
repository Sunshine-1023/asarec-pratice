"""Tests for experiment report writing and baseline protocol guards."""  # 实验报告与基线协议测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pytest  # 断言

from run_baseline import (  # 基线脚本中的协议函数
    assert_not_tuning_on_test,  # test 禁止搜权
    collect_baseline_variants,  # 变体收集
    load_frozen_fusion_weights,  # 只加载冻结权重
)  # 导入结束
from src.evaluate.experiment_report import save_experiment_outputs, score_users  # 报告
from src.evaluate.offline_eval import FusionEvalContext  # 假上下文
from src.experiment.config import load_experiment_config  # 配置


def test_score_users_reports_all_four_tiers() -> None:  # 四个活跃度层级都要出现
    config = load_experiment_config("configs/experiment.yaml")  # 协议
    users = [  # 每层一个用户
        {"actual": {"1"}, "pred": ["1", "2"], "history_len": 0},  # 冷启动
        {"actual": {"1"}, "pred": ["9", "1"], "history_len": 2},  # 低
        {"actual": {"1"}, "pred": ["1"], "history_len": 5},  # 中
        {"actual": {"1"}, "pred": ["1"], "history_len": 12},  # 高
    ]  # 用户结束
    overall, per_tier = score_users(users, 12, config.evaluation.activity_tiers)  # 计分
    assert overall["users_evaluated"] == 4  # 四人
    assert overall["MAP@12"] > 0  # 有命中
    tiers = {row["activity_tier"]: row["n_users"] for row in per_tier}  # 分层人数
    assert tiers == {"cold_start": 1, "low": 1, "medium": 1, "high": 1}  # 四层都有


def test_save_experiment_outputs_writes_required_files(tmp_path: Path) -> None:  # 写出三份标准文件
    paths = save_experiment_outputs(  # 落盘
        tmp_path / "run",  # 目录
        manifest={"git_sha": "abc", "files": {}},  # 清单
        metrics={"MAP@12": 0.02, "protocol": "offline_candidate_ranking"},  # 指标
        per_tier_rows=[  # 分层
            {  # 高活跃
                "variant": "popular",  # 变体
                "activity_tier": "high",  # 层
                "n_users": 3,  # 人数
                "users_evaluated": 3,  # 同上
                "MAP@12": 0.01,  # MAP
                "Recall@12": 0.02,  # Recall
                "NDCG@12": 0.03,  # NDCG
                "Hit@12": 0.04,  # Hit
                "k": 12,  # K
            }  # 行结束
        ],  # 分层结束
    )  # 落盘结束
    assert paths["manifest"].exists()  # 清单
    assert paths["metrics"].exists()  # 指标
    assert paths["per_tier_metrics"].exists()  # 分层 CSV
    text = paths["per_tier_metrics"].read_text(encoding="utf-8")  # 读 CSV
    assert "high" in text and "popular" in text  # 含分层与变体


def test_test_split_cannot_search_weights() -> None:  # test 上搜权必须被拒绝
    assert_not_tuning_on_test("valid", searching=False)  # valid 不搜权可通过
    with pytest.raises(ValueError, match="test"):  # test 搜权
        assert_not_tuning_on_test("test", searching=True)  # 必须失败


def test_missing_weights_json_skips_searched_fusion_instead_of_searching() -> None:  # 无权重文件就跳过，不现搜
    assert load_frozen_fusion_weights("test", None) is None  # test 无 JSON 返回空
    assert load_frozen_fusion_weights("valid", None) is None  # valid 同样不在基线里搜权


def test_collect_baseline_variants_covers_required_names() -> None:  # 六组对照名称齐全
    config = load_experiment_config("configs/experiment.yaml")  # 协议
    context = FusionEvalContext(  # 极小上下文
        targets={"u": {"1"}},  # 标签
        users=[  # 一个用户
            {  # 用户
                "user_id": "u",  # ID
                "actual_items": {"1"},  # 标签
                "history": ["9", "8", "7"],  # 中活跃
                "history_set": {"9", "8", "7"},  # 集合
                "channel_candidates": {  # 四路候选
                    "popular": [("1", 1.0)],  # 热门命中
                    "category_popular": [("2", 1.0)],  # 类别
                    "item2item": [("3", 1.0)],  # 共现
                    "sasrecf": [("4", 1.0)],  # 序列
                },  # 候选结束
            }  # 用户结束
        ],  # 用户列表结束
        sequence_channel="sasrecf",  # 序列通道
        final_top_k=12,  # K
    )  # 上下文结束
    variants = collect_baseline_variants(  # 收集对照
        context,  # 上下文
        config,  # 协议
        searched_weights=None,  # 不提供搜权结果
        sequence_csv_exists=True,  # 当作已有召回文件
    )  # 收集结束
    names = [item["name"] for item in variants]  # 变体名
    assert names[:4] == ["popular", "category_popular", "item2item", "sasrecf"]  # 四路单通道
    assert "fusion_default_weights" in names  # 默认融合
    assert "fusion_valid_search_weights" in names  # 搜权融合占位
    skipped = next(item for item in variants if item["name"] == "fusion_valid_search_weights")  # 搜权变体
    assert skipped["overall"]["skipped"] is True  # 无 JSON 时跳过而不是搜索
    popular = next(item for item in variants if item["name"] == "popular")  # 热门
    assert popular["overall"]["MAP@12"] == pytest.approx(1.0)  # 热门命中真实商品
