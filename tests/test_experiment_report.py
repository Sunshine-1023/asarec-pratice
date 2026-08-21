"""Tests for experiment report writing and baseline protocol guards."""  # 实验报告与基线协议测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pytest  # 断言

from fashionrec.evaluation.baseline_command import (  # 基线命令中的协议函数
    assert_not_tuning_on_test,  # test 禁止搜权
    collect_baseline_variants,  # 变体收集
    load_frozen_fusion_weights,  # 只加载冻结权重
)  # 导入结束
from fashionrec.evaluation.experiment_report import (  # 报告
    compare_ranker_variants,  # 排序器 gate
    load_ranked_predictions,  # 打分 CSV
    save_experiment_outputs,  # 落盘
    save_ranker_comparison,  # 对照 JSON
    score_users,  # 计分
    skipped_variant,  # 跳过占位
)
from fashionrec.evaluation.offline_eval import (  # 假上下文与对照收集
    FusionEvalContext,
    collect_ranker_comparison_variants,
)
from fashionrec.experiment.config import REQUIRED_TIERS, load_experiment_config  # 配置
from fashionrec.ranking.fusion import ACTIVITY_WEIGHTS  # 默认 RRF 权重


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


def _ranker_variant(name: str, map12: float, per_tier_maps: dict[str, float]) -> dict:
    overall = {
        "users_evaluated": 4,
        "k": 12,
        "MAP@12": map12,
        "Recall@12": map12,
        "NDCG@12": map12,
        "Hit@12": 1.0,
        "variant": name,
    }
    per_tier = [
        {
            "activity_tier": tier,
            "n_users": 1,
            "users_evaluated": 1,
            "MAP@12": per_tier_maps[tier],
            "Recall@12": per_tier_maps[tier],
            "NDCG@12": per_tier_maps[tier],
            "Hit@12": 1.0,
            "k": 12,
        }
        for tier in REQUIRED_TIERS
    ]
    return {"name": name, "overall": overall, "per_tier": per_tier}


def test_compare_ranker_replaces_default_when_improved_without_tier_drop() -> None:
    even = {tier: 0.20 for tier in REQUIRED_TIERS}
    variants = [
        _ranker_variant("fusion_default_weights", 0.18, even),
        _ranker_variant("fusion_valid_search_weights", 0.20, even),
        _ranker_variant("lambdarank", 0.22, {**even, "high": 0.21}),
        skipped_variant("lambdarank_rerank", "stage 5"),
    ]
    report = compare_ranker_variants(variants)
    assert report["overall_improved"] is True
    assert report["major_tier_regression"] is False
    assert report["replace_default_ranker"] is True
    assert report["pipeline_default_unchanged"] is True
    assert report["deltas"]["MAP@12"] == pytest.approx(0.02)


def test_compare_ranker_rejects_major_activity_tier_map_drop() -> None:
    baseline_tiers = {tier: 0.20 for tier in REQUIRED_TIERS}
    candidate_tiers = {tier: 0.22 for tier in REQUIRED_TIERS}
    candidate_tiers["cold_start"] = 0.10  # 相对下降 50%
    variants = [
        _ranker_variant("fusion_valid_search_weights", 0.20, baseline_tiers),
        _ranker_variant("lambdarank", 0.24, candidate_tiers),
    ]
    report = compare_ranker_variants(variants)
    assert report["overall_improved"] is True
    assert report["major_tier_regression"] is True
    assert report["replace_default_ranker"] is False
    assert "cold_start" in report["reason"]


def test_compare_ranker_requires_primary_map_improvement() -> None:
    even = {tier: 0.20 for tier in REQUIRED_TIERS}
    baseline = _ranker_variant("fusion_valid_search_weights", 0.20, even)
    candidate = _ranker_variant("lambdarank", 0.19, even)
    candidate["overall"]["Recall@12"] = 0.30
    report = compare_ranker_variants([baseline, candidate])
    assert report["overall_improved"] is True
    assert report["primary_metric_improved"] is False
    assert report["replace_default_ranker"] is False
    assert "MAP@12" in report["reason"]


def test_compare_ranker_skips_lambdarank_without_replacing_default() -> None:
    even = {tier: 0.20 for tier in REQUIRED_TIERS}
    variants = [
        _ranker_variant("fusion_default_weights", 0.18, even),
        skipped_variant("fusion_valid_search_weights", "no frozen weights"),
        skipped_variant("lambdarank", "no ranker scored csv"),
        skipped_variant("lambdarank_rerank", "stage 5"),
    ]
    report = compare_ranker_variants(variants)
    assert report["replace_default_ranker"] is False
    assert "skipped" in report["reason"]
    assert report["effective_baseline"] == "fusion_default_weights"


def test_load_ranked_predictions_keeps_rank_order(tmp_path: Path) -> None:
    path = tmp_path / "scored.csv"
    path.write_text(
        "user_id,item_id,rank,score\n"
        "u,sku-b,2,0.4\n"
        "u,sku-a,1,0.9\n"
        "u,sku-c,3,0.1\n",
        encoding="utf-8",
    )
    preds = load_ranked_predictions(path, top_k=2)
    assert preds["u"] == ["sku-a", "sku-b"]


def test_collect_ranker_comparison_skips_missing_artifacts() -> None:
    config = load_experiment_config("configs/experiment.yaml")
    context = FusionEvalContext(
        targets={"u": {"hit"}},
        users=[
            {
                "user_id": "u",
                "actual_items": {"hit"},
                "history": ["9", "8", "7"],
                "history_set": {"9", "8", "7"},
                "channel_candidates": {
                    "popular": [("hit", 1.0)],
                    "category_popular": [("miss-a", 1.0)],
                    "item2item": [("miss-b", 1.0)],
                    "sasrecf": [("miss-c", 1.0)],
                },
            }
        ],
        sequence_channel="sasrecf",
        final_top_k=12,
    )
    variants = collect_ranker_comparison_variants(
        context,
        activity_tiers=config.evaluation.activity_tiers,
        searched_weights=None,
        ranker_predictions={"u": ["hit"]},
    )
    names = [item["name"] for item in variants]
    assert names == [
        "fusion_default_weights",
        "fusion_valid_search_weights",
        "lambdarank",
        "lambdarank_rerank",
    ]
    skipped = {item["name"]: item["overall"] for item in variants if item["overall"].get("skipped")}
    assert skipped["fusion_valid_search_weights"]["skipped"] is True
    assert skipped["lambdarank_rerank"]["skipped"] is True
    lambdarank = next(item for item in variants if item["name"] == "lambdarank")
    assert lambdarank["overall"]["MAP@12"] == pytest.approx(1.0)
    default = next(item for item in variants if item["name"] == "fusion_default_weights")
    assert default["overall"]["MAP@12"] < 1.0
    with_search = collect_ranker_comparison_variants(
        context,
        activity_tiers=config.evaluation.activity_tiers,
        searched_weights=ACTIVITY_WEIGHTS,
        ranker_predictions=None,
    )
    searched = next(item for item in with_search if item["name"] == "fusion_valid_search_weights")
    assert searched["overall"].get("skipped") is not True
    missing_ranker = next(item for item in with_search if item["name"] == "lambdarank")
    assert missing_ranker["overall"]["skipped"] is True


def test_save_ranker_comparison_writes_gate_json(tmp_path: Path) -> None:
    path = save_ranker_comparison(
        tmp_path / "ranker_comparison_valid.json",
        compare_ranker_variants(
            [
                skipped_variant("fusion_valid_search_weights", "no weights"),
                skipped_variant("lambdarank", "no csv"),
            ]
        ),
    )
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "replace_default_ranker" in text
    assert "pipeline_default_unchanged" in text
