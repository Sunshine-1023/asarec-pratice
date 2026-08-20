"""Tests for unified experiment YAML loading."""  # 统一实验配置测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pytest  # 测试框架
import yaml  # 写临时 YAML

from fashionrec.experiment.config import (  # 配置加载
    ExperimentConfigError,  # 配置错误
    classify_activity_tier,  # 分层
    load_experiment_config,  # 加载
)  # 导入结束
from fashionrec.ranking.fusion import classify_activity_tier as fusion_classify  # 现有融合分层，用于对齐


def _minimal_payload(**overrides) -> dict:  # 最小合法配置，便于覆盖单字段
    payload = {  # 必填骨架
        "experiment": {"name": "mini", "seed": 1},  # 元信息
        "data": {  # 数据
            "history_weeks": 4,  # 训练
            "valid_weeks": 1,  # 验证
            "test_weeks": 1,  # 测试
            "max_user_history": 100,  # 历史
            "min_user_purchases": 5,  # 最少购买
        },  # 数据结束
        "candidate": {"per_channel_top_k": 100, "union_top_k": 300, "final_top_k": 12},  # 候选
        "evaluation": {  # 评估
            "activity_tiers": {  # 分层
                "cold_start": [0, 0],  # 冷启动
                "low": [1, 2],  # 低
                "medium": [3, 9],  # 中
                "high": [10, None],  # 高
            }  # 分层结束
        },  # 评估结束
    }  # 骨架结束
    for key, value in overrides.items():  # 浅合并顶层
        if isinstance(value, dict) and isinstance(payload.get(key), dict):  # 嵌套节
            merged = dict(payload[key])  # 拷贝
            merged.update(value)  # 覆盖字段
            payload[key] = merged  # 写回
        else:  # 顶层替换
            payload[key] = value  # 替换
    return payload  # 返回


def _write_yaml(tmp_path: Path, payload: dict, name: str = "cfg.yaml") -> Path:  # 写临时 YAML
    path = tmp_path / name  # 路径
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")  # 写出
    return path  # 返回


def test_default_experiment_config_loads() -> None:  # 默认 YAML 能加载且类型正确
    config = load_experiment_config("configs/experiment.yaml")  # 加载仓库默认配置
    assert config.experiment.name == "fashionrec_v3"  # 实验名
    assert config.experiment.seed == 2026  # 种子为整数
    assert isinstance(config.experiment.seed, int)  # 类型
    assert config.data.history_weeks == 26  # 训练 26 周，降低测试周假冷启动
    assert config.data.valid_weeks == 1  # 验证 1 周
    assert config.data.test_weeks == 1  # 测试 1 周
    assert config.data.total_weeks == 28  # 总周数
    assert config.data.max_user_history == 200  # 历史上限
    assert config.data.min_user_purchases == 5  # 最少购买
    assert config.data.snapshot_frequency == "weekly"  # 快照频率
    assert config.data.keep_full_item_universe is True  # 默认不截断 Top-30k
    assert config.data.deduplicate_user_day_item is True  # 同日同 SKU 去重
    assert config.model_selection.checkpoint_shortlist_size == 5
    assert config.label.horizon_days == 7  # 未来 7 天
    assert config.label.target_mode == "next_basket"  # 购物篮标签
    assert config.label.include_repeat_label is True  # 复购分层
    assert config.label.include_new_to_user_label is True  # 新购分层
    assert config.candidate.per_channel_top_k == 200  # 每通道
    assert config.candidate.final_top_k == 12  # 最终 K
    assert config.candidate.union_top_k == 500  # 并集上限
    assert config.candidate.repurchase_top_k == 200  # 复购通道
    assert config.candidate.union_feature_version == "hm.candidate_union.v2"  # 并集证据
    assert config.ranking.enabled is False  # 阶段 0 关闭学习排序
    assert config.ranking.library == "lightgbm"  # 默认库
    assert config.ranking.objective == "lambdarank"  # 默认目标
    assert config.ranking.top_k_for_training == 500  # 排序训练候选
    assert config.ranking.train_snapshot_limit == 4
    assert config.evaluation.primary_metric == "MAP@12"  # 主指标
    assert set(config.evaluation.activity_tiers) == {"cold_start", "low", "medium", "high"}  # 四层都在
    assert config.evaluation.activity_tiers["high"] == (10, None)  # 高活跃无上界


def test_optional_fields_have_defaults(tmp_path: Path) -> None:  # 缺省字段走默认值
    path = _write_yaml(tmp_path, _minimal_payload())  # 最小配置，无 label/ranking/新 data 字段
    config = load_experiment_config(path)  # 加载
    assert config.data.backtest_windows == 3  # 回测窗口默认 3
    assert config.data.snapshot_frequency == "weekly"  # 快照默认 weekly
    assert config.data.keep_full_item_universe is True  # 全量 SKU 默认开
    assert config.data.deduplicate_user_day_item is True  # 去重默认开
    assert config.candidate.popular_top_k == 100  # 未写时回退 per_channel_top_k
    assert config.evaluation.primary_metric == "MAP@12"  # 主指标默认
    assert config.model_selection.checkpoint_shortlist_size == 5
    assert config.label.horizon_days == 7  # 标签窗口默认
    assert config.label.target_mode == "next_basket"  # 标签语义默认
    assert config.label.include_repeat_label is True  # 复购默认
    assert config.label.include_new_to_user_label is True  # 新购默认
    assert config.ranking.enabled is False  # 排序默认关闭
    assert config.ranking.library == "lightgbm"  # 库默认
    assert config.ranking.objective == "lambdarank"  # 目标默认
    assert config.ranking.top_k_for_training == 500  # 训练候选默认
    assert config.ranking.train_snapshot_limit == 4


def test_missing_required_field_raises(tmp_path: Path) -> None:  # 缺必填字段报错
    payload = _minimal_payload()  # 最小配置
    del payload["experiment"]["seed"]  # 去掉 seed
    path = _write_yaml(tmp_path, payload, "broken.yaml")  # 临时文件
    with pytest.raises(ExperimentConfigError, match="seed"):  # 必须提示缺 seed
        load_experiment_config(path)  # 加载应失败


def test_horizon_days_must_be_at_least_one(tmp_path: Path) -> None:  # 标签窗口非法
    payload = _minimal_payload(label={"horizon_days": 0})  # 0 天
    path = _write_yaml(tmp_path, payload)  # 写出
    with pytest.raises(ExperimentConfigError, match="horizon_days"):  # 报错
        load_experiment_config(path)  # 加载应失败


def test_backtest_windows_must_be_at_least_one(tmp_path: Path) -> None:  # 回测至少保留官方窗口
    payload = _minimal_payload(data={"backtest_windows": 0})  # 0 窗
    path = _write_yaml(tmp_path, payload)  # 写出
    with pytest.raises(ExperimentConfigError, match="backtest_windows"):  # 报错
        load_experiment_config(path)  # 加载应失败


def test_union_top_k_must_cover_final_top_k(tmp_path: Path) -> None:  # 并集不能小于最终截断
    payload = _minimal_payload()  # 最小
    payload["candidate"]["union_top_k"] = 8  # 小于 final 12
    path = _write_yaml(tmp_path, payload)  # 写出
    with pytest.raises(ExperimentConfigError, match="union_top_k"):  # 报错
        load_experiment_config(path)  # 加载应失败


@pytest.mark.parametrize(
    ("ranking", "label", "message"),
    [
        ({"enabled": True, "library": "catboost", "objective": "yetirank"}, {}, "lightgbm"),
        ({"enabled": True}, {"target_mode": "next_item"}, "next_basket"),
    ],
)
def test_enabled_industrial_protocol_rejects_unsupported_contracts(
    tmp_path: Path,
    ranking: dict,
    label: dict,
    message: str,
) -> None:
    path = _write_yaml(tmp_path, _minimal_payload(ranking=ranking, label=label))
    with pytest.raises(ExperimentConfigError, match=message):
        load_experiment_config(path)


def test_activity_tiers_match_current_fusion() -> None:  # 配置分层与现有融合函数一致
    config = load_experiment_config("configs/experiment.yaml")  # 加载默认配置
    for history_len in range(0, 16):  # 覆盖冷启动到高活跃
        from_config = classify_activity_tier(history_len, config.evaluation.activity_tiers)  # 配置分层
        from_fusion = fusion_classify(history_len)  # 现有融合分层
        assert from_config == from_fusion  # 两者必须一致，避免报告口径漂移
