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


def test_default_experiment_config_loads() -> None:  # 默认 YAML 能加载且类型正确
    config = load_experiment_config("configs/experiment.yaml")  # 加载仓库默认配置
    assert config.experiment.name == "fashionrec_v3"  # 实验名
    assert config.experiment.seed == 2026  # 种子为整数
    assert isinstance(config.experiment.seed, int)  # 类型
    assert config.data.history_weeks == 4  # 训练 4 周
    assert config.data.valid_weeks == 1  # 验证 1 周
    assert config.data.test_weeks == 1  # 测试 1 周
    assert config.data.total_weeks == 6  # 总周数
    assert config.data.max_user_history == 100  # 历史上限
    assert config.data.min_user_purchases == 5  # 最少购买
    assert config.model_selection.checkpoint_shortlist_size == 5
    assert config.candidate.final_top_k == 12  # 最终 K
    assert config.candidate.union_top_k == 300  # 并集上限
    assert config.evaluation.primary_metric == "MAP@12"  # 主指标
    assert set(config.evaluation.activity_tiers) == {"cold_start", "low", "medium", "high"}  # 四层都在
    assert config.evaluation.activity_tiers["high"] == (10, None)  # 高活跃无上界


def test_optional_fields_have_defaults(tmp_path: Path) -> None:  # 缺省字段走默认值
    payload = {  # 最小必填配置
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
    }  # 配置结束
    path = tmp_path / "mini.yaml"  # 临时文件
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")  # 写出
    config = load_experiment_config(path)  # 加载
    assert config.data.backtest_windows == 3  # 回测窗口默认 3
    assert config.candidate.popular_top_k == 100  # 未写时回退 per_channel_top_k
    assert config.evaluation.primary_metric == "MAP@12"  # 主指标默认
    assert config.model_selection.checkpoint_shortlist_size == 5


def test_missing_required_field_raises(tmp_path: Path) -> None:  # 缺必填字段报错
    payload = {  # 缺 seed
        "experiment": {"name": "broken"},  # 只有名字
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
    }  # 配置结束
    path = tmp_path / "broken.yaml"  # 临时文件
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")  # 写出
    with pytest.raises(ExperimentConfigError, match="seed"):  # 必须提示缺 seed
        load_experiment_config(path)  # 加载应失败


def test_activity_tiers_match_current_fusion() -> None:  # 配置分层与现有融合函数一致
    config = load_experiment_config("configs/experiment.yaml")  # 加载默认配置
    for history_len in range(0, 16):  # 覆盖冷启动到高活跃
        from_config = classify_activity_tier(history_len, config.evaluation.activity_tiers)  # 配置分层
        from_fusion = fusion_classify(history_len)  # 现有融合分层
        assert from_config == from_fusion  # 两者必须一致，避免报告口径漂移
