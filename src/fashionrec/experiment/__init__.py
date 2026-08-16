"""Unified experiment protocol loading."""  # 统一实验协议加载

from fashionrec.experiment.config import (  # 导出配置类型与加载函数
    CandidateConfig,  # 候选召回配置
    DataConfig,  # 数据切分配置
    EvaluationConfig,  # 评估配置
    ExperimentConfig,  # 完整实验配置
    ExperimentMeta,  # 实验元信息
    classify_activity_tier,  # 按配置划分活跃度
    load_experiment_config,  # 加载 YAML 配置
)  # 导出结束

__all__ = [  # 公开接口
    "CandidateConfig",  # 候选召回配置
    "DataConfig",  # 数据切分配置
    "EvaluationConfig",  # 评估配置
    "ExperimentConfig",  # 完整实验配置
    "ExperimentMeta",  # 实验元信息
    "classify_activity_tier",  # 活跃度划分
    "load_experiment_config",  # 配置加载
]  # 公开接口结束
