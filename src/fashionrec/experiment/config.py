"""Load and validate the unified experiment YAML protocol."""  # 加载并校验统一实验 YAML 协议

from __future__ import annotations  # 启用延迟注解

from dataclasses import dataclass  # 数据类
from pathlib import Path  # 路径类型
from typing import Any, Mapping  # 类型注解

import yaml  # YAML 解析


DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment.yaml")  # 默认实验配置路径
REQUIRED_TIERS = ("cold_start", "low", "medium", "high")  # 必须存在的四个活跃度分层
PRIMARY_METRIC_DEFAULT = "MAP@12"  # 默认主指标


class ExperimentConfigError(ValueError):  # 实验配置错误
    """Raised when the experiment YAML is missing fields or has invalid types."""  # YAML 缺字段或类型错误时抛出


@dataclass(frozen=True)  # 不可变数据类
class ExperimentMeta:  # 实验元信息
    name: str  # 实验名称
    seed: int  # 随机种子


@dataclass(frozen=True)  # 不可变数据类
class DataConfig:  # 数据协议
    history_weeks: int  # 训练历史周数
    valid_weeks: int  # 验证周数
    test_weeks: int  # 测试周数
    backtest_windows: int  # 回测窗口数
    max_user_history: int  # 每用户最大历史条数
    min_user_purchases: int  # 用户最少购买次数
    snapshot_frequency: str  # 快照频率，当前仅 weekly
    keep_full_item_universe: bool  # 是否保留全量 SKU，不默认截断 Top-30k
    deduplicate_user_day_item: bool  # 是否将同日同 SKU 聚合成一个事件

    @property
    def total_weeks(self) -> int:  # 训练+验证+测试总周数
        return self.history_weeks + self.valid_weeks + self.test_weeks  # 三窗口求和


@dataclass(frozen=True)  # 不可变数据类
class CandidateConfig:  # 候选召回协议
    per_channel_top_k: int  # 每通道默认 Top-K
    popular_top_k: int  # 热门通道 Top-K
    category_popular_top_k: int  # 类别热门通道 Top-K
    item2item_top_k: int  # item2item 通道 Top-K
    sequence_top_k: int  # 序列模型通道 Top-K
    union_top_k: int  # 多通道并集上限
    final_top_k: int  # 最终推荐条数


@dataclass(frozen=True)
class ModelSelectionConfig:
    checkpoint_shortlist_size: int


@dataclass(frozen=True)  # 不可变数据类
class LabelConfig:  # 标签协议
    horizon_days: int  # 标签窗口天数
    target_mode: str  # next_basket 或 next_item
    include_repeat_label: bool  # 是否拆出复购标签
    include_new_to_user_label: bool  # 是否拆出用户首次购买标签


@dataclass(frozen=True)  # 不可变数据类
class RankingConfig:  # 学习排序协议；阶段 0 默认关闭
    enabled: bool  # 是否启用学习排序
    library: str  # lightgbm 或 catboost
    objective: str  # 排序目标，默认 lambdarank
    top_k_for_training: int  # 排序训练使用的候选上限


@dataclass(frozen=True)  # 不可变数据类
class EvaluationConfig:  # 评估协议
    primary_metric: str  # 主指标名称
    protocol: str  # 评估口径标识
    activity_tiers: dict[str, tuple[int, int | None]]  # 分层闭区间 [lo, hi]，hi=None 表示无上界


@dataclass(frozen=True)  # 不可变数据类
class ExperimentConfig:  # 完整实验配置
    experiment: ExperimentMeta  # 实验元信息
    data: DataConfig  # 数据协议
    model_selection: ModelSelectionConfig
    candidate: CandidateConfig  # 候选协议
    label: LabelConfig  # 标签协议
    ranking: RankingConfig  # 排序协议
    evaluation: EvaluationConfig  # 评估协议
    source_path: Path  # 配置文件路径


def _require_mapping(payload: Any, where: str) -> dict[str, Any]:  # 要求节点为字典
    if not isinstance(payload, dict):  # 类型不符
        raise ExperimentConfigError(f"{where} must be a mapping")  # 抛出配置错误
    return payload  # 返回字典


def _require_int(payload: Mapping[str, Any], key: str, where: str, default: int | None = None) -> int:  # 读取整数
    if key not in payload:  # 缺字段
        if default is None:  # 无默认值
            raise ExperimentConfigError(f"Missing required field: {where}.{key}")  # 抛出缺字段
        return default  # 使用默认值
    value = payload[key]  # 取出原值
    if isinstance(value, bool) or not isinstance(value, int):  # bool 是 int 子类，需排除
        raise ExperimentConfigError(f"{where}.{key} must be an integer, got {type(value).__name__}")  # 类型错误
    return value  # 返回整数


def _require_str(payload: Mapping[str, Any], key: str, where: str, default: str | None = None) -> str:  # 读取字符串
    if key not in payload:  # 缺字段
        if default is None:  # 无默认值
            raise ExperimentConfigError(f"Missing required field: {where}.{key}")  # 抛出缺字段
        return default  # 使用默认值
    value = payload[key]  # 取出原值
    if not isinstance(value, str) or not value.strip():  # 必须是非空字符串
        raise ExperimentConfigError(f"{where}.{key} must be a non-empty string")  # 类型错误
    return value  # 返回字符串


def _require_bool(payload: Mapping[str, Any], key: str, where: str, default: bool | None = None) -> bool:  # 读取布尔
    if key not in payload:  # 缺字段
        if default is None:  # 无默认值
            raise ExperimentConfigError(f"Missing required field: {where}.{key}")  # 抛出缺字段
        return default  # 使用默认值
    value = payload[key]  # 取出原值
    if not isinstance(value, bool):  # YAML 必须是 true/false
        raise ExperimentConfigError(f"{where}.{key} must be a boolean, got {type(value).__name__}")  # 类型错误
    return value  # 返回布尔


def _require_choice(payload: Mapping[str, Any], key: str, where: str, allowed: tuple[str, ...], default: str) -> str:  # 枚举字符串
    value = _require_str(payload, key, where, default=default)  # 先按字符串读取
    if value not in allowed:  # 不在允许集合
        raise ExperimentConfigError(f"{where}.{key} must be one of {allowed}, got {value!r}")  # 枚举错误
    return value  # 返回合法取值


def _parse_tier_bounds(raw: Any, tier: str) -> tuple[int, int | None]:  # 解析 [lo, hi]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:  # 必须是二元列表
        raise ExperimentConfigError(f"evaluation.activity_tiers.{tier} must be [low, high]")  # 格式错误
    lo, hi = raw[0], raw[1]  # 拆出上下界
    if not isinstance(lo, int) or isinstance(lo, bool):  # 下界必须是整数
        raise ExperimentConfigError(f"evaluation.activity_tiers.{tier}[0] must be an integer")  # 类型错误
    if hi is not None and (not isinstance(hi, int) or isinstance(hi, bool)):  # 上界为整数或 null
        raise ExperimentConfigError(f"evaluation.activity_tiers.{tier}[1] must be an integer or null")  # 类型错误
    if hi is not None and hi < lo:  # 上界不能小于下界
        raise ExperimentConfigError(f"evaluation.activity_tiers.{tier} has high < low")  # 区间非法
    return lo, hi  # 返回闭区间


def _parse_activity_tiers(raw: Any) -> dict[str, tuple[int, int | None]]:  # 解析四个活跃度分层
    mapping = _require_mapping(raw, "evaluation.activity_tiers")  # 要求为字典
    missing = [tier for tier in REQUIRED_TIERS if tier not in mapping]  # 检查缺层
    if missing:  # 有缺失分层
        raise ExperimentConfigError(f"evaluation.activity_tiers missing: {missing}")  # 抛出缺层
    return {tier: _parse_tier_bounds(mapping[tier], tier) for tier in REQUIRED_TIERS}  # 按固定顺序解析


def classify_activity_tier(  # 按实验配置划分用户活跃度
    history_len: int,  # 用户历史购买次数
    activity_tiers: Mapping[str, tuple[int, int | None]],  # 分层闭区间
) -> str:  # 返回分层名称
    if history_len < 0:  # 历史长度非法
        raise ValueError(f"history_len must be >= 0, got {history_len}")  # 抛出错误
    for tier in REQUIRED_TIERS:  # 按冷启动到高活跃顺序匹配
        lo, hi = activity_tiers[tier]  # 取出区间
        if history_len < lo:  # 未到该层下界
            continue  # 看下一层
        if hi is None or history_len <= hi:  # 无上界或落在闭区间内
            return tier  # 命中该层
    raise ValueError(f"history_len={history_len} does not match any activity tier")  # 未匹配任何层


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:  # 加载并校验实验配置
    config_path = Path(path) if path is not None else DEFAULT_EXPERIMENT_CONFIG  # 解析配置路径
    if not config_path.exists():  # 文件不存在
        raise FileNotFoundError(f"Experiment config not found: {config_path}")  # 抛出文件错误

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))  # 读取 YAML
    root = _require_mapping(payload, "root")  # 根节点必须是字典

    experiment_raw = _require_mapping(root.get("experiment"), "experiment")  # 实验元信息节点
    data_raw = _require_mapping(root.get("data"), "data")  # 数据协议节点
    model_selection_raw = _require_mapping(root.get("model_selection", {}), "model_selection")
    candidate_raw = _require_mapping(root.get("candidate"), "candidate")  # 候选协议节点
    label_raw = _require_mapping(root.get("label", {}), "label")  # 标签协议，旧配置可缺省
    ranking_raw = _require_mapping(root.get("ranking", {}), "ranking")  # 排序协议，旧配置可缺省
    evaluation_raw = _require_mapping(root.get("evaluation"), "evaluation")  # 评估协议节点

    per_channel_top_k = _require_int(candidate_raw, "per_channel_top_k", "candidate")  # 每通道默认 Top-K
    experiment = ExperimentMeta(  # 组装实验元信息
        name=_require_str(experiment_raw, "name", "experiment"),  # 实验名
        seed=_require_int(experiment_raw, "seed", "experiment"),  # 随机种子
    )  # 实验元信息结束
    data = DataConfig(  # 组装数据协议
        history_weeks=_require_int(data_raw, "history_weeks", "data"),  # 训练周数
        valid_weeks=_require_int(data_raw, "valid_weeks", "data"),  # 验证周数
        test_weeks=_require_int(data_raw, "test_weeks", "data"),  # 测试周数
        backtest_windows=_require_int(data_raw, "backtest_windows", "data", default=3),  # 回测窗口数
        max_user_history=_require_int(data_raw, "max_user_history", "data"),  # 历史长度上限
        min_user_purchases=_require_int(data_raw, "min_user_purchases", "data"),  # 最少购买次数
        snapshot_frequency=_require_choice(  # 快照频率
            data_raw, "snapshot_frequency", "data", ("weekly",), default="weekly"
        ),  # 快照频率结束
        keep_full_item_universe=_require_bool(data_raw, "keep_full_item_universe", "data", default=True),  # 全量商品
        deduplicate_user_day_item=_require_bool(data_raw, "deduplicate_user_day_item", "data", default=True),  # 同日同 SKU 去重
    )  # 数据协议结束
    model_selection = ModelSelectionConfig(
        checkpoint_shortlist_size=_require_int(
            model_selection_raw,
            "checkpoint_shortlist_size",
            "model_selection",
            default=5,
        )
    )
    candidate = CandidateConfig(  # 组装候选协议
        per_channel_top_k=per_channel_top_k,  # 每通道默认 Top-K
        popular_top_k=_require_int(candidate_raw, "popular_top_k", "candidate", default=per_channel_top_k),  # 热门 Top-K
        category_popular_top_k=_require_int(  # 类别热门 Top-K
            candidate_raw, "category_popular_top_k", "candidate", default=per_channel_top_k
        ),  # 类别热门 Top-K 结束
        item2item_top_k=_require_int(candidate_raw, "item2item_top_k", "candidate", default=per_channel_top_k),  # item2item Top-K
        sequence_top_k=_require_int(candidate_raw, "sequence_top_k", "candidate", default=per_channel_top_k),  # 序列通道 Top-K
        union_top_k=_require_int(candidate_raw, "union_top_k", "candidate"),  # 并集上限
        final_top_k=_require_int(candidate_raw, "final_top_k", "candidate"),  # 最终 Top-K
    )  # 候选协议结束
    label = LabelConfig(  # 组装标签协议
        horizon_days=_require_int(label_raw, "horizon_days", "label", default=7),  # 未来窗口
        target_mode=_require_choice(  # 标签语义
            label_raw, "target_mode", "label", ("next_basket", "next_item"), default="next_basket"
        ),  # 标签语义结束
        include_repeat_label=_require_bool(label_raw, "include_repeat_label", "label", default=True),  # 复购标签
        include_new_to_user_label=_require_bool(label_raw, "include_new_to_user_label", "label", default=True),  # 新购标签
    )  # 标签协议结束
    ranking = RankingConfig(  # 组装排序协议
        enabled=_require_bool(ranking_raw, "enabled", "ranking", default=False),  # 默认关闭学习排序
        library=_require_choice(ranking_raw, "library", "ranking", ("lightgbm", "catboost"), default="lightgbm"),  # 库
        objective=_require_choice(  # 目标函数
            ranking_raw, "objective", "ranking", ("lambdarank", "rankxendcg", "yetirank"), default="lambdarank"
        ),  # 目标结束
        top_k_for_training=_require_int(ranking_raw, "top_k_for_training", "ranking", default=500),  # 训练候选上限
    )  # 排序协议结束
    evaluation = EvaluationConfig(  # 组装评估协议
        primary_metric=_require_str(evaluation_raw, "primary_metric", "evaluation", default=PRIMARY_METRIC_DEFAULT),  # 主指标
        protocol=_require_str(evaluation_raw, "protocol", "evaluation", default="offline_candidate_ranking"),  # 评估口径
        activity_tiers=_parse_activity_tiers(evaluation_raw.get("activity_tiers")),  # 活跃度分层
    )  # 评估协议结束

    if data.history_weeks <= 0 or data.valid_weeks <= 0 or data.test_weeks <= 0:  # 周数必须为正
        raise ExperimentConfigError("data week counts must be positive")  # 抛出错误
    if data.backtest_windows < 1:  # 至少保留官方窗口
        raise ExperimentConfigError("data.backtest_windows must be >= 1")  # 抛出错误
    if candidate.final_top_k <= 0:  # 最终 K 必须为正
        raise ExperimentConfigError("candidate.final_top_k must be positive")  # 抛出错误
    if candidate.union_top_k < candidate.final_top_k:  # 并集不能小于最终截断
        raise ExperimentConfigError("candidate.union_top_k must be >= candidate.final_top_k")  # 互斥校验
    if label.horizon_days < 1:  # 标签窗口至少 1 天
        raise ExperimentConfigError("label.horizon_days must be >= 1")  # 抛出错误
    if ranking.top_k_for_training < candidate.final_top_k:  # 排序训练候选不能小于最终 K
        raise ExperimentConfigError("ranking.top_k_for_training must be >= candidate.final_top_k")  # 抛出错误
    if model_selection.checkpoint_shortlist_size <= 0:
        raise ExperimentConfigError("model_selection.checkpoint_shortlist_size must be positive")

    return ExperimentConfig(  # 返回完整配置
        experiment=experiment,  # 元信息
        data=data,  # 数据协议
        model_selection=model_selection,
        candidate=candidate,  # 候选协议
        label=label,  # 标签协议
        ranking=ranking,  # 排序协议
        evaluation=evaluation,  # 评估协议
        source_path=config_path.resolve(),  # 配置绝对路径
    )  # 配置组装结束
