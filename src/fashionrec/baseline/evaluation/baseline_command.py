"""Evaluate current recall/fusion variants without training a new model."""  # 评估当前召回与融合变体，不训练新模型

from __future__ import annotations  # 启用延迟注解

import argparse  # 命令行参数
import json  # 打印摘要
from pathlib import Path  # 路径
from typing import Any, Mapping  # 类型

from fashionrec.baseline.data.manifest import build_processed_hm_manifest  # 数据清单
from fashionrec.baseline.evaluation.candidate_diagnostics import diagnose_users  # 候选诊断
from fashionrec.baseline.evaluation.experiment_report import (  # 实验报告
    save_candidate_diagnostics,  # 候选覆盖报告
    save_experiment_outputs,  # 落盘
    score_users,  # 计分
    utc_run_id,  # 运行 ID
)  # 报告模块结束
from fashionrec.baseline.evaluation.offline_eval import (  # 复用现有融合评估上下文
    FusionEvalContext,  # 上下文类型
    build_fusion_eval_context,  # 构建上下文
    default_sasrec_recall_csv,  # 默认序列召回路径
)  # 离线评估结束
from fashionrec.baseline.evaluation.weight_search import load_best_weights  # 只加载已冻结权重，不在此搜索
from fashionrec.experiment.config import ExperimentConfig, load_experiment_config  # 实验协议
from fashionrec.baseline.ranking.fusion import (  # 融合
    ACTIVITY_WEIGHTS,  # 默认分层权重
    ActivityTier,  # 分层类型
    fuse_candidates,  # 加权融合
    get_channel_weights_for_user,  # 按用户取权重
)  # 融合结束


def assert_not_tuning_on_test(eval_split: str, searching: bool) -> None:  # test 禁止搜权
    if eval_split == "test" and searching:  # 若在 test 上调参
        raise ValueError("test split must not be used for weight search, feature selection, or model selection")  # 拒绝


def load_frozen_fusion_weights(  # 加载已冻结的 valid 搜权结果，绝不重新搜索
    eval_split: str,  # valid 或 test
    weights_json: Path | None,  # 权重文件
) -> dict[ActivityTier, dict[str, float]] | None:  # 分层权重或空
    assert_not_tuning_on_test(eval_split, searching=False)  # 本函数只加载，不搜索
    if weights_json is None:  # 未提供权重文件
        return None  # 跳过搜权融合变体
    payload = load_best_weights(weights_json)  # 读取 JSON
    return payload["best_weights"]  # 返回分层权重


def _channel_users(context: FusionEvalContext, channel: str, top_k: int) -> list[dict[str, Any]]:  # 单通道 Top-K 预测
    users: list[dict[str, Any]] = []  # 结果
    for row in context.users:  # 遍历评估用户
        candidates = row["channel_candidates"].get(channel, [])  # 该通道候选
        pred = [item_id for item_id, _score in candidates[:top_k]]  # 截断到最终 K
        users.append(  # 组装计分行
            {  # 用户字典
                "user_id": row["user_id"],  # 用户
                "actual": row["actual_items"],  # 标签
                "pred": pred,  # 预测
                "history_len": len(row["history"]),  # 历史长度
            }  # 字典结束
        )  # 追加结束
    return users  # 返回


def _fusion_users(  # 按给定分层权重融合后的预测
    context: FusionEvalContext,  # 评估上下文
    activity_weights: dict[ActivityTier, dict[str, float]],  # 分层权重
    exclude_seen: bool = False,  # 是否排除已购
) -> list[dict[str, Any]]:  # 计分行
    users: list[dict[str, Any]] = []  # 结果
    for row in context.users:  # 遍历用户
        weights = get_channel_weights_for_user(  # 按历史长度取权重
            len(row["history"]),  # 历史长度
            context.sequence_channel,  # 序列通道名
            activity_weights=activity_weights,  # 分层权重表
        )  # 权重结束
        fused = fuse_candidates(  # 融合
            user_id=row["user_id"],  # 用户
            user_history=row["history_set"],  # 历史集合
            channel_candidates=row["channel_candidates"],  # 各通道候选
            channel_weights=weights,  # 通道权重
            top_k=context.final_top_k,  # 最终 K
            exclude_seen=exclude_seen,  # 排除已购
        )  # 融合结束
        users.append(  # 计分行
            {  # 字典
                "user_id": row["user_id"],  # 用户
                "actual": row["actual_items"],  # 标签
                "pred": [item_id for item_id, _score in fused],  # 预测
                "history_len": len(row["history"]),  # 历史长度
            }  # 字典结束
        )  # 追加结束
    return users  # 返回


def _score_named_variant(  # 为一个变体计分
    name: str,  # 变体名
    users: list[dict[str, Any]],  # 计分行
    config: ExperimentConfig,  # 实验配置
    extra: Mapping[str, Any] | None = None,  # 附加字段
) -> dict[str, Any]:  # 含 overall 与 per_tier
    k = config.candidate.final_top_k  # 最终 K
    payload = {"variant": name, "eval_protocol": config.evaluation.protocol}  # 基础附加字段
    if extra:  # 有额外字段
        payload.update(dict(extra))  # 合并
    overall, per_tier = score_users(  # 计分
        users,  # 用户预测
        k,  # K
        config.evaluation.activity_tiers,  # 分层
        extra=payload,  # 附加
    )  # 计分结束
    return {"name": name, "overall": overall, "per_tier": per_tier}  # 变体结果


def collect_baseline_variants(  # 收集当前代码的六组对照，不训练、不搜权
    context: FusionEvalContext,  # 已构建的评估上下文
    config: ExperimentConfig,  # 实验协议
    searched_weights: dict[ActivityTier, dict[str, float]] | None,  # 冻结的 valid 搜权结果
    exclude_seen: bool = False,  # 融合是否排除已购
    sequence_csv_exists: bool = True,  # 序列召回文件是否存在
) -> list[dict[str, Any]]:  # 变体结果列表
    k = config.candidate.final_top_k  # 最终 K
    variants: list[dict[str, Any]] = []  # 结果
    for channel in ("popular", "category_popular", "item2item"):  # 三路规则召回
        variants.append(  # 单通道
            _score_named_variant(channel, _channel_users(context, channel, k), config)  # 计分
        )  # 追加结束
    sequence_channel = context.sequence_channel  # sasrec 或 sasrecf
    if sequence_csv_exists:  # 有序列召回文件才评估该通道
        variants.append(  # 序列单通道
            _score_named_variant(  # 计分
                sequence_channel,  # 通道名
                _channel_users(context, sequence_channel, k),  # 预测
                config,  # 配置
            )  # 计分结束
        )  # 追加结束
    else:  # 缺少序列召回
        variants.append(  # 记录跳过原因，避免误当成训练失败
            {  # 占位结果
                "name": sequence_channel,  # 通道名
                "overall": {"skipped": True, "reason": "sequence recall csv not found; not training"},  # 跳过
                "per_tier": [],  # 无分层
            }  # 占位结束
        )  # 追加结束
    variants.append(  # 当前默认分层权重融合
        _score_named_variant(  # 计分
            "fusion_default_weights",  # 变体名
            _fusion_users(context, ACTIVITY_WEIGHTS, exclude_seen=exclude_seen),  # 默认权重预测
            config,  # 配置
            extra={"weights_source": "ACTIVITY_WEIGHTS"},  # 权重来源
        )  # 计分结束
    )  # 追加结束
    if searched_weights is not None:  # 提供了冻结权重
        variants.append(  # valid 搜权后的融合
            _score_named_variant(  # 计分
                "fusion_valid_search_weights",  # 变体名
                _fusion_users(context, searched_weights, exclude_seen=exclude_seen),  # 搜权权重预测
                config,  # 配置
                extra={"weights_source": "weights_json"},  # 权重来源
            )  # 计分结束
        )  # 追加结束
    else:  # 无冻结权重
        variants.append(  # 明确跳过，而不是在 test 上现搜
            {  # 占位
                "name": "fusion_valid_search_weights",  # 变体名
                "overall": {"skipped": True, "reason": "no frozen weights json; refusing to search on this split"},  # 原因
                "per_tier": [],  # 无分层
            }  # 占位结束
        )  # 追加结束
    return variants  # 返回全部变体


def run_baseline(  # 在统一协议下评估当前代码基线
    config: ExperimentConfig,  # 实验配置
    eval_split: str,  # valid 或 test
    weights_json: Path | None = None,  # 冻结权重
    output_root: Path | None = None,  # 实验输出根目录
    sasrec_recall_csv: Path | None = None,  # 可选序列召回
    exclude_seen: bool = False,  # 是否排除已购
) -> Path:  # 返回本次 run 目录
    if eval_split not in {"valid", "test"}:  # 非法划分
        raise ValueError("eval_split must be 'valid' or 'test'")  # 抛出错误
    assert_not_tuning_on_test(eval_split, searching=False)  # 基线脚本不搜权
    searched_weights = load_frozen_fusion_weights(eval_split, weights_json)  # 只加载冻结权重

    recall_csv = Path(sasrec_recall_csv) if sasrec_recall_csv is not None else default_sasrec_recall_csv(eval_split)  # 序列召回路径
    sequence_csv_exists = recall_csv.exists()  # 是否已有召回文件
    context = build_fusion_eval_context(  # 构建评估上下文（用已有召回，不训练）
        eval_split=eval_split,  # 评估划分
        recall_top_k=config.candidate.sequence_top_k,  # 序列 Top-K
        popular_recall_top_k=config.candidate.popular_top_k,  # 热门 Top-K
        category_popular_recall_top_k=config.candidate.category_popular_top_k,  # 类别热门 Top-K
        item2item_recall_top_k=config.candidate.item2item_top_k,  # item2item Top-K
        final_top_k=config.candidate.final_top_k,  # 最终 K
        sasrec_recall_csv=recall_csv,  # 序列召回
        max_user_history=config.data.max_user_history,  # 用户分层与已购过滤历史上限
    )  # 上下文结束

    variants = collect_baseline_variants(  # 六组对照
        context,  # 上下文
        config,  # 协议
        searched_weights,  # 冻结权重
        exclude_seen=exclude_seen,  # 排除已购
        sequence_csv_exists=sequence_csv_exists,  # 序列文件是否存在
    )  # 变体结束

    run_id = utc_run_id(f"{config.experiment.name}_{eval_split}")  # 运行 ID
    run_dir = (output_root or Path("outputs/experiments")) / run_id  # 输出目录
    manifest = build_processed_hm_manifest(  # 当前处理后数据快照
        preprocess={  # 协议参数
            "experiment_name": config.experiment.name,  # 实验名
            "seed": config.experiment.seed,  # 种子
            "eval_split": eval_split,  # 评估划分
            "config_path": str(config.source_path),  # 配置路径
        }  # 预处理字段结束
    )  # 清单结束
    metrics = {  # 总体报告
        "run_id": run_id,  # 运行 ID
        "eval_split": eval_split,  # 划分
        "protocol": config.evaluation.protocol,  # 评估口径
        "primary_metric": config.evaluation.primary_metric,  # 主指标
        "final_top_k": config.candidate.final_top_k,  # K
        "sequence_channel": context.sequence_channel,  # 序列通道
        "sequence_recall_csv": str(recall_csv),  # 召回文件
        "weights_json": str(weights_json) if weights_json else None,  # 冻结权重路径
        "variants": {item["name"]: item["overall"] for item in variants},  # 各变体总体指标
    }  # 指标结束
    per_tier_rows: list[dict[str, Any]] = []  # 分层长表
    for item in variants:  # 每个变体
        for row in item["per_tier"]:  # 每一层
            per_tier_rows.append({"variant": item["name"], **row})  # 加上变体名
    output_paths = save_experiment_outputs(run_dir, manifest, metrics, per_tier_rows, k=config.candidate.final_top_k)  # 落盘
    diagnostic_users = [  # 诊断输入
        {
            "user_id": row["user_id"],  # 用户
            "actual_items": row["actual_items"],  # 标签
            "history": row["history"],  # 历史序列
            "history_set": row["history_set"],  # 历史集合
            "history_len": len(row["history"]),  # 长度
            "channel_candidates": row["channel_candidates"],  # 各通道候选
        }
        for row in context.users  # 评估用户
    ]
    channels = ["popular", "category_popular", "item2item"]  # 规则通道
    if sequence_csv_exists:  # 有序列召回
        channels.append(context.sequence_channel)  # 序列通道
    diagnostics = diagnose_users(  # 候选诊断
        diagnostic_users,  # 用户
        channels=channels,  # 通道顺序
        activity_tiers=config.evaluation.activity_tiers,  # 活跃度分层
        union_k_for_counts=config.candidate.union_top_k,  # 并集 K
    )
    coverage_paths = save_candidate_diagnostics(run_dir, diagnostics)  # JSON + CSV
    output_paths.update(coverage_paths)  # 合并路径
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics, "candidate_diagnostics": str(coverage_paths["candidate_diagnostics"])}, ensure_ascii=False, indent=2))  # 打印摘要
    return run_dir  # 返回目录


def main(argv: list[str] | None = None) -> None:  # 命令行入口
    parser = argparse.ArgumentParser(  # 参数解析
        prog="fashionrec baseline",  # 统一 CLI 子命令名
        description="Evaluate current-code baselines without training. test never searches weights.",  # 说明
    )  # 解析器结束
    parser.add_argument("--config", type=Path, default=Path("configs/baseline/experiment.yaml"))  # 实验配置
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")  # 评估划分
    parser.add_argument("--weights-json", type=Path, default=None, help="Frozen valid-search weights; never searched here")  # 冻结权重
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments"))  # 输出根目录
    parser.add_argument("--sasrec-recall-csv", type=Path, default=None)  # 可选序列召回
    parser.add_argument("--exclude-seen", action="store_true")  # 排除已购
    args = parser.parse_args(argv)  # 解析显式命令参数

    config = load_experiment_config(args.config)  # 加载协议
    run_baseline(  # 跑基线评估（不训练）
        config=config,  # 配置
        eval_split=args.eval_split,  # 划分
        weights_json=args.weights_json,  # 冻结权重
        output_root=args.output_root,  # 输出根
        sasrec_recall_csv=args.sasrec_recall_csv,  # 序列召回
        exclude_seen=args.exclude_seen,  # 排除已购
    )  # 结束


if __name__ == "__main__":  # 直接运行
    main()  # 入口
