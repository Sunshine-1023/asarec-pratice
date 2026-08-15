"""Write structured experiment reports for comparable runs."""  # 写出可对照的结构化实验报告

from __future__ import annotations  # 启用延迟注解

import csv  # 写分层指标 CSV
import json  # 写 JSON
from datetime import datetime, timezone  # 运行时间
from pathlib import Path  # 路径
from typing import Any, Iterable, Mapping, Sequence  # 类型

from src.evaluate.metrics import hit_at_k, map_at_k, mean_metric, ndcg_at_k, recall_at_k  # 统一指标
from src.experiment.config import REQUIRED_TIERS, classify_activity_tier  # 活跃度分层


def utc_run_id(prefix: str, now: datetime | None = None) -> str:  # 生成运行 ID
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")  # UTC 时间戳
    return f"{prefix}_{stamp}"  # 前缀加时间


def per_user_metrics(  # 计算单用户指标
    actual: Iterable[object],  # 真实标签
    pred: Sequence[object],  # 预测列表
    k: int,  # Top-K
) -> dict[str, float]:  # 返回指标字典
    return {  # 四项核心指标
        f"MAP@{k}": map_at_k(actual, pred, k),  # MAP
        f"Recall@{k}": recall_at_k(actual, pred, k),  # Recall
        f"NDCG@{k}": ndcg_at_k(actual, pred, k),  # NDCG
        f"Hit@{k}": hit_at_k(actual, pred, k),  # Hit
    }  # 指标结束


def aggregate_metrics(  # 对多用户指标取平均
    rows: Sequence[Mapping[str, float]],  # 每用户指标
    k: int,  # Top-K
    extra: Mapping[str, Any] | None = None,  # 附加字段
) -> dict[str, Any]:  # 返回汇总
    keys = [f"MAP@{k}", f"Recall@{k}", f"NDCG@{k}", f"Hit@{k}"]  # 需要平均的键
    summary: dict[str, Any] = {  # 先写用户数
        "users_evaluated": len(rows),  # 评估用户数
        "k": k,  # Top-K
    }  # 基础字段结束
    for key in keys:  # 逐指标平均
        summary[key] = mean_metric([float(row[key]) for row in rows])  # 用户平均
    if extra:  # 有附加字段
        summary.update(dict(extra))  # 合并
    return summary  # 返回汇总


def metrics_by_activity_tier(  # 按活跃度分层汇总
    users: Sequence[Mapping[str, Any]],  # 每用户：history_len / actual / pred
    k: int,  # Top-K
    activity_tiers: Mapping[str, tuple[int, int | None]],  # 分层区间
) -> list[dict[str, Any]]:  # 返回分层行
    grouped: dict[str, list[dict[str, float]]] = {tier: [] for tier in REQUIRED_TIERS}  # 分层桶
    for user in users:  # 遍历用户
        history_len = int(user["history_len"])  # 历史长度
        tier = classify_activity_tier(history_len, activity_tiers)  # 判定分层
        grouped[tier].append(per_user_metrics(user["actual"], user["pred"], k))  # 计算该用户指标
    rows: list[dict[str, Any]] = []  # 分层结果
    for tier in REQUIRED_TIERS:  # 固定输出顺序
        summary = aggregate_metrics(grouped[tier], k, extra={"activity_tier": tier, "n_users": len(grouped[tier])})  # 分层平均
        rows.append(summary)  # 追加一行
    return rows  # 返回全部层


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:  # 写 JSON
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 写出
    return path  # 返回路径


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:  # 写 CSV
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录
    with path.open("w", newline="", encoding="utf-8") as handle:  # 打开文件
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))  # 按指定列写
        writer.writeheader()  # 表头
        for row in rows:  # 逐行
            writer.writerow({key: row.get(key, "") for key in fieldnames})  # 缺列写空
    return path  # 返回路径


def score_users(  # 对带 pred 的用户列表计算总体与分层指标
    users: Sequence[Mapping[str, Any]],  # 每用户 actual / pred / history_len
    k: int,  # Top-K
    activity_tiers: Mapping[str, tuple[int, int | None]],  # 分层区间
    extra: Mapping[str, Any] | None = None,  # 附加字段
) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # 总体指标, 分层指标
    per_user = [per_user_metrics(user["actual"], user["pred"], k) for user in users]  # 每用户指标
    overall = aggregate_metrics(per_user, k, extra=extra)  # 总体平均
    per_tier = metrics_by_activity_tier(users, k, activity_tiers)  # 分层平均
    return overall, per_tier  # 返回两份结果


def save_experiment_outputs(  # 将一次实验的清单与指标落到 run 目录
    run_dir: Path,  # 输出目录 outputs/experiments/<run_id>
    manifest: Mapping[str, Any],  # 数据清单
    metrics: Mapping[str, Any],  # 总体指标
    per_tier_rows: Sequence[Mapping[str, Any]],  # 分层指标
    k: int = 12,  # Top-K
) -> dict[str, Path]:  # 返回写出的文件路径
    run_dir.mkdir(parents=True, exist_ok=True)  # 创建运行目录
    tier_fields = [  # 分层 CSV 列顺序
        "variant",  # 对照变体名（单通道或融合）
        "activity_tier",  # 分层名
        "n_users",  # 该层用户数
        "users_evaluated",  # 与 n_users 相同，便于对照
        f"MAP@{k}",  # MAP
        f"Recall@{k}",  # Recall
        f"NDCG@{k}",  # NDCG
        f"Hit@{k}",  # Hit
        "k",  # Top-K
    ]  # 列结束
    return {  # 写出三个标准文件
        "manifest": write_json(run_dir / "manifest.json", manifest),  # 数据快照
        "metrics": write_json(run_dir / "metrics.json", metrics),  # 总体指标
        "per_tier_metrics": write_csv(run_dir / "per_tier_metrics.csv", per_tier_rows, tier_fields),  # 分层指标
    }  # 路径字典结束
