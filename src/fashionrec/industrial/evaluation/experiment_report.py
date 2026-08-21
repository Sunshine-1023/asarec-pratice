"""Write structured experiment reports for comparable runs."""  # 写出可对照的结构化实验报告

from __future__ import annotations  # 启用延迟注解

import csv  # 写分层指标 CSV
import json  # 写 JSON
from datetime import datetime, timezone  # 运行时间
from pathlib import Path  # 路径
from typing import Any, Iterable, Mapping, Sequence  # 类型

from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # 对照评估统一 ID
from fashionrec.shared.metrics.ranking import hit_at_k, map_at_k, mean_metric, ndcg_at_k, recall_at_k  # 统一指标
from fashionrec.experiment.config import REQUIRED_TIERS, classify_activity_tier  # 活跃度分层


DEFAULT_ACTIVITY_TIERS: dict[str, tuple[int, int | None]] = {  # 与 configs/experiment.yaml 一致，供不加载 YAML 的 evaluate 使用
    "cold_start": (0, 0),
    "low": (1, 2),
    "medium": (3, 9),
    "high": (10, None),
}
RANKER_COMPARE_VARIANT_NAMES = (  # 阶段 4.3 固定对照名
    "fusion_default_weights",
    "fusion_valid_search_weights",
    "lambdarank",
    "lambdarank_rerank",
)
TIER_REGRESSION_TOLERANCE = 0.05  # 任一主要分层平均 MAP 相对下降超过 5% 即否决
PRIMARY_GATE_METRIC_NAMES = ("MAP", "Recall", "NDCG")  # Hit 只报告，不进替换门闩


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


def save_candidate_diagnostics(  # 候选诊断 JSON + CSV
    run_dir: Path,  # 实验目录
    payload: Mapping[str, Any],  # diagnose_users 输出
) -> dict[str, Path]:  # 写出的文件
    run_dir.mkdir(parents=True, exist_ok=True)  # 确保目录
    json_path = write_json(run_dir / "candidate_diagnostics.json", payload)  # 完整 JSON
    channel_rows = list(payload.get("per_channel_metrics", []))  # 单通道
    union_rows = list(payload.get("union_metrics", []))  # 并集
    channel_fields = sorted({key for row in channel_rows for key in row.keys()}, key=lambda name: (name != "channel", name)) if channel_rows else ["channel", "k", "scope", "n_users"]  # 列
    union_fields = sorted({key for row in union_rows for key in row.keys()}, key=lambda name: (name != "k", name)) if union_rows else ["k", "scope", "n_users"]  # 列
    return {  # 路径
        "candidate_diagnostics": json_path,  # JSON
        "candidate_channel_metrics": write_csv(run_dir / "candidate_channel_metrics.csv", channel_rows, channel_fields),  # 单通道
        "candidate_union_metrics": write_csv(run_dir / "candidate_union_metrics.csv", union_rows, union_fields),  # 并集
        "candidate_pair_jaccard": write_csv(  # 通道对
            run_dir / "candidate_pair_jaccard.csv",
            list(payload.get("channel_pair_jaccard", [])),
            ["channel_a", "channel_b", "mean_jaccard", "n_users"],
        ),
        "candidate_exclusive_hits": write_csv(  # 独占命中
            run_dir / "candidate_exclusive_hits.csv",
            list(payload.get("exclusive_hits", [])),
            ["channel", "k", "mean_exclusive_hit_rate", "mean_exclusive_hits", "n_users_with_hits"],
        ),
    }  # 返回


def skipped_variant(name: str, reason: str) -> dict[str, Any]:  # 缺产物时占位，避免把跳过当成训练失败
    return {"name": name, "overall": {"skipped": True, "reason": reason}, "per_tier": []}


def score_named_variant(  # 对已有 pred 的用户列表计分并带上变体名
    name: str,
    users: Sequence[Mapping[str, Any]],
    k: int,
    activity_tiers: Mapping[str, tuple[int, int | None]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"variant": name}
    if extra:
        payload.update(dict(extra))
    overall, per_tier = score_users(users, k, activity_tiers, extra=payload)
    return {"name": name, "overall": overall, "per_tier": per_tier}


def load_ranked_predictions(path: str | Path, *, top_k: int) -> dict[str, list[str]]:  # 从 ranker-predict CSV 还原每用户 Top-K
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    grouped: dict[str, list[tuple[int, str]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "user_id" not in reader.fieldnames or "item_id" not in reader.fieldnames:
            raise ValueError(f"ranked csv must contain user_id and item_id: {path}")
        has_rank = "rank" in reader.fieldnames
        for row in reader:
            user_id = canonical_user_id(row["user_id"])
            item_id = canonical_item_id(row["item_id"])
            rank = int(float(row["rank"])) if has_rank and str(row.get("rank", "")).strip() else 10**9
            grouped.setdefault(user_id, []).append((rank, item_id))
    predictions: dict[str, list[str]] = {}
    for user_id, rows in grouped.items():
        rows.sort(key=lambda item: (item[0], item[1]))
        ordered: list[str] = []
        seen: set[str] = set()
        for _rank, item_id in rows:
            if item_id in seen:
                continue
            seen.add(item_id)
            ordered.append(item_id)
            if len(ordered) >= top_k:
                break
        predictions[user_id] = ordered
    return predictions


def _variant_by_name(variants: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in variants:
        indexed[str(item["name"])] = item
    return indexed


def _is_skipped(variant: Mapping[str, Any] | None) -> bool:
    if variant is None:
        return True
    overall = variant.get("overall") or {}
    return bool(overall.get("skipped"))


def _metric_keys(k: int) -> tuple[str, ...]:
    return tuple(f"{name}@{k}" for name in PRIMARY_GATE_METRIC_NAMES)


def _tier_map(per_tier: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["activity_tier"]): row for row in per_tier if "activity_tier" in row}


def compare_ranker_variants(  # 判断 LambdaRank 是否可替换默认 RRF；只出建议，不改 pipeline
    variants: Sequence[Mapping[str, Any]],
    *,
    k: int = 12,
    baseline: str = "fusion_valid_search_weights",
    candidate: str = "lambdarank",
    fallback_baseline: str = "fusion_default_weights",
    tier_regression_tolerance: float = TIER_REGRESSION_TOLERANCE,
) -> dict[str, Any]:
    indexed = _variant_by_name(variants)
    gate_keys = _metric_keys(k)
    candidate_variant = indexed.get(candidate)
    baseline_name = baseline
    baseline_variant = indexed.get(baseline)
    if _is_skipped(baseline_variant):
        baseline_name = fallback_baseline
        baseline_variant = indexed.get(fallback_baseline)

    payload: dict[str, Any] = {
        "k": k,
        "requested_baseline": baseline,
        "effective_baseline": baseline_name,
        "candidate": candidate,
        "gate_metrics": list(gate_keys),
        "tier_regression_tolerance": tier_regression_tolerance,
        "variants": {item["name"]: item.get("overall", {}) for item in variants},
        "deltas": {},
        "improved_metrics": [],
        "overall_improved": False,
        "tier_regressions": [],
        "major_tier_regression": False,
        "replace_default_ranker": False,
        "pipeline_default_unchanged": True,
        "reason": "",
    }

    if _is_skipped(candidate_variant):
        reason = "missing"
        if candidate_variant is not None:
            reason = str((candidate_variant.get("overall") or {}).get("reason") or "skipped")
        payload["reason"] = f"{candidate} skipped: {reason}"
        return payload
    if _is_skipped(baseline_variant):
        payload["reason"] = f"no usable RRF baseline ({baseline} and {fallback_baseline} skipped)"
        return payload

    assert candidate_variant is not None and baseline_variant is not None
    candidate_overall = dict(candidate_variant["overall"])
    baseline_overall = dict(baseline_variant["overall"])
    deltas: dict[str, float] = {}
    improved: list[str] = []
    for key in (*gate_keys, f"Hit@{k}"):
        if key not in candidate_overall or key not in baseline_overall:
            continue
        delta = float(candidate_overall[key]) - float(baseline_overall[key])
        deltas[key] = delta
        if key in gate_keys and delta > 0:
            improved.append(key)
    payload["deltas"] = deltas
    payload["improved_metrics"] = improved
    map_key = f"MAP@{k}"
    payload["overall_improved"] = bool(improved)
    payload["primary_metric_improved"] = deltas.get(map_key, 0.0) > 0.0

    regressions: list[dict[str, Any]] = []
    candidate_tiers = _tier_map(list(candidate_variant.get("per_tier") or []))
    baseline_tiers = _tier_map(list(baseline_variant.get("per_tier") or []))
    for tier in REQUIRED_TIERS:
        baseline_row = baseline_tiers.get(tier)
        candidate_row = candidate_tiers.get(tier)
        if baseline_row is None or candidate_row is None:
            continue
        if int(baseline_row.get("n_users") or 0) <= 0 or int(candidate_row.get("n_users") or 0) <= 0:
            continue
        baseline_map = float(baseline_row.get(map_key) or 0.0)
        candidate_map = float(candidate_row.get(map_key) or 0.0)
        if baseline_map <= 0:
            continue
        relative_drop = (baseline_map - candidate_map) / baseline_map
        if relative_drop > tier_regression_tolerance:
            regressions.append(
                {
                    "activity_tier": tier,
                    "metric": map_key,
                    "baseline": baseline_map,
                    "candidate": candidate_map,
                    "relative_drop": relative_drop,
                }
            )
    payload["tier_regressions"] = regressions
    payload["major_tier_regression"] = bool(regressions)
    replace = payload["primary_metric_improved"] and not payload["major_tier_regression"]
    payload["replace_default_ranker"] = replace
    if replace:
        payload["reason"] = (
            f"{candidate} improves {', '.join(improved)} vs {baseline_name} without MAP tier drop > "
            f"{tier_regression_tolerance:.0%}; keep pipeline on RRF until ranking.enabled is flipped"
        )
    elif payload["major_tier_regression"]:
        dropped = ", ".join(row["activity_tier"] for row in regressions)
        payload["reason"] = f"{candidate} has major MAP regression on tiers: {dropped}"
    elif not payload["primary_metric_improved"]:
        payload["reason"] = f"{candidate} does not improve primary metric MAP@{k} vs {baseline_name}"
    else:
        payload["reason"] = f"{candidate} does not pass the replacement gate vs {baseline_name}"
    return payload


def save_ranker_comparison(path: str | Path, payload: Mapping[str, Any]) -> Path:  # 写出对照与 gate
    return write_json(Path(path), payload)
