"""Candidate recall diagnostics before ranking experiments."""  # 排序前的候选诊断报告

from __future__ import annotations  # 延迟注解

from collections.abc import Iterable, Mapping, Sequence  # 类型
from itertools import combinations  # 通道对
from pathlib import Path  # 路径
from typing import Any  # JSON 友好

from fashionrec.evaluation.coverage_metrics import (  # 覆盖指标
    candidate_count_summary,
    exclusive_hit_items,
    jaccard_similarity,
    top_item_ids,
    top_item_set,
    union_top_items,
    user_coverage,
)
from fashionrec.evaluation.metrics import canonicalize_item_set, hit_at_k, mean_metric, recall_at_k  # 召回指标
from fashionrec.experiment.config import REQUIRED_TIERS, classify_activity_tier  # 活跃度分层


DIAGNOSTICS_SCHEMA_VERSION = "hm.candidate_diagnostics.v1"  # 报告语义
DEFAULT_CHANNEL_KS = (50, 100, 300)  # 单通道 Recall/Hit@K
DEFAULT_UNION_KS = (100, 300, 500)  # 并集 Recall/Hit@K
WARM_COLD_STRATA = ("warm", "cold")  # 是否有历史
PURCHASE_STRATA = ("repeat_only", "new_only", "mixed")  # 标签复购结构


def _actual_set(actual: Iterable[object]) -> set[str]:  # 规范化标签
    return canonicalize_item_set(actual)  # 返回集合


def classify_warm_cold(history_len: int) -> str:  # 冷启动 vs 有历史
    return "cold" if history_len <= 0 else "warm"  # 划分


def classify_purchase_stratum(actual: set[str], history_set: set[str]) -> str | None:  # 复购结构
    if not actual:  # 无标签
        return None  # 跳过
    overlap = actual & history_set  # 复购 SKU
    if overlap and overlap == actual:  # 全部复购
        return "repeat_only"  # 纯复购
    if not overlap:  # 全部新购
        return "new_only"  # 纯新购
    return "mixed"  # 混合购物篮


def _score_predictions(actual: set[str], pred: Sequence[str], k: int) -> dict[str, float]:  # Recall/Hit@K
    return {  # 两指标
        f"Recall@{k}": recall_at_k(actual, pred, k),  # 召回
        f"Hit@{k}": hit_at_k(actual, pred, k),  # 命中
    }  # 返回


def _mean_metric_rows(rows: Sequence[Mapping[str, float]], key: str) -> float:  # 对某键取平均
    values = [float(row[key]) for row in rows if key in row]  # 收集
    return mean_metric(values)  # 平均


def diagnose_users(  # 对评估用户批量诊断
    users: Sequence[Mapping[str, Any]],  # user_id / actual / history / history_set / channel_candidates
    *,
    channels: Sequence[str] | None = None,  # 通道顺序
    channel_ks: Sequence[int] = DEFAULT_CHANNEL_KS,  # 单通道 K
    union_ks: Sequence[int] = DEFAULT_UNION_KS,  # 并集 K
    activity_tiers: Mapping[str, tuple[int, int | None]],  # 活跃度分层
    union_k_for_counts: int = 500,  # 统计候选数时的并集 K
) -> dict[str, Any]:  # 完整报告
    if not users:  # 无用户
        raise ValueError("users must not be empty")  # 拒绝
    channel_names = list(channels) if channels is not None else sorted(  # 默认从首用户推断
        {name for row in users for name in row.get("channel_candidates", {}).keys()}  # 全部通道名
    )
    if not channel_names:  # 没通道
        raise ValueError("channel_candidates must not be empty")  # 拒绝

    per_user_rows: list[dict[str, Any]] = []  # 每用户明细
    union_counts: list[int] = []  # 并集候选数
    users_with_candidates = 0  # 至少有一个候选的用户
    pair_jaccard_values: dict[tuple[str, str], list[float]] = {}  # 通道对 Jaccard

    for row in users:  # 逐用户
        actual = _actual_set(row.get("actual") or row.get("actual_items") or set())  # 标签
        history_set = {str(x) for x in row.get("history_set", set())}  # 历史集合
        history_len = int(row.get("history_len", len(row.get("history", []))))  # 历史长度
        channel_candidates: dict[str, list[tuple[str, float]]] = dict(row["channel_candidates"])  # 各通道
        union_list = union_top_items(channel_candidates, union_k_for_counts)  # 并集
        union_count = len(union_list)  # 候选数
        union_counts.append(union_count)  # 记录
        if union_count > 0:  # 有候选
            users_with_candidates += 1  # 覆盖 +1
        channel_topk = {channel: top_item_set(channel_candidates.get(channel, []), max(channel_ks)) for channel in channel_names}  # 各通道集合
        for left, right in combinations(channel_names, 2):  # 通道对
            score = jaccard_similarity(  # Jaccard
                top_item_ids(channel_candidates.get(left, []), max(channel_ks)),  # 左通道
                top_item_ids(channel_candidates.get(right, []), max(channel_ks)),  # 右通道
            )
            pair_jaccard_values.setdefault((left, right), []).append(score)  # 收集
        per_user_rows.append(  # 用户行
            {
                "user_id": row.get("user_id"),  # ID
                "history_len": history_len,  # 历史长度
                "activity_tier": classify_activity_tier(history_len, activity_tiers),  # 活跃度
                "warm_cold": classify_warm_cold(history_len),  # 冷/暖
                "purchase_stratum": classify_purchase_stratum(actual, history_set),  # 复购结构
                "union_candidate_count": union_count,  # 并集大小
                "channel_topk": channel_topk,  # 各通道 Top 集合
                "actual": actual,  # 标签
                "channel_candidates": channel_candidates,  # 原始候选
            }
        )  # 追加

    def _aggregate_metric(  # 对分层子集算 Recall/Hit
        predicate,
        *,
        channel: str | None,
        k: int,
        union: bool,
    ) -> dict[str, Any]:  # 一行汇总
        subset = [row for row in per_user_rows if predicate(row)]  # 过滤
        if not subset:  # 空层
            return {"n_users": 0, f"Recall@{k}": 0.0, f"Hit@{k}": 0.0}  # 零
        scores: list[dict[str, float]] = []  # 每用户
        for row in subset:  # 子集用户
            if union:  # 并集
                pred = union_top_items(row["channel_candidates"], k)  # 并集 Top-K
            else:  # 单通道
                pred = top_item_ids(row["channel_candidates"].get(channel or "", []), k)  # 通道 Top-K
            scores.append(_score_predictions(row["actual"], pred, k))  # 计分
        return {  # 汇总
            "n_users": len(subset),  # 人数
            f"Recall@{k}": _mean_metric_rows(scores, f"Recall@{k}"),  # 平均 Recall
            f"Hit@{k}": _mean_metric_rows(scores, f"Hit@{k}"),  # 平均 Hit
        }  # 返回

    per_channel_metrics: list[dict[str, Any]] = []  # 单通道表
    for channel in channel_names:  # 每通道
        for k in channel_ks:  # 各 K
            overall = _aggregate_metric(lambda _row: True, channel=channel, k=k, union=False)  # 全体
            per_channel_metrics.append({"channel": channel, "k": k, "scope": "overall", **overall})  # 总体
            for tier in REQUIRED_TIERS:  # 活跃度
                row = _aggregate_metric(lambda r, t=tier: r["activity_tier"] == t, channel=channel, k=k, union=False)  # 分层
                per_channel_metrics.append({"channel": channel, "k": k, "scope": f"activity:{tier}", **row})  # 追加
            for warm in WARM_COLD_STRATA:  # 冷/暖
                row = _aggregate_metric(lambda r, w=warm: r["warm_cold"] == w, channel=channel, k=k, union=False)  # 分层
                per_channel_metrics.append({"channel": channel, "k": k, "scope": f"warm_cold:{warm}", **row})  # 追加
            for purchase in PURCHASE_STRATA:  # 复购结构
                row = _aggregate_metric(lambda r, p=purchase: r["purchase_stratum"] == p, channel=channel, k=k, union=False)  # 分层
                per_channel_metrics.append({"channel": channel, "k": k, "scope": f"purchase:{purchase}", **row})  # 追加

    union_metrics: list[dict[str, Any]] = []  # 并集表
    for k in union_ks:  # 各 K
        overall = _aggregate_metric(lambda _row: True, channel=None, k=k, union=True)  # 全体
        union_metrics.append({"k": k, "scope": "overall", **overall})  # 总体
        for tier in REQUIRED_TIERS:  # 活跃度
            row = _aggregate_metric(lambda r, t=tier: r["activity_tier"] == t, channel=None, k=k, union=True)  # 分层
            union_metrics.append({"k": k, "scope": f"activity:{tier}", **row})  # 追加
        for warm in WARM_COLD_STRATA:  # 冷/暖
            row = _aggregate_metric(lambda r, w=warm: r["warm_cold"] == w, channel=None, k=k, union=True)  # 分层
            union_metrics.append({"k": k, "scope": f"warm_cold:{warm}", **row})  # 追加
        for purchase in PURCHASE_STRATA:  # 复购结构
            row = _aggregate_metric(lambda r, p=purchase: r["purchase_stratum"] == p, channel=None, k=k, union=True)  # 分层
            union_metrics.append({"k": k, "scope": f"purchase:{purchase}", **row})  # 追加

    pair_rows: list[dict[str, Any]] = []  # 通道对 Jaccard
    for (left, right), values in sorted(pair_jaccard_values.items()):  # 排序输出
        pair_rows.append(  # 一行
            {
                "channel_a": left,  # 左通道
                "channel_b": right,  # 右通道
                "mean_jaccard": mean_metric(values),  # 平均 Jaccard
                "n_users": len(values),  # 用户数
            }
        )  # 追加

    exclusive_rows: list[dict[str, Any]] = []  # 独占命中
    eval_k = max(channel_ks)  # 用最大单通道 K 评估独占
    for channel in channel_names:  # 每通道
        channel_rates: list[float] = []  # 独占占全部命中的比例
        exclusive_counts: list[float] = []  # 独占 SKU 数
        for row in per_user_rows:  # 用户
            if not row["actual"]:  # 无标签
                continue  # 跳过
            topk = {name: top_item_set(row["channel_candidates"].get(name, []), eval_k) for name in channel_names}  # Top 集合
            hits = set().union(*(preds & row["actual"] for preds in topk.values()))  # 全部命中
            if not hits:  # 无命中
                continue  # 跳过
            exclusive = exclusive_hit_items(row["actual"], topk)[channel]  # 该通道独占
            channel_rates.append(len(exclusive) / len(hits))  # 占比
            exclusive_counts.append(float(len(exclusive)))  # 数量
        exclusive_rows.append(  # 汇总
            {
                "channel": channel,  # 通道
                "k": eval_k,  # 评估 K
                "mean_exclusive_hit_rate": mean_metric(channel_rates),  # 平均独占率
                "mean_exclusive_hits": mean_metric(exclusive_counts),  # 平均独占 SKU 数
                "n_users_with_hits": len(channel_rates),  # 至少一通道命中的用户
            }
        )  # 追加

    coverage = {  # 覆盖摘要
        "users_evaluated": len(users),  # 评估用户
        "users_with_candidates": users_with_candidates,  # 有候选用户
        "user_coverage": user_coverage(users_with_candidates, len(users)),  # 覆盖率
        "union_candidate_count": candidate_count_summary(union_counts),  # 并集候选数分布
    }  # 覆盖结束

    return {  # 完整 JSON
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,  # 版本
        "channels": channel_names,  # 通道列表
        "channel_ks": list(channel_ks),  # 单通道 K
        "union_ks": list(union_ks),  # 并集 K
        "coverage": coverage,  # 覆盖
        "per_channel_metrics": per_channel_metrics,  # 单通道
        "union_metrics": union_metrics,  # 并集
        "channel_pair_jaccard": pair_rows,  # 通道对
        "exclusive_hits": exclusive_rows,  # 独占命中
    }  # 返回


def assert_candidate_diagnostics_present(run_dir: Path) -> Path:  # 排序比较前必须有诊断报告
    run_dir = Path(run_dir)  # 规范化
    json_path = run_dir / "candidate_diagnostics.json"  # JSON
    if not json_path.is_file():  # 缺失
        raise FileNotFoundError(  # 拒绝进入排序比较
            f"candidate coverage report missing: {json_path}; run candidate diagnostics before ranking comparison"
        )
    return json_path  # 返回路径
