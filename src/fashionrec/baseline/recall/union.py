"""Deterministic per-user candidate union with multi-channel evidence preserved."""  # 候选并集

from __future__ import annotations  # 延迟注解

import csv  # 证据 CSV
from collections import defaultdict  # 按用户分组
from collections.abc import Iterable  # 类型
from dataclasses import dataclass  # 证据行
from datetime import UTC, datetime  # 时间戳
from pathlib import Path  # 路径

from fashionrec.shared.domain.candidates import Candidate  # 统一候选


UNION_SCHEMA_VERSION = "hm.candidate_union.v2"  # 并集证据语义
DEFAULT_UNION_TOP_K = 500  # 默认并集上限


@dataclass(frozen=True, slots=True)  # 宽表证据
class UnionEvidenceRow:  # 每个 user-item 一行
    user_id: str  # 用户
    item_id: str  # 商品
    split: str  # valid / test
    channel_count: int  # 命中通道数
    best_channel_rank: int  # 最佳通道内 rank
    max_channel_score: float  # 最高通道分
    source_timestamp: str  # 索引/召回生成时刻（UTC ISO）
    feature_version: str  # 证据 schema 版本

    def as_dict(self, *, channels: tuple[str, ...], evidence: dict[str, Candidate]) -> dict[str, object]:  # 展开通道列
        row: dict[str, object] = {  # 基础字段
            "user_id": self.user_id,
            "item_id": self.item_id,
            "split": self.split,
            "channel_count": self.channel_count,
            "best_channel_rank": self.best_channel_rank,
            "max_channel_score": self.max_channel_score,
            "source_timestamp": self.source_timestamp,
            "feature_version": self.feature_version,
        }
        for channel in channels:  # 每路 present/rank/score
            candidate = evidence.get(channel)
            row[f"{channel}_present"] = int(candidate is not None)
            row[f"{channel}_score"] = float(candidate.score) if candidate is not None else 0.0
            row[f"{channel}_rank"] = int(candidate.rank) if candidate is not None else 0
        return row  # 返回


def _default_source_timestamp() -> str:  # UTC ISO 时间戳
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")  # 稳定格式


def _dedupe_channel_rows(rows: Iterable[Candidate]) -> dict[tuple[str, str], Candidate]:  # (item, channel) 去重
    unique: dict[tuple[str, str], Candidate] = {}  # 收集
    for candidate in rows:  # 每行
        key = (candidate.item_id, candidate.channel)  # 键
        previous = unique.get(key)  # 已有
        if previous is None or (candidate.rank, -candidate.score) < (previous.rank, -previous.score):  # 更优
            unique[key] = candidate  # 保留
    return unique  # 返回


def _item_selection_priority(rows: list[Candidate]) -> tuple[int, int, float, str]:  # 越小越优先
    channel_count = len({row.channel for row in rows})  # 多路覆盖
    best_rank = min(row.rank for row in rows)  # 最佳 rank
    max_score = max(row.score for row in rows)  # 最高分
    return (-channel_count, best_rank, -max_score, rows[0].item_id)  # 先覆盖后 rank


def _group_rows_by_item(rows: Iterable[Candidate]) -> dict[str, list[Candidate]]:  # item -> 通道行
    grouped: dict[str, list[Candidate]] = defaultdict(list)  # 收集
    for row in rows:  # 每通道行
        grouped[row.item_id].append(row)  # 追加
    return grouped  # 返回


def select_union_items(rows: Iterable[Candidate], top_k_items_per_user: int) -> set[str]:  # 选出 item 集合
    if top_k_items_per_user < 1:  # 非法 K
        raise ValueError("top_k_items_per_user must be >= 1")  # 报错
    by_item = _group_rows_by_item(rows)  # 分组
    ranked_items = sorted(by_item.items(), key=lambda pair: _item_selection_priority(pair[1]))  # 多路优先
    return {item_id for item_id, _item_rows in ranked_items[:top_k_items_per_user]}  # Top-K item


def union_candidates(  # 保留选中 item 的全部通道证据行
    candidates: Iterable[Candidate],
    top_k_items_per_user: int,
    *,
    source_timestamp: str | None = None,
    feature_version: str = UNION_SCHEMA_VERSION,
) -> list[Candidate]:
    """De-duplicate channel rows, cap unique items per user, prefer multi-channel coverage."""
    _ = source_timestamp, feature_version  # 元数据由 build_union_evidence 写出；此处保持 Candidate 契约
    if top_k_items_per_user < 1:  # 非法 K
        raise ValueError("top_k_items_per_user must be >= 1")  # 报错

    by_user: dict[str, list[Candidate]] = defaultdict(list)  # 按用户
    for candidate in candidates:  # 全部通道行
        by_user[candidate.user_id].append(candidate)  # 收集

    result: list[Candidate] = []  # 输出
    for user_id in sorted(by_user):  # 稳定用户序
        unique_rows = _dedupe_channel_rows(by_user[user_id])  # 去重
        rows = list(unique_rows.values())  # 行列表
        selected_items = select_union_items(rows, top_k_items_per_user)  # 选 item
        result.extend(  # 保留全部通道证据
            sorted(
                (row for row in rows if row.item_id in selected_items),
                key=lambda row: (row.user_id, row.item_id, row.channel, row.rank),
            )
        )
    return result  # 返回


def build_union_evidence(  # 宽表证据
    candidates: Iterable[Candidate],
    top_k_items_per_user: int,
    *,
    channels: Iterable[str] | None = None,
    source_timestamp: str | None = None,
    feature_version: str = UNION_SCHEMA_VERSION,
) -> list[dict[str, object]]:
    """Build one wide row per selected user-item with per-channel present/rank/score."""
    timestamp = source_timestamp or _default_source_timestamp()  # 时间戳
    materialized = list(candidates)  # 物化
    if channels is not None:  # 显式通道列表
        channel_names = tuple(dict.fromkeys(str(name).strip().lower() for name in channels))  # 稳定序
    else:  # 从候选推断
        channel_names = tuple(dict.fromkeys(row.channel for row in materialized))  # 出现顺序

    by_user: dict[str, list[Candidate]] = defaultdict(list)  # 按用户
    for candidate in materialized:  # 全部行
        by_user[candidate.user_id].append(candidate)  # 收集

    evidence_rows: list[dict[str, object]] = []  # 输出
    for user_id in sorted(by_user):  # 稳定序
        unique_rows = _dedupe_channel_rows(by_user[user_id])  # 去重
        rows = list(unique_rows.values())  # 行列表
        if not rows:  # 空用户
            continue  # 跳过
        split = rows[0].split  # 划分
        by_item = _group_rows_by_item(rows)  # 分组
        selected_items = select_union_items(rows, top_k_items_per_user)  # 选 item
        for item_id in sorted(selected_items):  # 稳定 item 序
            item_rows = by_item[item_id]  # 该 item 全部通道行
            per_channel = {row.channel: row for row in item_rows}  # 通道 -> 行
            summary = UnionEvidenceRow(  # 汇总
                user_id=user_id,
                item_id=item_id,
                split=split,
                channel_count=len(per_channel),
                best_channel_rank=min(row.rank for row in item_rows),
                max_channel_score=max(row.score for row in item_rows),
                source_timestamp=timestamp,
                feature_version=feature_version,
            )
            evidence_rows.append(summary.as_dict(channels=channel_names, evidence=per_channel))  # 宽行
    return evidence_rows  # 返回


def union_evidence_fieldnames(channels: Iterable[str]) -> list[str]:  # CSV 列序
    base = [  # 固定列
        "user_id",
        "item_id",
        "split",
        "channel_count",
        "best_channel_rank",
        "max_channel_score",
        "source_timestamp",
        "feature_version",
    ]
    names = list(base)  # 拷贝
    for channel in channels:  # 每通道三列
        names.extend([f"{channel}_present", f"{channel}_score", f"{channel}_rank"])
    return names  # 返回


def write_union_evidence_csv(  # 写证据 CSV
    rows: Iterable[dict[str, object]],
    output_path: str | Path,
    *,
    channels: Iterable[str],
) -> Path:
    path = Path(output_path)  # 规范化
    path.parent.mkdir(parents=True, exist_ok=True)  # 建目录
    fieldnames = union_evidence_fieldnames(channels)  # 列
    with path.open("w", newline="", encoding="utf-8") as handle:  # 打开
        writer = csv.DictWriter(handle, fieldnames=fieldnames)  # 写表头
        writer.writeheader()  # 头
        for row in rows:  # 每行
            writer.writerow({name: row.get(name, "") for name in fieldnames})  # 按列写
    return path  # 返回
