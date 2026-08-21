"""Weighted fusion for multi-channel candidate lists."""  # 多通道候选列表的加权融合模块

from __future__ import annotations  # 启用延迟注解评估

import csv  # 导入 CSV 读写模块
from collections import defaultdict  # 导入带默认值的字典
from pathlib import Path  # 导入路径处理类
from typing import Literal  # 导入字面量类型

import pandas as pd  # 导入 pandas 数据分析库

from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # 统一 ID 契约

MAX_USER_HISTORY = 100  # 每用户保留的最大历史条数（与序列模型 MAX_ITEM_LIST_LENGTH 对齐）

ActivityTier = Literal["high", "medium", "low", "cold_start"]  # 用户活跃度分层

# 序列模型通道权重模板（sasrec / sasrecf 共用）
ACTIVITY_WEIGHTS: dict[ActivityTier, dict[str, float]] = {  # 各活跃度分层的通道权重模板
    "high": {"sequence": 0.60, "popular": 0.10, "category_popular": 0.10, "item2item": 0.20},  # 历史 >= 10
    "medium": {"sequence": 0.40, "popular": 0.15, "category_popular": 0.15, "item2item": 0.30},  # 历史 3~9
    "low": {"sequence": 0.15, "popular": 0.35, "category_popular": 0.25, "item2item": 0.25},  # 历史 1~2
    "cold_start": {"sequence": 0.00, "popular": 0.55, "category_popular": 0.30, "item2item": 0.15},  # 无历史
}  # 权重模板字典结束

def classify_activity_tier(history_len: int) -> ActivityTier:  # 按历史购买次数划分活跃度
    if history_len <= 0:  # 冷启动
        return "cold_start"  # 返回冷启动分层
    if history_len <= 2:  # 低活跃
        return "low"  # 返回低活跃分层
    if history_len <= 9:  # 中活跃
        return "medium"  # 返回中活跃分层
    return "high"  # 高活跃（>= 10）


def get_channel_weights_for_user(  # 按用户历史长度返回通道权重
    history_len: int,  # 用户历史购买次数
    sequence_channel: str = "sasrec",  # 序列模型通道名
    activity_weights: dict[ActivityTier, dict[str, float]] | None = None,  # 可选自定义分层权重
) -> dict[str, float]:  # 返回各通道权重字典
    """Return per-user fusion weights; sequence channel key matches sasrec or sasrecf."""  # 返回每用户融合权重，序列通道键与 sasrec 或 sasrecf 对应
    weights_table = activity_weights or ACTIVITY_WEIGHTS  # 使用自定义或默认权重表
    tier = classify_activity_tier(history_len)  # 判定活跃度分层
    template = weights_table[tier]  # 取对应权重模板
    weights = {  # 组装成熟通道权重字典
        sequence_channel: template["sequence"],  # 序列模型通道权重
        "popular": template["popular"],  # 热门召回通道权重
        "category_popular": template["category_popular"],  # 类别热门通道权重
        "item2item": template["item2item"],  # 商品共现通道权重
    }  # 成熟通道结束
    return weights


def infer_sequence_channel(recall_csv: str | Path) -> str:  # 从召回文件名推断序列通道名
    name = Path(recall_csv).stem.lower()  # 文件名小写
    if name.startswith("sasrecf"):  # SASRecF 召回
        return "sasrecf"  # 返回 sasrecf 通道名
    return "sasrec"  # 默认 SASRec


def build_user_history(  # 从交互文件构建每用户有序历史
    *inter_paths: str | Path,  # 一个或多个 .inter 文件路径
    max_user_history: int = MAX_USER_HISTORY,  # 每用户最多保留的历史条数
) -> dict[str, list[str]]:  # 返回用户 ID 到物品序列的映射
    """Build per-user ordered history from one or more .inter files."""  # 从一个或多个 .inter 文件构建每用户有序历史
    frames = []  # 初始化 DataFrame 列表
    for path in inter_paths:  # 遍历每个交互文件路径
        df = pd.read_csv(  # 读取用户、物品与时间戳列
            path,
            sep="\t",
            usecols=["user_id:token", "item_id:token", "timestamp:float"],
            dtype={"user_id:token": "string", "item_id:token": "string"},
        )
        df["user_id:token"] = df["user_id:token"].map(canonical_user_id)  # 统一用户 ID
        df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 统一商品 ID
        frames.append(df)  # 将当前 DataFrame 加入列表
    merged = pd.concat(frames, ignore_index=True)  # 合并所有交互记录
    merged = merged.sort_values(["user_id:token", "timestamp:float"])  # 按用户与时间戳排序
    history = (  # 构建用户到物品序列的映射
        merged.groupby("user_id:token")["item_id:token"]  # 按用户分组并取物品列
        .apply(lambda s: [canonical_item_id(x) for x in s.tolist()[-max_user_history:]])  # 每用户只保留最近 N 条
        .to_dict()  # 转为字典
    )  # 结束历史映射构建
    return history  # 返回用户历史字典


def normalize_item_id(item_id: str) -> str:  # 统一 item_id 格式（hm 与 hm_seq/RecBole 导出）
    """Return the shared ten-character H&M item ID representation."""  # 返回统一十位商品 ID
    return canonical_item_id(item_id)  # 保留兼容函数名，委托领域契约


def load_channel_recall_csv(  # 加载单通道召回 CSV 文件
    path: str | Path,  # 召回结果文件路径
    user_col: str = "user_id",  # 用户 ID 列名
    item_col: str = "item_id",  # 物品 ID 列名
    score_col: str = "score",  # 分数列名
    rank_col: str = "rank",  # 排名列名
) -> dict[str, list[tuple[str, float, int]]]:  # 返回用户到 (物品, 分数, 排名) 列表的映射
    """Load a channel recall csv as: {user_id: [(item_id, score, rank), ...]}."""  # 将渠道召回 CSV 加载为按用户分组的候选列表
    path = Path(path)  # 将路径转为 Path 对象
    if not path.exists():  # 若文件不存在
        return {}  # 返回空字典

    rows_by_user: dict[str, list[tuple[str, float, int]]] = defaultdict(list)  # 初始化按用户聚合的行列表
    with path.open("r", newline="", encoding="utf-8") as f:  # 以 UTF-8 打开 CSV 文件
        reader = csv.DictReader(f)  # 创建字典形式 CSV 读取器
        for row in reader:  # 逐行读取召回记录
            uid = canonical_user_id(row[user_col])  # 读取并规范化用户 ID
            iid = normalize_item_id(row[item_col])  # 读取并规范化物品 ID
            score = float(row.get(score_col, 0.0))  # 读取分数，缺失时默认为 0.0
            rank = int(row.get(rank_col, 999999))  # 读取排名，缺失时默认为 999999
            rows_by_user[uid].append((iid, score, rank))  # 追加当前用户的候选三元组

    for uid in rows_by_user:  # 遍历每个用户
        rows_by_user[uid].sort(key=lambda x: x[2])  # 按排名升序排序候选
    return dict(rows_by_user)  # 返回普通字典


def fuse_candidates(  # 对多通道候选进行加权融合
    user_id: str,  # 当前用户 ID
    user_history: set[str],  # 用户历史商品集合（exclude_seen=True 时用于过滤）
    channel_candidates: dict[str, list[tuple[str, float]]],  # 各通道候选列表
    channel_weights: dict[str, float],  # 各通道权重
    top_k: int = 12,  # 最终返回的 Top-K 数量
    exclude_seen: bool = False,  # 是否排除历史已购商品
) -> list[tuple[str, float]]:  # 返回融合后的 (物品, 分数) 列表
    """Compatibility facade; the ranking layer owns the RRF implementation."""
    from fashionrec.baseline.ranking.weighted_rrf import WeightedRRFRanker

    ranked = WeightedRRFRanker(channel_weights, exclude_seen=exclude_seen).rank(
        user_id=user_id,
        user_history=user_history,
        channel_candidates=channel_candidates,
        top_k=top_k,
    )
    return [(item.item_id, item.score) for item in ranked]
