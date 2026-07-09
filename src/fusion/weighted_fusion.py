"""Weighted fusion for multi-channel candidate lists."""  # 多通道候选列表的加权融合模块

from __future__ import annotations  # 启用延迟注解评估

import csv  # 导入 CSV 读写模块
from collections import defaultdict  # 导入带默认值的字典
from pathlib import Path  # 导入路径处理类
from typing import Literal  # 导入字面量类型

import pandas as pd  # 导入 pandas 数据分析库


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
    """Return per-user fusion weights; sequence channel key matches sasrec or sasrecf."""
    weights_table = activity_weights or ACTIVITY_WEIGHTS  # 使用自定义或默认权重表
    tier = classify_activity_tier(history_len)  # 判定活跃度分层
    template = weights_table[tier]  # 取对应权重模板
    return {  # 组装通道权重字典
        sequence_channel: template["sequence"],  # 序列模型通道权重
        "popular": template["popular"],  # 热门召回通道权重
        "category_popular": template["category_popular"],  # 类别热门通道权重
        "item2item": template["item2item"],  # 商品共现通道权重
    }  # 权重字典结束


def infer_sequence_channel(recall_csv: str | Path) -> str:  # 从召回文件名推断序列通道名
    name = Path(recall_csv).stem.lower()  # 文件名小写
    if name.startswith("sasrecf"):  # SASRecF 召回
        return "sasrecf"  # 返回 sasrecf 通道名
    return "sasrec"  # 默认 SASRec


def build_user_history(*inter_paths: str | Path) -> dict[str, list[str]]:  # 从交互文件构建用户历史
    """Build per-user ordered history from one or more .inter files."""  # 从一个或多个 .inter 文件构建按时间排序的用户历史
    frames = []  # 初始化 DataFrame 列表
    for path in inter_paths:  # 遍历每个交互文件路径
        df = pd.read_csv(path, sep="\t", usecols=["user_id:token", "item_id:token", "timestamp:float"])  # 读取用户、物品与时间戳列
        frames.append(df)  # 将当前 DataFrame 加入列表
    merged = pd.concat(frames, ignore_index=True)  # 合并所有交互记录
    merged = merged.sort_values(["user_id:token", "timestamp:float"])  # 按用户与时间戳排序
    history = (  # 构建用户到物品序列的映射
        merged.groupby("user_id:token")["item_id:token"]  # 按用户分组并取物品列
        .apply(lambda s: [str(x) for x in s.tolist()])  # 将每组物品转为字符串列表
        .to_dict()  # 转为字典
    )  # 结束历史映射构建
    return history  # 返回用户历史字典


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
            uid = str(row[user_col])  # 读取并规范化用户 ID
            iid = str(row[item_col])  # 读取并规范化物品 ID
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
    """Weighted rank fusion: weight * (1 / (rank + 1)), summed over channels."""
    history = {str(x) for x in user_history} if exclude_seen else set()  # 按需构建历史过滤集合
    merged_scores: dict[str, float] = defaultdict(float)  # 初始化融合得分字典

    for channel, candidates in channel_candidates.items():  # 遍历每个召回通道
        w = channel_weights.get(channel, 0.0)  # 获取当前通道权重
        if w <= 0:  # 若权重非正
            continue  # 跳过该通道
        for rank, (item_id, _) in enumerate(candidates):  # 按排名遍历通道候选
            item_id = str(item_id)  # 规范化物品 ID
            if item_id in history:  # 排除已购（可选）
                continue
            merged_scores[item_id] += w * (1.0 / (rank + 1))  # 累加加权倒数排名得分

    ranked = sorted(merged_scores.items(), key=lambda x: (-x[1], x[0]))  # 得分降序、ID 升序去重
    return ranked[:top_k]  # 返回 Top-K 融合结果（同 item 只保留一次）
