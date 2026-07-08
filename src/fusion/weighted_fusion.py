"""Weighted fusion for multi-channel candidate lists."""  # 多通道候选列表的加权融合模块

from __future__ import annotations  # 启用延迟注解评估

import csv  # 导入 CSV 读写模块
from collections import defaultdict  # 导入带默认值的字典
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas 数据分析库


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
    user_history: set[str],  # 用户已交互物品集合
    channel_candidates: dict[str, list[tuple[str, float]]],  # 各通道候选列表
    channel_weights: dict[str, float],  # 各通道权重
    top_k: int = 12,  # 最终返回的 Top-K 数量
) -> list[tuple[str, float]]:  # 返回融合后的 (物品, 分数) 列表
    """Weighted rank fusion: weight * (1 / (rank + 1)), summed over channels."""  # 加权排名融合：各通道按权重与倒数排名累加得分
    history = {str(x) for x in user_history}  # 将历史物品统一转为字符串集合
    merged_scores: dict[str, float] = defaultdict(float)  # 初始化融合得分字典

    for channel, candidates in channel_candidates.items():  # 遍历每个召回通道
        w = channel_weights.get(channel, 0.0)  # 获取当前通道权重
        if w <= 0:  # 若权重非正
            continue  # 跳过该通道
        for rank, (item_id, _) in enumerate(candidates):  # 按排名遍历通道候选
            item_id = str(item_id)  # 规范化物品 ID
            if item_id in history:  # 若物品已在用户历史中
                continue  # 跳过已交互物品
            merged_scores[item_id] += w * (1.0 / (rank + 1))  # 累加加权倒数排名得分

    ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)  # 按融合得分降序排序
    return ranked[:top_k]  # 返回 Top-K 融合结果
