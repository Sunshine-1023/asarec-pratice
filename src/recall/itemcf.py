"""ItemCF recall channel."""  # 基于物品的协同过滤召回通道模块

from __future__ import annotations  # 启用延迟注解评估

from collections import defaultdict  # 导入默认字典用于聚合统计
from math import sqrt  # 导入平方根用于相似度归一化
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas 用于读取交互数据


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集交互文件路径


def build_itemcf_index(  # 构建物品协同过滤相似度索引
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_INTER_PATH,  # 交互文件路径
    min_cooccur: int = 2,  # 最小共现次数阈值
    top_sim_k: int = 100,  # 每个物品保留的最相似邻居数
    max_user_items: int = 50,  # 每个用户最多保留的交互商品数
) -> dict[str, dict[str, float]]:  # 返回物品到相似物品的映射
    """Build item-to-item similarity index from one or more interaction files."""  # 从交互数据构建物品相似度索引
    if isinstance(inter_paths, (str, Path)):  # 若传入单个路径
        paths = [inter_paths]  # 包装成列表
    else:  # 若传入多个路径
        paths = list(inter_paths)  # 转为列表
    if not paths:  # 若路径列表为空
        paths = [DEFAULT_INTER_PATH]  # 使用默认训练集路径

    frames = []  # 初始化 DataFrame 列表
    for path in paths:  # 遍历每个交互文件
        frames.append(  # 追加读取结果
            pd.read_csv(  # 读取 CSV 交互文件
                path,  # 文件路径
                sep="\t",  # 制表符分隔
                usecols=["user_id:token", "item_id:token", "timestamp:float"],  # 仅读取必要列
            )  # 结束 read_csv 调用
        )  # 结束 append
    df = pd.concat(frames, ignore_index=True)  # 合并所有交互数据
    df = df.sort_values(["user_id:token", "timestamp:float"])  # 按用户和时间排序

    user_items: list[list[str]] = []  # 初始化用户交互序列列表
    for _, group in df.groupby("user_id:token", sort=False):  # 按用户分组
        items = group["item_id:token"].astype(str).tolist()  # 提取该用户的商品序列
        if len(items) > max_user_items:  # 若序列过长
            items = items[-max_user_items:]  # 仅保留最近 max_user_items 个商品

        seen: set[str] = set()  # 记录已出现商品用于去重
        unique_items: list[str] = []  # 去重后的商品序列
        for item in items:  # 遍历用户交互商品
            if item in seen:  # 若商品已出现
                continue  # 跳过重复项
            seen.add(item)  # 标记商品已出现
            unique_items.append(item)  # 追加到去重序列
        if unique_items:  # 若去重后仍有商品
            user_items.append(unique_items)  # 保存该用户序列

    item_count: defaultdict[str, int] = defaultdict(int)  # 统计每个商品出现次数
    cooccur: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))  # 统计商品共现次数

    for items in user_items:  # 遍历每个用户序列
        for item in items:  # 统计单品频次
            item_count[item] += 1  # 商品计数加一
        for i, item_i in enumerate(items):  # 遍历序列中每个商品
            for item_j in items[i + 1 :]:  # 与后续商品组成共现对
                cooccur[item_i][item_j] += 1  # 正向共现加一
                cooccur[item_j][item_i] += 1  # 反向共现加一

    sim_index: dict[str, dict[str, float]] = {}  # 初始化相似度索引
    for item_i, neighbors in cooccur.items():  # 遍历每个商品及其邻居
        scored_neighbors: dict[str, float] = {}  # 初始化邻居得分字典
        for item_j, cij in neighbors.items():  # 遍历每个邻居及共现次数
            if cij < min_cooccur:  # 若共现次数低于阈值
                continue  # 跳过该邻居
            score = cij / sqrt(item_count[item_i] * item_count[item_j])  # 按余弦风格归一化计算相似度
            scored_neighbors[item_j] = score  # 记录邻居相似度
        if scored_neighbors:  # 若存在有效邻居
            top_neighbors = sorted(  # 按相似度降序排序
                scored_neighbors.items(), key=lambda x: x[1], reverse=True  # 以相似度分数为键
            )[:top_sim_k]  # 截取 Top-K 邻居
            sim_index[item_i] = dict(top_neighbors)  # 写入相似度索引
    return sim_index  # 返回物品相似度索引


def recall_itemcf(  # 基于 ItemCF 索引召回商品
    user_history: list[str] | set[str],  # 用户历史交互商品
    itemcf_index: dict[str, dict[str, float]],  # 物品相似度索引
    top_k: int = 100,  # 召回数量上限
) -> list[tuple[str, float]]:  # 返回商品 ID 与聚合分数列表
    """Recall top-k items by aggregating similar neighbors of history items."""  # 聚合历史商品相似邻居进行召回
    history_list = [str(x) for x in user_history]  # 规范化历史商品为字符串列表
    history_set = set(history_list)  # 转为集合便于过滤
    scores: defaultdict[str, float] = defaultdict(float)  # 初始化候选商品聚合得分

    for item in history_list:  # 遍历用户历史中的每个商品
        for neighbor, sim in itemcf_index.get(item, {}).items():  # 遍历该商品的相似邻居
            if neighbor in history_set:  # 若邻居已在历史中
                continue  # 跳过该邻居
            scores[neighbor] += sim  # 累加相似度得分

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 按得分降序排序
    return ranked[:top_k]  # 返回 Top-K 召回结果


if __name__ == "__main__":  # 脚本直接运行入口
    index = build_itemcf_index()  # 构建 ItemCF 索引
    sample_history = ["0706016001", "0685814001", "0751471001"]  # 示例用户历史
    sample = recall_itemcf(sample_history, index, top_k=10)  # 召回 Top-10 示例
    print(f"ItemCF index size: {len(index):,}")  # 打印索引规模
    print("Top-10 sample:", sample)  # 打印 Top-10 示例结果
