"""Popular-item recall channel."""  # 热门商品召回通道模块

from __future__ import annotations  # 启用延迟注解评估

from collections import Counter  # 导入计数器用于统计频次
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas 用于读取交互数据


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集交互文件路径


def build_popular_index(*inter_paths: str | Path) -> list[tuple[str, float]]:  # 构建全局热门商品索引
    """Build a global popularity index from one or more interaction files."""  # 从一个或多个交互文件构建热门度索引
    if not inter_paths:  # 若未指定交互文件路径
        inter_paths = (DEFAULT_INTER_PATH,)  # 使用默认训练集路径
    item_ids: list[str] = []  # 初始化商品 ID 列表
    for path in inter_paths:  # 遍历每个交互文件
        df = pd.read_csv(path, sep="\t", usecols=["item_id:token"])  # 读取商品 ID 列
        item_ids.extend(df["item_id:token"].astype(str).tolist())  # 追加当前文件中的商品 ID
    counts = Counter(item_ids)  # 统计每个商品出现次数
    return [(item_id, float(count)) for item_id, count in counts.most_common()]  # 按频次降序返回商品及分数


def recall_popular(  # 基于热门度召回商品
    popular_index: list[tuple[str, float]],  # 热门商品索引列表
    user_history: set[str] | None = None,  # 用户历史交互商品集合
    top_k: int = 100,  # 召回数量上限
) -> list[tuple[str, float]]:  # 返回商品 ID 与分数列表
    """Recall top-k popular items excluding the user's history."""  # 召回 Top-K 热门商品并排除用户历史
    history = {str(x) for x in user_history} if user_history else set()  # 规范化用户历史为字符串集合
    results: list[tuple[str, float]] = []  # 初始化召回结果列表
    for item_id, score in popular_index:  # 按热门度顺序遍历商品
        if item_id in history:  # 若商品已在用户历史中
            continue  # 跳过该商品
        results.append((item_id, score))  # 加入召回结果
        if len(results) >= top_k:  # 若已凑够 Top-K
            break  # 停止遍历
    return results  # 返回召回列表


if __name__ == "__main__":  # 脚本直接运行入口
    index = build_popular_index()  # 构建热门索引
    sample = recall_popular(index, user_history=set(), top_k=10)  # 召回 Top-10 示例
    print(f"Popular index size: {len(index):,}")  # 打印索引规模
    print("Top-10 sample:", sample)  # 打印 Top-10 示例结果
