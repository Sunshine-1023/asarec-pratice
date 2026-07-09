"""Popular-item recall channel with time-decayed popularity."""  # 带时间衰减的热门商品召回通道

from __future__ import annotations  # 启用延迟注解评估

from collections import Counter  # 导入计数器用于统计频次
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas 用于读取交互数据


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集交互文件路径
POPULAR_RECALL_TOP_K = 50  # 全局热门召回 Top-K
WINDOW_WEEKS = (1, 2, 4, 8)  # 时间衰减窗口（周）
WINDOW_WEIGHTS = (0.5, 0.3, 0.15, 0.05)  # 各窗口融合权重


def _week_window_start(max_date: pd.Timestamp, weeks: int) -> pd.Timestamp:  # 计算含 max_date 的 N 周窗口起始日
    max_day = pd.Timestamp(max_date).normalize()  # 归一化到自然日
    return max_day - pd.Timedelta(days=weeks * 7 - 1)  # 含首尾共 weeks*7 天


def _load_interactions(*inter_paths: str | Path) -> pd.DataFrame:  # 读取交互文件并合并
    if not inter_paths:  # 若未指定交互文件路径
        inter_paths = (DEFAULT_INTER_PATH,)  # 使用默认训练集路径

    frames: list[pd.DataFrame] = []  # 存放各文件 DataFrame
    for path in inter_paths:  # 遍历每个交互文件
        df = pd.read_csv(  # 读取商品 ID 与时间戳
            path,  # 交互文件路径
            sep="\t",  # 制表符分隔
            usecols=["item_id:token", "timestamp:float"],  # 仅读取商品 ID 与时间戳列
        )  # 结束 read_csv 调用
        df["item_id:token"] = df["item_id:token"].astype(str)  # 商品 ID 转字符串
        df["date"] = pd.to_datetime(df["timestamp:float"], unit="s").dt.normalize()  # 时间戳转自然日
        frames.append(df[["item_id:token", "date"]])  # 只保留所需列

    return pd.concat(frames, ignore_index=True)  # 合并全部交互


def build_popular_index(  # 构建时间衰减热门商品索引
    *inter_paths: str | Path,  # 一个或多个交互文件路径
    window_weeks: tuple[int, ...] = WINDOW_WEEKS,  # 时间窗口周数列表
    window_weights: tuple[float, ...] = WINDOW_WEIGHTS,  # 各窗口融合权重
) -> list[tuple[str, float]]:  # 返回 (商品ID, 热度分) 排序列表
    """  # 函数文档字符串开始
    Build a time-decayed popularity index from one or more interaction files.  # 从一个或多个交互文件构建时间衰减热门度索引

    Instead of all-time global counts, blend rolling-window heat:  # 融合滚动窗口热度而非全量计数
      hot_score = 0.5 * heat_1w + 0.3 * heat_2w + 0.15 * heat_4w + 0.05 * heat_8w  # 多窗口加权热度公式
    """  # 从交互数据构建多窗口时间衰减热门度索引
    if len(window_weeks) != len(window_weights):  # 校验窗口与权重长度
        raise ValueError("window_weeks and window_weights must have the same length")  # 长度不一致则报错

    df = _load_interactions(*inter_paths)  # 加载交互数据
    if df.empty:  # 若无交互记录
        return []  # 返回空索引

    max_date = df["date"].max()  # 数据最大日期
    window_counts: list[Counter[str]] = []  # 各时间窗口的商品计数
    for weeks in window_weeks:  # 遍历每个时间窗口
        start_date = _week_window_start(max_date, weeks)  # 计算窗口起始日
        window_df = df[df["date"] >= start_date]  # 保留窗口内交互
        window_counts.append(Counter(window_df["item_id:token"].tolist()))  # 统计窗口内热度

    all_items = set().union(*window_counts)  # 合并全部出现过的商品
    scores: dict[str, float] = {}  # 商品到衰减热度的映射
    for item_id in all_items:  # 遍历每个商品
        score = 0.0  # 初始化融合分数
        for weight, counts in zip(window_weights, window_counts):  # 按权重融合各窗口热度
            score += weight * float(counts.get(item_id, 0))  # 累加加权热度
        scores[item_id] = score  # 保存商品分数

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # 按分数降序、ID 升序排序
    return ranked  # 返回热门商品索引


def recall_popular(  # 基于热门度召回商品
    popular_index: list[tuple[str, float]],  # 热门商品索引列表
    user_history: set[str] | None = None,  # 保留以兼容调用方，不再用于过滤已购商品
    top_k: int = POPULAR_RECALL_TOP_K,  # 召回数量上限
) -> list[tuple[str, float]]:  # 返回商品 ID 与分数列表
    """Recall top-k popular items (may include repeat purchases)."""  # 召回 Top-K 热门商品（允许复购同款）
    return popular_index[:top_k]  # 直接取热门索引前 Top-K


if __name__ == "__main__":  # 脚本直接运行入口
    index = build_popular_index()  # 构建时间衰减热门索引
    sample = recall_popular(index, user_history=set(), top_k=10)  # 召回 Top-10 示例
    print(f"Popular index size: {len(index):,}")  # 打印索引规模
    print("Top-10 sample:", sample)  # 打印 Top-10 示例结果
