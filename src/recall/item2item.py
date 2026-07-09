"""Item-to-item co-occurrence recall channel."""  # 商品共现召回通道

from __future__ import annotations  # 启用延迟注解评估

from collections import defaultdict  # 导入默认字典
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集交互文件路径
COOCCUR_WEEKS = 8  # 共现统计窗口（周，4~8 周取上限）
TOP_SIM_K = 20  # 每个商品保留的相似邻居数
SEED_ITEMS = 10  # 用用户最近 N 个购买商品做种子
ITEM2ITEM_RECALL_TOP_K = 50  # 召回输出 Top-K


def _week_window_start(max_date: pd.Timestamp, weeks: int) -> pd.Timestamp:  # 计算含 max_date 的 N 周窗口起始日
    max_day = pd.Timestamp(max_date).normalize()  # 归一化到自然日
    return max_day - pd.Timedelta(days=weeks * 7 - 1)  # 含首尾共 weeks*7 天


def _load_windowed_interactions(  # 读取并截取最近 N 周交互
    inter_paths: list[str | Path],
    cooccur_weeks: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []  # 各文件 DataFrame
    for path in inter_paths:  # 遍历交互文件
        df = pd.read_csv(  # 读取用户、商品、时间戳
            path,
            sep="\t",
            usecols=["user_id:token", "item_id:token", "timestamp:float"],
        )
        df["item_id:token"] = df["item_id:token"].astype(str)  # 商品 ID 转字符串
        df["date"] = pd.to_datetime(df["timestamp:float"], unit="s").dt.normalize()  # 转自然日
        frames.append(df)  # 追加

    merged = pd.concat(frames, ignore_index=True)  # 合并
    if merged.empty:  # 空数据
        return merged

    cutoff = _week_window_start(merged["date"].max(), cooccur_weeks)  # 窗口起始日
    merged = merged[merged["date"] >= cutoff]  # 只保留最近 cooccur_weeks 周
    return merged.sort_values(["user_id:token", "timestamp:float"])  # 按用户与时间排序


def build_item2item_index(  # 构建 A→B 共现索引
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_INTER_PATH,
    cooccur_weeks: int = COOCCUR_WEEKS,
    top_sim_k: int = TOP_SIM_K,
    min_cooccur: int = 1,
) -> dict[str, dict[str, float]]:
    """
    Build directed item co-occurrence index from recent-week interactions.

    For each user, pair items bought together; count A→B co-occurrence and keep Top-K neighbors per item.
    """
    if isinstance(inter_paths, (str, Path)):  # 单路径
        paths = [inter_paths]
    else:
        paths = list(inter_paths)
    if not paths:
        paths = [DEFAULT_INTER_PATH]

    df = _load_windowed_interactions(paths, cooccur_weeks=cooccur_weeks)  # 加载窗口内交互
    if df.empty:
        return {}

    cooccur: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))  # A→B 共现计数

    for _, group in df.groupby("user_id:token", sort=False):  # 按用户分组
        items = group["item_id:token"].tolist()  # 时间序商品列表
        seen: set[str] = set()  # 用户内去重
        unique_items: list[str] = []  # 去重后序列
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            unique_items.append(item)

        for item_a in unique_items:  # 种子商品 A
            for item_b in unique_items:  # 同用户其他商品 B
                if item_a == item_b:  # 跳过自身
                    continue
                cooccur[item_a][item_b] += 1  # 统计 A→B 共现

    index: dict[str, dict[str, float]] = {}  # 输出索引
    for item_a, neighbors in cooccur.items():  # 每个商品 A
        filtered = {item_b: float(cnt) for item_b, cnt in neighbors.items() if cnt >= min_cooccur}  # 过滤低频
        if not filtered:
            continue
        top_neighbors = sorted(filtered.items(), key=lambda x: (-x[1], x[0]))[:top_sim_k]  # Top-K 邻居
        index[item_a] = dict(top_neighbors)  # 写入索引
    return index


def recall_item2item(  # 基于最近购买商品召回相似商品
    user_history: list[str] | set[str],
    item2item_index: dict[str, dict[str, float]],
    seed_items: int = SEED_ITEMS,
    top_k: int = ITEM2ITEM_RECALL_TOP_K,
) -> list[tuple[str, float]]:
    """Recall by aggregating co-occurrence neighbors of the user's recent purchases."""
    history_list = [str(x) for x in user_history]  # 规范化历史
    if not history_list:
        return []

    history_set = set(history_list)  # 历史集合
    seeds = history_list[-seed_items:]  # 最近 seed_items 个商品作种子
    scores: defaultdict[str, float] = defaultdict(float)  # 候选聚合分

    for item_a in seeds:  # 遍历种子 A
        for item_b, cooccur_score in item2item_index.get(item_a, {}).items():  # A 的共现邻居 B
            if item_b in history_set:  # 排除已购买
                continue
            scores[item_b] += cooccur_score  # 累加共现分

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # 降序排序
    return ranked[:top_k]  # Top-K


if __name__ == "__main__":  # 脚本直接运行
    index = build_item2item_index()  # 构建索引
    sample_history = list(index.keys())[:3]  # 示例历史
    sample = recall_item2item(sample_history, index, top_k=10)  # 示例召回
    print(f"Item2Item index size: {len(index):,}")  # 索引规模
    print("Top-10 sample:", sample)  # 示例结果
