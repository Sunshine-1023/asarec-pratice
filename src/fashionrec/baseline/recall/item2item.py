"""Item-to-item similarity recall with multiple debiased variants."""  # 商品相似度召回

from __future__ import annotations  # 延迟注解

import math  # log / exp
from collections import defaultdict  # 聚合
from dataclasses import dataclass  # 序列
from pathlib import Path  # 路径
from typing import Literal  # 变体

import pandas as pd  # 表格

from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # ID
from fashionrec.baseline.data.time import week_window_start
from fashionrec.baseline.recall.window_scores import filter_interactions_as_of  # as-of 截断


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集
COOCCUR_WEEKS = 8  # 共现统计窗口（周）
TOP_SIM_K = 20  # 每个商品保留邻居数
SEED_ITEMS = 10  # 种子商品数
ITEM2ITEM_RECALL_TOP_K = 50  # 召回 Top-K
ITEM2ITEM_SCHEMA_VERSION = "hm.item2item.v2"  # 索引语义
SWING_ALPHA = 1.0  # Swing 平滑项
TIME_DECAY_HALF_LIFE_DAYS = 14.0  # 时间衰减半衰期（天）

Item2ItemSimilarityMode = Literal["raw_cooccur", "cosine_iuf", "time_decay", "sequential", "swing"]  # 实验变体
DEFAULT_SIMILARITY_MODE: Item2ItemSimilarityMode = "cosine_iuf"  # 默认：降低热门偏差
SIMILARITY_MODES: tuple[Item2ItemSimilarityMode, ...] = (  # 全部变体
    "raw_cooccur",
    "cosine_iuf",
    "time_decay",
    "sequential",
    "swing",
)


@dataclass(frozen=True)  # 用户购买序列
class _UserSequence:  # 去重保序
    user_id: str  # 用户
    items: tuple[str, ...]  # 商品
    dates: tuple[pd.Timestamp, ...]  # 首次购买日


def _load_windowed_interactions(  # 读取窗口内交互
    inter_paths: list[str | Path],  # 路径
    *,
    cooccur_weeks: int,  # 窗口周数
    as_of: pd.Timestamp | str | None = None,  # as-of 预测日
) -> pd.DataFrame:  # 交互表
    frames: list[pd.DataFrame] = []  # 收集
    for path in inter_paths:  # 每个文件
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=["user_id:token", "item_id:token", "timestamp:float"],
            dtype={"user_id:token": "string", "item_id:token": "string"},
        )
        df["user_id:token"] = df["user_id:token"].map(canonical_user_id)  # 用户
        df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 商品
        df["date"] = pd.to_datetime(df["timestamp:float"], unit="s").dt.normalize()  # 自然日
        frames.append(df[["user_id:token", "item_id:token", "date", "timestamp:float"]])  # 保留列
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()  # 合并
    if merged.empty:  # 空
        return merged  # 返回
    merged = filter_interactions_as_of(merged, as_of)  # as-of 截断
    if merged.empty:  # 截断后空
        return merged  # 返回
    cutoff = week_window_start(merged["date"].max(), cooccur_weeks)  # 窗口起
    merged = merged[merged["date"] >= cutoff]  # 最近 N 周
    return merged.sort_values(["user_id:token", "timestamp:float"])  # 稳定序


def _extract_user_sequences(frame: pd.DataFrame) -> list[_UserSequence]:  # 用户去重序列
    sequences: list[_UserSequence] = []  # 收集
    for user_id, group in frame.groupby("user_id:token", sort=False):  # 每用户
        items: list[str] = []  # 商品
        dates: list[pd.Timestamp] = []  # 日期
        seen: set[str] = set()  # 去重
        for _ts, item_id, day in zip(group["timestamp:float"], group["item_id:token"], group["date"]):  # 行
            item_id = str(item_id)  # 规范
            if item_id in seen:  # 重复
                continue  # 跳过
            seen.add(item_id)  # 标记
            items.append(item_id)  # 追加
            dates.append(pd.Timestamp(day).normalize())  # 日期
        if items:  # 非空
            sequences.append(_UserSequence(user_id=str(user_id), items=tuple(items), dates=tuple(dates)))  # 记录
    return sequences  # 返回


def _decay_weight(day: pd.Timestamp, *, anchor: pd.Timestamp, half_life_days: float) -> float:  # 时间衰减权
    age_days = max(0.0, float((anchor - pd.Timestamp(day).normalize()).days))  # 距锚点天数
    if half_life_days <= 0.0:  # 无衰减
        return 1.0  # 常数
    return math.exp(-math.log(2.0) * age_days / half_life_days)  # 半衰期


def _accumulate_pair_counts(  # 共购对计数（含 IUF 权）
    sequences: list[_UserSequence],
    *,
    weighted: bool,
) -> tuple[defaultdict[str, defaultdict[str, float]], defaultdict[str, float], dict[tuple[str, str], set[str]]]:  # 共现/度/用户对
    cooccur: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))  # A->B
    strength: defaultdict[str, float] = defaultdict(float)  # 商品度
    pair_users: dict[tuple[str, str], set[str]] = defaultdict(set)  # 共现用户对
    user_items: dict[str, set[str]] = {}  # 用户商品集
    for seq in sequences:  # 每用户
        items = list(seq.items)  # 序列
        user_items[seq.user_id] = set(items)  # 记录
        weight = 1.0 / math.log(1.0 + len(items)) if weighted else 1.0  # IUF
        for item in items:  # 度
            strength[item] += weight  # 累加
        for i, item_a in enumerate(items):  # 全对
            for item_b in items[i + 1 :]:  # 无序对
                cooccur[item_a][item_b] += weight  # A-B
                cooccur[item_b][item_a] += weight  # B-A
                pair_users[(item_a, item_b)].add(seq.user_id)  # 用户
                pair_users[(item_b, item_a)].add(seq.user_id)  # 对称
    return cooccur, strength, pair_users  # 返回


def _accumulate_sequential(  # 相邻转移计数
    sequences: list[_UserSequence],
) -> defaultdict[str, defaultdict[str, float]]:  # A->B 有向
    transitions: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))  # 转移
    for seq in sequences:  # 每用户
        items = seq.items  # 序列
        for idx in range(len(items) - 1):  # 相邻
            transitions[items[idx]][items[idx + 1]] += 1.0  # 有向 +1
    return transitions  # 返回


def _accumulate_time_decay_pairs(  # 时间衰减共购
    sequences: list[_UserSequence],
    *,
    anchor: pd.Timestamp,
    half_life_days: float,
) -> tuple[defaultdict[str, defaultdict[str, float]], defaultdict[str, defaultdict[str, int]]]:  # 加权 + 原始计数
    cooccur: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))  # 共现
    pair_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))  # 共现次数
    for seq in sequences:  # 每用户
        items = seq.items  # 商品
        dates = seq.dates  # 日期
        day_weights = [_decay_weight(day, anchor=anchor, half_life_days=half_life_days) for day in dates]  # 衰减
        for i, item_a in enumerate(items):  # 全对
            for j in range(i + 1, len(items)):  # B 在 A 之后或同 basket
                item_b = items[j]  # 邻居
                pair_weight = (day_weights[i] + day_weights[j]) / 2.0  # 平均衰减权
                cooccur[item_a][item_b] += pair_weight  # A-B
                cooccur[item_b][item_a] += pair_weight  # B-A
                pair_counts[item_a][item_b] += 1  # 计数
                pair_counts[item_b][item_a] += 1  # 对称
    return cooccur, pair_counts  # 返回


def _cosine_from_cooccur(  # 余弦/IUF 归一化
    cooccur: defaultdict[str, defaultdict[str, float]],
    strength: defaultdict[str, float],
) -> dict[str, dict[str, float]]:  # 相似度
    index: dict[str, dict[str, float]] = {}  # 输出
    for item_a, neighbors in cooccur.items():  # 每个 A
        denom_a = strength.get(item_a, 0.0)  # 度
        if denom_a <= 0.0:  # 无边
            continue  # 跳过
        scored: dict[str, float] = {}  # 邻居分
        for item_b, weight in neighbors.items():  # 每个 B
            denom_b = strength.get(item_b, 0.0)  # B 度
            if denom_b <= 0.0:  # 无效
                continue  # 跳过
            scored[item_b] = weight / math.sqrt(denom_a * denom_b)  # 余弦
        if scored:  # 非空
            index[item_a] = scored  # 写入
    return index  # 返回


def _swing_from_pairs(  # Swing 分数
    pair_users: dict[tuple[str, str], set[str]],
    user_items: dict[str, set[str]],
    *,
    alpha: float,
) -> dict[str, dict[str, float]]:  # 相似度
    scores: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))  # 收集
    for (item_a, item_b), users in pair_users.items():  # 每对
        if item_a >= item_b:  # 只算一次无序对
            continue  # 跳过重复
        user_list = sorted(users)  # 稳定
        swing = 0.0  # 分数
        for i, user_u in enumerate(user_list):  # u
            items_u = user_items.get(user_u, set())  # I_u
            for user_v in user_list[i + 1 :]:  # v
                overlap = len(items_u & user_items.get(user_v, set()))  # |I_u ∩ I_v|
                swing += 1.0 / (alpha + overlap)  # 贡献
        if swing > 0.0:  # 有效
            scores[item_a][item_b] = swing  # 写入
            scores[item_b][item_a] = swing  # 对称
    return {item: dict(neighbors) for item, neighbors in scores.items()}  # 普通 dict


def _top_neighbors(  # 截断 Top-K
    neighbors: dict[str, float],
    *,
    top_sim_k: int,
    min_cooccur: float,
) -> dict[str, float]:  # Top 邻居
    filtered = {item: score for item, score in neighbors.items() if score >= min_cooccur and item}  # 过滤
    ranked = sorted(filtered.items(), key=lambda pair: (-pair[1], pair[0]))[:top_sim_k]  # 排序截断
    return dict(ranked)  # 返回


def _finalize_index(  # 统一 Top-K 截断
    raw_index: dict[str, dict[str, float]],
    *,
    top_sim_k: int,
    min_cooccur: float,
    pair_counts: dict[str, dict[str, int]] | None = None,  # 原始共现次数（加权模式）
) -> dict[str, dict[str, float]]:  # 最终索引
    index: dict[str, dict[str, float]] = {}  # 输出
    for item_a, neighbors in raw_index.items():  # 每个种子
        filtered_neighbors: dict[str, float] = {}  # 阈值过滤
        for item_b, score in neighbors.items():  # 每个邻居
            if pair_counts is not None:  # 加权模式：按原始共现次数过滤
                if pair_counts.get(item_a, {}).get(item_b, 0) < int(min_cooccur):  # 未达最小共现
                    continue  # 跳过
            elif score < min_cooccur:  # 计数模式：分数即次数
                continue  # 跳过
            filtered_neighbors[item_b] = score  # 保留
        top = _top_neighbors(filtered_neighbors, top_sim_k=top_sim_k, min_cooccur=0.0)  # Top-K
        if top:  # 非空
            index[item_a] = top  # 写入
    return index  # 返回


def build_item2item_index(  # 构建 Item2Item 索引
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_INTER_PATH,  # 交互
    cooccur_weeks: int = COOCCUR_WEEKS,  # 窗口
    top_sim_k: int = TOP_SIM_K,  # 邻居数
    min_cooccur: int = 1,  # 最小共现（或最小加权共现）
    similarity_mode: Item2ItemSimilarityMode = DEFAULT_SIMILARITY_MODE,  # 变体
    as_of: pd.Timestamp | str | None = None,  # as-of 预测日
    swing_alpha: float = SWING_ALPHA,  # Swing 平滑
    time_decay_half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS,  # 衰减半衰期
) -> dict[str, dict[str, float]]:  # item -> neighbors
    """Build item similarity index; default cosine/IUF reduces popularity bias."""  # 默认 cosine/IUF 降热门偏差
    if similarity_mode not in SIMILARITY_MODES:  # 非法变体
        raise ValueError(f"unsupported similarity_mode={similarity_mode!r}; expected one of {SIMILARITY_MODES}")  # 报错
    if isinstance(inter_paths, (str, Path)):  # 单路径
        paths = [inter_paths]  # 包装
    else:  # 多路径
        paths = list(inter_paths)  # 列表
    if not paths:  # 空
        paths = [DEFAULT_INTER_PATH]  # 默认

    frame = _load_windowed_interactions(paths, cooccur_weeks=cooccur_weeks, as_of=as_of)  # 读窗口
    if frame.empty:  # 无数据
        return {}  # 空索引

    sequences = _extract_user_sequences(frame)  # 用户序列
    if not sequences:  # 无序列
        return {}  # 空

    min_score = float(min_cooccur)  # 阈值
    if similarity_mode == "raw_cooccur":  # 基线：原始共现
        cooccur, _strength, _pairs = _accumulate_pair_counts(sequences, weighted=False)  # 计数
        raw = {item_a: dict(neighbors) for item_a, neighbors in cooccur.items()}  # 转 dict
        return _finalize_index(raw, top_sim_k=top_sim_k, min_cooccur=min_score)  # 截断

    if similarity_mode == "sequential":  # 有向相邻转移
        transitions = _accumulate_sequential(sequences)  # 转移
        raw = {item_a: dict(neighbors) for item_a, neighbors in transitions.items()}  # 转 dict
        return _finalize_index(raw, top_sim_k=top_sim_k, min_cooccur=min_score)  # 截断

    if similarity_mode == "time_decay":  # 时间衰减共购
        anchor = pd.Timestamp(frame["date"].max()).normalize()  # 锚点
        cooccur, pair_counts = _accumulate_time_decay_pairs(sequences, anchor=anchor, half_life_days=time_decay_half_life_days)  # 衰减
        raw = {item_a: dict(neighbors) for item_a, neighbors in cooccur.items()}  # 转 dict
        counts = {item_a: dict(neighbors) for item_a, neighbors in pair_counts.items()}  # 转 dict
        return _finalize_index(raw, top_sim_k=top_sim_k, min_cooccur=min_score, pair_counts=counts)  # 截断

    cooccur, strength, pair_users = _accumulate_pair_counts(sequences, weighted=(similarity_mode == "cosine_iuf"))  # 共现
    user_items = {seq.user_id: set(seq.items) for seq in sequences}  # 用户商品

    if similarity_mode == "cosine_iuf":  # 余弦/IUF
        cosine_counts = {  # 原始共现用户数
            item_a: {item_b: len(pair_users.get((item_a, item_b), set())) for item_b in neighbors}
            for item_a, neighbors in cooccur.items()
        }
        raw = _cosine_from_cooccur(cooccur, strength)  # 归一化
        return _finalize_index(raw, top_sim_k=top_sim_k, min_cooccur=min_score, pair_counts=cosine_counts)  # 截断

    if similarity_mode == "swing":  # Swing
        raw = _swing_from_pairs(pair_users, user_items, alpha=swing_alpha)  # Swing
        swing_counts = {item_a: {item_b: len(pair_users.get((item_a, item_b), pair_users.get((item_b, item_a), set()))) for item_b in neighbors} for item_a, neighbors in raw.items()}  # 共现用户数
        return _finalize_index(raw, top_sim_k=top_sim_k, min_cooccur=min_score, pair_counts=swing_counts)  # 截断

    raise ValueError(f"unsupported similarity_mode={similarity_mode!r}")  # 兜底


def recall_item2item(  # 召回 Top-K
    user_history: list[str] | set[str],  # 历史
    item2item_index: dict[str, dict[str, float]],  # 索引
    seed_items: int = SEED_ITEMS,  # 种子数
    top_k: int = ITEM2ITEM_RECALL_TOP_K,  # K
) -> list[tuple[str, float]]:  # 候选
    """Recall by aggregating neighbors of the user's recent purchases."""  # 聚合最近购买邻居
    history_list = [canonical_item_id(x) for x in user_history]  # 规范
    if not history_list:  # 无历史
        return []  # 空
    seeds = history_list[-seed_items:]  # 最近种子
    scores: defaultdict[str, float] = defaultdict(float)  # 聚合
    for item_a in seeds:  # 每个种子
        for item_b, sim_score in item2item_index.get(item_a, {}).items():  # 邻居
            scores[item_b] += sim_score  # 累加
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))  # 降序
    return ranked[:top_k]  # Top-K


if __name__ == "__main__":  # 脚本入口
    index = build_item2item_index(similarity_mode=DEFAULT_SIMILARITY_MODE)  # 构建
    sample_history = list(index.keys())[:3]  # 示例历史
    sample = recall_item2item(sample_history, index, top_k=10)  # 召回
    print(f"Item2Item index size: {len(index):,} (mode={DEFAULT_SIMILARITY_MODE}, schema={ITEM2ITEM_SCHEMA_VERSION})")  # 规模
    print("Top-10 sample:", sample)  # 样例
