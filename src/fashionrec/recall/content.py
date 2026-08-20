"""Content recall: hashed text TF-IDF + categorical overlap."""  # 内容召回

from __future__ import annotations  # 延迟注解

import math  # sqrt
import re  # 分词
from collections import Counter  # 词频
from dataclasses import dataclass, field  # 索引
from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.data.item_features import (  # 文本/类别清洗
    RECB_CATEGORY_COLUMNS,
    clean_category_token,
    clean_text,
    load_articles_table,
)
from fashionrec.domain.ids import canonical_item_id  # SKU
from fashionrec.recall.window_scores import build_window_popularity, filter_interactions_as_of  # 冷启动热度


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集
CONTENT_RECALL_TOP_K = 50  # 召回 Top-K
CONTENT_SEED_ITEMS = 10  # 种子数
CONTENT_SCHEMA_VERSION = "hm.content.v1"  # 索引语义
_WORD = re.compile(r"[a-z0-9]+")  # 简易分词


@dataclass(frozen=True)  # 内容索引
class ContentIndex:
    item_vectors: dict[str, dict[str, float]] = field(default_factory=dict)  # 稀疏 TF-IDF
    cold_start_ranked: list[tuple[str, float]] = field(default_factory=list)  # 无历史 fallback


def _tokenize_text(text: str) -> list[str]:  # 文本 -> token
    return _WORD.findall(text)  # 字母数字词


def _item_tokens(row: pd.Series) -> Counter[str]:  # 单 SKU token 计数
    tokens: Counter[str] = Counter()  # 收集
    for col in ("prod_name", "detail_desc"):  # 文本
        text = clean_text(row.get(col))  # 清洗
        for token in _tokenize_text(text):  # 分词
            tokens[f"txt:{token}"] += 1.0  # 文本 token
    for col in RECB_CATEGORY_COLUMNS:  # 类别
        token = clean_category_token(row.get(col))  # 清洗
        if token != "unknown":  # 有效
            tokens[f"cat:{col}:{token}"] += 1.0  # 类别 token
    return tokens  # 返回


def _build_tfidf_vectors(token_counts: dict[str, Counter[str]]) -> dict[str, dict[str, float]]:  # TF-IDF
    doc_freq: Counter[str] = Counter()  # 文档频率
    for counts in token_counts.values():  # 每个商品
        for token in counts:  # 每个 token
            doc_freq[token] += 1  # +1 文档
    n_docs = max(len(token_counts), 1)  # 文档数
    vectors: dict[str, dict[str, float]] = {}  # 输出
    for item_id, counts in token_counts.items():  # 每个 SKU
        total = float(sum(counts.values())) or 1.0  # TF 归一化分母
        vec: dict[str, float] = {}  # 稀疏向量
        for token, count in counts.items():  # 每个 token
            tf = count / total  # 词频
            idf = math.log((1.0 + n_docs) / (1.0 + doc_freq[token])) + 1.0  # 平滑 IDF
            vec[token] = tf * idf  # TF-IDF
        if vec:  # 非空
            vectors[item_id] = vec  # 写入
    return vectors  # 返回


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:  # 稀疏余弦
    if not a or not b:  # 空向量
        return 0.0  # 无相似
    dot = sum(weight * b.get(token, 0.0) for token, weight in a.items())  # 点积
    norm_a = math.sqrt(sum(weight * weight for weight in a.values()))  # ||a||
    norm_b = math.sqrt(sum(weight * weight for weight in b.values()))  # ||b||
    if norm_a <= 0.0 or norm_b <= 0.0:  # 退化
        return 0.0  # 零
    return dot / (norm_a * norm_b)  # 余弦


def _load_interactions(*inter_paths: str | Path) -> pd.DataFrame:  # 读交互
    if not inter_paths:  # 默认
        inter_paths = (DEFAULT_INTER_PATH,)  # 训练集
    frames: list[pd.DataFrame] = []  # 收集
    for path in inter_paths:  # 每个文件
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=["item_id:token", "timestamp:float"],
            dtype={"item_id:token": "string"},
        )
        df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 商品
        df["date"] = pd.to_datetime(df["timestamp:float"], unit="s").dt.normalize()  # 自然日
        frames.append(df[["item_id:token", "date"]])  # 保留列
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()  # 合并


def build_content_index(  # 构建内容索引
    articles_path: Path | str,
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    *,
    as_of: pd.Timestamp | str | None = None,
) -> ContentIndex:
    articles = load_articles_table(Path(articles_path))  # 全量主数据
    token_counts: dict[str, Counter[str]] = {}  # item -> tokens
    for row in articles.itertuples(index=False):  # 每个 SKU
        item_id = canonical_item_id(getattr(row, "item_id", ""))  # 规范
        series = pd.Series(row._asdict())  # 行转 Series
        counts = _item_tokens(series)  # token
        if counts:  # 非空
            token_counts[item_id] = counts  # 写入
    vectors = _build_tfidf_vectors(token_counts)  # TF-IDF

    cold_start_ranked: list[tuple[str, float]] = []  # 冷启动
    if inter_paths is not None:  # 可选交互
        if isinstance(inter_paths, (str, Path)):  # 单路径
            paths = [inter_paths]  # 包装
        else:  # 多路径
            paths = list(inter_paths)  # 列表
        frame = _load_interactions(*paths)  # 读交互
        frame = filter_interactions_as_of(frame, as_of)  # as-of
        if not frame.empty:  # 有数据
            scores = build_window_popularity(frame, item_col="item_id:token", as_of=as_of)  # 热度
            cold_start_ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))  # 降序
    return ContentIndex(item_vectors=vectors, cold_start_ranked=cold_start_ranked)  # 返回


def recall_content(  # 内容召回
    user_history: list[str] | set[str],
    index: ContentIndex,
    *,
    seed_items: int = CONTENT_SEED_ITEMS,
    top_k: int = CONTENT_RECALL_TOP_K,
) -> list[tuple[str, float]]:
    history_list = [canonical_item_id(x) for x in user_history]  # 规范
    if not history_list:  # 冷启动
        return index.cold_start_ranked[:top_k]  # 全局热度 fallback
    history_set = set(history_list)  # 已购
    seeds = history_list[-seed_items:]  # 最近种子
    profile: dict[str, float] = {}  # 用户内容向量（种子均值）
    used = 0  # 有效种子数
    for item_id in seeds:  # 每个种子
        vec = index.item_vectors.get(item_id)  # TF-IDF
        if not vec:  # 无向量
            continue  # 跳过
        used += 1  # 计数
        for token, weight in vec.items():  # 累加
            profile[token] = profile.get(token, 0.0) + weight  # 求和
    if used == 0:  # 种子无内容
        return index.cold_start_ranked[:top_k]  # fallback
    for token in profile:  # 均值
        profile[token] /= float(used)  # 平均

    scored: dict[str, float] = {}  # 候选分
    for item_id, vec in index.item_vectors.items():  # 全 catalog
        if item_id in history_set:  # 已购
            continue  # 跳过
        sim = _cosine(profile, vec)  # 余弦
        if sim > 0.0:  # 有重叠
            scored[item_id] = sim  # 记录
    ranked = sorted(scored.items(), key=lambda pair: (-pair[1], pair[0]))  # 降序
    return ranked[:top_k]  # Top-K
