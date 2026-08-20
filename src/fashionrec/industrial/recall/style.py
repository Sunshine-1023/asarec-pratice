"""Style recall: sibling SKUs under recently purchased product_code."""  # 款式/新色召回

from __future__ import annotations  # 延迟注解

from collections import Counter  # 窗口计数
from dataclasses import dataclass, field  # 索引
from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.industrial.data.labels import load_product_codes  # item -> product_code
from fashionrec.shared.domain.ids import canonical_item_id  # SKU
from fashionrec.industrial.recall.window_scores import (  # 窗口热度
    DEFAULT_WINDOW_WEEKS,
    DEFAULT_WINDOW_WEIGHTS,
    build_window_popularity,
    filter_interactions_as_of,
)


DEFAULT_INTER_PATH = Path("data/processed/hm/hm.train.inter")  # 默认训练集
STYLE_RECALL_TOP_K = 50  # 召回 Top-K
STYLE_SEED_ITEMS = 10  # 种子 SKU 数
STYLE_SCHEMA_VERSION = "hm.style.v1"  # 索引语义


@dataclass(frozen=True)  # 款式索引
class StyleIndex:
    item_to_code: dict[str, str] = field(default_factory=dict)  # SKU -> product_code
    code_to_items: dict[str, list[tuple[str, float]]] = field(default_factory=dict)  # 款式内 SKU 热度


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


def build_style_index(  # 构建款式索引
    inter_paths: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_INTER_PATH,
    *,
    articles_path: Path | str | None = None,
    as_of: pd.Timestamp | str | None = None,
    window_weeks: tuple[int, ...] = DEFAULT_WINDOW_WEEKS,
    window_weights: tuple[float, ...] = DEFAULT_WINDOW_WEIGHTS,
) -> StyleIndex:
    if articles_path is None:  # 缺主数据
        return StyleIndex()  # 空索引
    codes = load_product_codes(Path(articles_path))  # item -> code
    item_to_code = {  # 映射
        str(row.item_id): str(row.product_code)
        for row in codes.itertuples(index=False)
        if str(row.product_code).strip()
    }
    if not item_to_code:  # 无款式
        return StyleIndex()  # 空

    if isinstance(inter_paths, (str, Path)):  # 单路径
        paths = [inter_paths]  # 包装
    else:  # 多路径
        paths = list(inter_paths)  # 列表
    frame = _load_interactions(*paths)  # 读交互
    frame = filter_interactions_as_of(frame, as_of)  # as-of 截断

    code_popularity: dict[str, Counter[str]] = {}  # 款式内全局计数（用于排序）
    if not frame.empty:  # 有交互
        scores = build_window_popularity(  # 多窗口融合
            frame,
            item_col="item_id:token",
            window_weeks=window_weeks,
            window_weights=window_weights,
            as_of=as_of,
        )
        for item_id, score in scores.items():  # 每个 SKU
            code = item_to_code.get(item_id)  # 款式
            if not code:  # 无映射
                continue  # 跳过
            code_popularity.setdefault(code, Counter())[item_id] += max(score, 1e-9)  # 累加

    code_to_items: dict[str, list[tuple[str, float]]] = {}  # 输出
    codes_by_style: dict[str, list[str]] = {}  # code -> SKUs
    for item_id, code in item_to_code.items():  # 全 catalog
        codes_by_style.setdefault(code, []).append(item_id)  # 分组
    for code, items in codes_by_style.items():  # 每个款式
        counter = code_popularity.get(code, Counter())  # 热度
        ranked = sorted(((item, float(counter.get(item, 0.0))) for item in items), key=lambda pair: (-pair[1], pair[0]))  # 降序
        code_to_items[code] = ranked  # 写入
    return StyleIndex(item_to_code=item_to_code, code_to_items=code_to_items)  # 返回


def recall_style(  # 款式召回：同 product_code 其它 SKU/颜色
    user_history: list[str] | set[str],
    index: StyleIndex,
    *,
    seed_items: int = STYLE_SEED_ITEMS,
    top_k: int = STYLE_RECALL_TOP_K,
) -> list[tuple[str, float]]:
    history_list = [canonical_item_id(x) for x in user_history]  # 规范
    if not history_list or not index.item_to_code:  # 冷启动
        return []  # 空
    history_set = set(history_list)  # 已购 SKU
    seeds = history_list[-seed_items:]  # 最近种子
    styles: set[str] = set()  # 款式集合
    for item_id in seeds:  # 每个种子
        code = index.item_to_code.get(item_id)  # 款式
        if code:  # 有效
            styles.add(code)  # 记录
    if not styles:  # 无款式
        return []  # 空

    scores: dict[str, float] = {}  # 聚合
    for code in styles:  # 每个款式
        for item_id, pop_score in index.code_to_items.get(code, []):  # 同款 SKU
            if item_id in history_set:  # 已买过
                continue  # 跳过
            scores[item_id] = scores.get(item_id, 0.0) + pop_score  # 累加热度
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))  # 降序
    return ranked[:top_k]  # Top-K
