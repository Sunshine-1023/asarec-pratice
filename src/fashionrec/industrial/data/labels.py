"""Next-basket labels for weekly snapshots."""  # 未来 horizon 内去重购买集合，数量不进 AP

from __future__ import annotations  # 延迟注解

import shutil  # 覆盖旧分区
from pathlib import Path  # 路径

import pandas as pd  # 聚合

from fashionrec.industrial.data.build_events import aggregate_user_day_item_events  # 无事件文件时从交易现算
from fashionrec.industrial.data.snapshots import (  # 快照协议
    SNAPSHOT_SCHEMA_VERSION,  # 索引进度提示
    SnapshotSpec,  # 快照
    build_snapshot_index,  # 样本索引
    label_window,  # 标签窗口
    snapshot_specs_from_split,  # 从切分生成快照
    write_snapshots_parquet,  # 写出索引
)
from fashionrec.industrial.data.split import TimeSplitResult  # 时间切分
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # ID 规范


LABEL_SCHEMA_VERSION = "hm.next_basket.v1"  # 标签语义
PARTITION_COL = "as_of_date"  # 与快照同一分区键
LABEL_COLUMNS = (  # 每个 user-item-as_of 只出现一次
    "user_id",  # 用户
    "item_id",  # 十位商品
    "as_of_date",  # 预测日
    "split",  # train / valid / test
    "label_purchase",  # 窗口内购买
    "label_repeat",  # 预测日及之前买过同一 SKU
    "label_new_to_user",  # 用户首次购买该 SKU
    "label_same_style_new_color",  # 新 SKU，但同 product_code 买过
    "label_quantity",  # 窗口内件数，辅助任务，不替代二值相关性
)  # 列结束
DEFAULT_ARTICLES = Path("data/raw/articles.csv")  # 默认商品主数据


def load_product_codes(articles_path: Path) -> pd.DataFrame:  # article_id -> product_code
    articles_path = Path(articles_path)  # 规范化
    if not articles_path.is_file():  # 缺文件
        raise FileNotFoundError(f"articles file not found: {articles_path}")  # 无法算同款
    articles = pd.read_csv(  # 只读映射所需列
        articles_path,  # 路径
        dtype={"article_id": "string", "product_code": "string"},  # ID 当字符串
        usecols=lambda name: name in {"article_id", "product_code"},  # 兼容缺列
    )
    if "article_id" not in articles.columns or "product_code" not in articles.columns:  # 缺映射
        raise ValueError("articles must contain article_id and product_code")  # 报错
    frame = pd.DataFrame(  # 规范化
        {
            "item_id": articles["article_id"].map(canonical_item_id),  # 十位 SKU
            "product_code": articles["product_code"].astype("string").fillna("").str.strip(),  # 款式
        }
    )
    frame = frame.drop_duplicates("item_id", keep="first")  # 一 SKU 一款式
    return frame  # 返回


def load_events_for_labels(  # 读事件目录或从交易现算
    *,
    events_dir: Path | None = None,  # 本 run 事件
    transactions_path: Path | None = None,  # 无事件时的 CSV
) -> pd.DataFrame:  # 事件表
    if events_dir is None and transactions_path is None:  # 两个都没给
        raise ValueError("load_events_for_labels requires events_dir or transactions_path")  # 报错
    if events_dir is not None:  # 优先事件
        events_dir = Path(events_dir)  # 规范化
        if not events_dir.exists():  # 不存在
            raise FileNotFoundError(f"events dir not found: {events_dir}")  # 报错
        events = pd.read_parquet(events_dir)  # 回读
    else:  # 从交易聚合
        transactions_path = Path(transactions_path)  # 规范化
        if not transactions_path.is_file():  # 缺文件
            raise FileNotFoundError(f"transactions file not found: {transactions_path}")  # 报错
        transactions = pd.read_csv(  # 读 CSV
            transactions_path,  # 路径
            dtype={"customer_id": "string", "article_id": "string"},  # 保前导零
        )
        events = aggregate_user_day_item_events(transactions)  # 同行同日同 SKU 先合成事件
    events = events.copy()  # 不改调用方
    events["user_id"] = events["user_id"].map(canonical_user_id)  # 用户
    events["item_id"] = events["item_id"].map(canonical_item_id)  # 商品
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()  # 自然日
    if "quantity" not in events.columns:  # 兼容无数量列
        events["quantity"] = 1  # 一行一件
    events["quantity"] = pd.to_numeric(events["quantity"], errors="coerce").fillna(1).astype("int64")  # 件数
    return events  # 返回


def build_next_basket_labels(  # 每个快照一份去重 user-item 标签
    events: pd.DataFrame,  # 事件
    specs: list[SnapshotSpec],  # 快照
    *,
    horizon_days: int,  # 未来窗口
    product_codes: pd.DataFrame | None = None,  # item_id, product_code
) -> pd.DataFrame:  # 标签表
    required = {"user_id", "item_id", "date", "quantity"}  # 最少列
    missing = required.difference(events.columns)  # 缺列
    if missing:  # schema 不对
        raise ValueError(f"events missing columns: {sorted(missing)}")  # 报错
    if not specs:  # 无快照
        raise ValueError("specs must not be empty")  # 报错

    frame = events.loc[:, ["user_id", "item_id", "date", "quantity"]].copy()  # 标签所需列
    frame["user_id"] = frame["user_id"].map(canonical_user_id)  # 用户
    frame["item_id"] = frame["item_id"].map(canonical_item_id)  # 商品
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()  # 自然日
    styles = product_codes.copy() if product_codes is not None else pd.DataFrame(columns=["item_id", "product_code"])  # 款式
    if not styles.empty:  # 有主数据
        styles["item_id"] = styles["item_id"].map(canonical_item_id)  # 十位
        styles["product_code"] = styles["product_code"].astype("string").fillna("").str.strip()  # 款式
        styles = styles.drop_duplicates("item_id", keep="first")  # 去重

    rows: list[pd.DataFrame] = []  # 按快照拼接
    for spec in specs:  # 每个预测日
        as_of = pd.Timestamp(spec.as_of_date).normalize()  # 预测日
        start, end = label_window(as_of, horizon_days)  # 窗口
        label_events = frame[(frame["date"] >= start) & (frame["date"] <= end)]  # 未来购买
        if label_events.empty:  # 无正例
            continue  # 跳过
        grouped = (  # 同一 user-item 只留一行
            label_events.groupby(["user_id", "item_id"], sort=True, observed=True)["quantity"]
            .sum()
            .rename("label_quantity")
            .reset_index()
        )
        grouped["as_of_date"] = as_of  # 预测日
        grouped["split"] = spec.split  # 划分
        grouped["label_purchase"] = True  # 正例表
        history = frame[frame["date"] <= as_of][["user_id", "item_id"]].drop_duplicates()  # 含当天
        history["label_repeat"] = True  # 见过该 SKU
        grouped = grouped.merge(history, on=["user_id", "item_id"], how="left")  # 对齐复购
        grouped["label_repeat"] = grouped["label_repeat"].fillna(False).astype(bool)  # 未见过则为否
        grouped["label_new_to_user"] = ~grouped["label_repeat"]  # 与复购互斥
        grouped = grouped.merge(styles, on="item_id", how="left")  # 款式
        grouped["product_code"] = grouped["product_code"].astype("string").fillna("").str.strip()  # 空款式
        style_hist = history.merge(styles, on="item_id", how="left")  # 历史款式
        style_hist["product_code"] = style_hist["product_code"].astype("string").fillna("").str.strip()
        style_hist = style_hist.loc[style_hist["product_code"] != "", ["user_id", "product_code"]].drop_duplicates()
        style_hist["seen_style"] = True  # 见过该款
        grouped = grouped.merge(style_hist, on=["user_id", "product_code"], how="left")  # 同款
        grouped["seen_style"] = grouped["seen_style"].fillna(False).astype(bool)  # 未见过款
        grouped["label_same_style_new_color"] = (  # 新 SKU 但同款
            grouped["label_new_to_user"] & grouped["seen_style"] & grouped["product_code"].ne("")
        )
        grouped["label_quantity"] = grouped["label_quantity"].astype("int64")  # 件数
        rows.append(grouped.loc[:, list(LABEL_COLUMNS)])  # 固定列
    if not rows:  # 没有任何正例
        raise ValueError("no next-basket labels found in label windows")  # 报错
    labels = pd.concat(rows, ignore_index=True)  # 合并快照
    labels = labels.sort_values(["as_of_date", "user_id", "item_id"], kind="mergesort").reset_index(drop=True)  # 稳定
    return labels  # 返回


def write_labels_parquet(labels: pd.DataFrame, output_dir: Path) -> Path:  # 按 as_of_date 分区写出
    output_dir = Path(output_dir)  # 规范化
    if labels.empty:  # 空表
        raise ValueError("cannot write empty labels table")  # 拒绝
    if output_dir.exists():  # 清旧分区
        shutil.rmtree(output_dir)  # 删除
    output_dir.mkdir(parents=True, exist_ok=True)  # 重建
    frame = labels.copy()  # 不改调用方
    frame[PARTITION_COL] = pd.to_datetime(frame["as_of_date"]).dt.strftime("%Y-%m-%d")  # 分区键
    frame.to_parquet(output_dir, partition_cols=[PARTITION_COL], index=False, engine="pyarrow")  # 写出
    return output_dir  # 返回


def build_labels(  # 从切分、事件/交易和商品表写出快照索引与标签
    *,
    split: TimeSplitResult,  # 时间切分
    snapshots_dir: Path,  # 样本索引目录
    labels_dir: Path,  # 标签目录
    horizon_days: int,  # 未来窗口
    events_dir: Path | None = None,  # 事件
    transactions_path: Path | None = None,  # 交易
    articles_path: Path | None = DEFAULT_ARTICLES,  # 商品主数据
    target_mode: str = "next_basket",  # 目前只实现 next-basket
) -> tuple[Path, Path]:  # 两个写出目录
    if target_mode != "next_basket":  # 下一行商品语义不走这套标签
        raise ValueError(f"build_labels supports target_mode='next_basket', got {target_mode!r}")  # 报错
    events = load_events_for_labels(events_dir=events_dir, transactions_path=transactions_path)  # 事件
    specs = snapshot_specs_from_split(split, horizon_days=horizon_days)  # 快照日历
    product_codes = load_product_codes(articles_path) if articles_path is not None else None  # 款式
    snapshots = build_snapshot_index(events, specs, horizon_days=horizon_days)  # 样本索引
    labels = build_next_basket_labels(  # 去重标签
        events,  # 事件
        specs,  # 快照
        horizon_days=horizon_days,  # 窗口
        product_codes=product_codes,  # 款式
    )
    written_snapshots = write_snapshots_parquet(snapshots, snapshots_dir)  # 索引
    written_labels = write_labels_parquet(labels, labels_dir)  # 标签
    print(f"saved snapshots: {written_snapshots} ({len(snapshots):,} rows, schema {SNAPSHOT_SCHEMA_VERSION})")  # 索引
    print(f"saved labels: {written_labels} ({len(labels):,} rows, schema {LABEL_SCHEMA_VERSION})")  # 标签
    return written_snapshots, written_labels  # 返回
