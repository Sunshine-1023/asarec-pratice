"""User-item cross features for ranking; history includes as_of day only."""  # 用户×商品交叉特征

from __future__ import annotations  # 延迟注解

import shutil  # 覆盖分区
from pathlib import Path  # 路径

import pandas as pd  # 聚合

from fashionrec.baseline.data.customer_features import (  # 人群分桶
    DEFAULT_CUSTOMERS,
    build_customer_feature_table,
    load_customers_table,
)
from fashionrec.baseline.data.labels import load_events_for_labels  # 事件来源
from fashionrec.baseline.data.snapshots import PARTITION_COL  # 分区键
from fashionrec.baseline.data.user_features import (  # 复用 enrichment / 截断
    DEFAULT_ARTICLES,
    enrich_events,
    history_as_of,
    load_item_metadata,
)
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # ID


CROSS_FEATURE_SCHEMA_VERSION = "hm.cross_features.v1"  # 特征语义
POPULARITY_WINDOWS = (7, 28)  # 候选热度窗口（天）
PAIR_COLUMNS = ("user_id", "item_id", "as_of_date")  # 主键


def _as_day(value: pd.Timestamp | str) -> pd.Timestamp:  # 自然日
    return pd.Timestamp(value).normalize()


def _recency_days(as_of: pd.Timestamp, last_date: pd.Timestamp | None) -> float:  # 距上次购买天数
    if last_date is None:  # 从未购买
        return float("nan")  # 缺失
    return float((as_of - _as_day(last_date)).days)  # 天数差


def _window_hist(hist: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:  # [as_of-days+1, as_of]
    if hist.empty:  # 无历史
        return hist.copy()  # 空窗
    start = as_of - pd.Timedelta(days=days - 1)  # 窗口起
    return hist[(hist["date"] >= start) & (hist["date"] <= as_of)].copy()  # 含 as_of


def _weighted_share(hist: pd.DataFrame, column: str, token: str) -> float:  # 用户对该属性偏好的件数占比
    if hist.empty:  # 无历史
        return 0.0  # 零匹配
    total = float(hist["quantity"].sum())  # 总件数
    if total <= 0.0:  # 无购买
        return 0.0  # 零
    matched = float(hist.loc[hist[column].astype(str) == str(token), "quantity"].sum())  # 匹配件数
    return matched / total  # 占比


def _dominant_channel(hist: pd.DataFrame) -> int:  # 用户主渠道（按件数）
    if hist.empty:  # 无历史
        return 0  # 占位
    grouped = hist.groupby("sales_channel_mode")["quantity"].sum()  # 各渠道件数
    if grouped.empty:  # 空
        return 0  # 占位
    return int(grouped.idxmax())  # 众数渠道


def _item_mean_price(hist: pd.DataFrame, item_id: str) -> float:  # 候选 SKU 在 as-of 前的均价
    rows = hist.loc[hist["item_id"] == item_id]  # 该 SKU
    priced = rows.loc[rows["mean_price"].notna()]  # 有效价
    if priced.empty:  # 无价格
        return float("nan")  # 缺失
    spend = (priced["mean_price"] * priced["quantity"]).sum()  # 近似 spend
    qty = float(priced["quantity"].sum())  # 件数
    return float(spend / qty) if qty > 0.0 else float("nan")  # 均价


def _count_level(hist: pd.DataFrame, column: str, token: str) -> float:  # 用户在某层级购买件数
    if hist.empty or not str(token).strip():  # 空 token
        return 0.0  # 零
    return float(hist.loc[hist[column].astype(str) == str(token), "quantity"].sum())  # 件数


def _last_purchase_date(hist: pd.DataFrame, column: str, token: str) -> pd.Timestamp | None:  # 上次购买日
    if hist.empty or not str(token).strip():  # 无 token
        return None  # 从未
    rows = hist.loc[hist[column].astype(str) == str(token)]  # 匹配行
    if rows.empty:  # 无购买
        return None  # 从未
    return _as_day(rows["date"].max())  # 最近一天


def load_cross_feature_pairs(  # 从标签或候选 CSV 读 user-item-as_of
    *,
    labels_dir: Path | None = None,  # 分区标签
    candidates_path: Path | None = None,  # 显式候选
) -> pd.DataFrame:  # 至少含主键三列
    if labels_dir is not None and Path(labels_dir).exists():  # 优先标签
        frame = pd.read_parquet(labels_dir)  # 回读
    elif candidates_path is not None:  # 候选文件
        candidates_path = Path(candidates_path)  # 规范化
        if not candidates_path.is_file():  # 缺文件
            raise FileNotFoundError(f"candidates file not found: {candidates_path}")  # 报错
        frame = pd.read_csv(candidates_path, dtype={"user_id": "string", "item_id": "string"})  # 读 CSV
    else:  # 都没给
        raise ValueError("load_cross_feature_pairs requires labels_dir or candidates_path")  # 报错
    missing = set(PAIR_COLUMNS).difference(frame.columns)  # 缺主键
    if missing:  # schema 不对
        raise ValueError(f"pairs missing columns: {sorted(missing)}")  # 报错
    pairs = frame.loc[:, list(PAIR_COLUMNS)].copy()  # 只要主键
    if "split" in frame.columns:  # 可选划分
        pairs["split"] = frame["split"].astype(str)  # 保留
    pairs["user_id"] = pairs["user_id"].map(canonical_user_id)  # 规范
    pairs["item_id"] = pairs["item_id"].map(canonical_item_id)  # 规范
    pairs["as_of_date"] = pd.to_datetime(pairs["as_of_date"]).dt.normalize()  # 自然日
    return pairs.drop_duplicates(list(PAIR_COLUMNS), keep="first").reset_index(drop=True)  # 去重


def _load_user_cohorts(customers_path: Path | None) -> pd.DataFrame:  # user_id -> age_bucket
    if customers_path is None:  # 未提供
        return pd.DataFrame(columns=["user_id", "age_bucket"])  # 空表
    customers_path = Path(customers_path)  # 规范化
    if not customers_path.is_file():  # 缺文件
        return pd.DataFrame(columns=["user_id", "age_bucket"])  # 空表
    customers = load_customers_table(customers_path)  # 读主数据
    features = build_customer_feature_table(customers, keep_full_customer_universe=True)  # 全量
    return features.loc[:, ["user_id", "age_bucket"]].copy()  # 人群键


def compute_cross_feature_row(  # 单 user-item-as_of
    user_id: str,
    item_id: str,
    as_of: pd.Timestamp | str,
    *,
    user_hist: pd.DataFrame,  # 该用户 as-of 历史（可含未来，内部不再截）
    item_row: pd.Series,  # 候选商品属性
    global_hist: pd.DataFrame,  # 全体 as-of 历史
    cohort_hist: pd.DataFrame | None = None,  # 同 age_bucket 用户历史
    split: str | None = None,  # train/valid/test
) -> dict[str, object]:  # 一行特征
    user_id = canonical_user_id(user_id)  # 规范
    item_id = canonical_item_id(item_id)  # 规范
    as_of_day = _as_day(as_of)  # 预测日
    hist = user_hist.copy()  # 用户历史
    if not hist.empty:  # 有行
        hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()  # 自然日
    product_code = str(item_row.get("product_code", "") or "").strip()  # 款式
    department = str(item_row.get("department_name", "") or "").strip()  # 部门
    colour = str(item_row.get("colour_group_name", "") or "").strip()  # 颜色
    product_type = str(item_row.get("product_type_name", "") or "").strip()  # 类型
    row: dict[str, object] = {  # 主键
        "user_id": user_id,  # 用户
        "item_id": item_id,  # 候选 SKU
        "as_of_date": as_of_day,  # 预测日
        "feature_version": CROSS_FEATURE_SCHEMA_VERSION,  # 版本
    }  # 主键结束
    if split is not None:  # 可选划分
        row["split"] = split  # 写入
    row["user_item_purchase_count"] = _count_level(hist, "item_id", item_id)  # SKU 件数
    row["user_style_purchase_count"] = _count_level(hist, "product_code", product_code) if product_code else 0.0  # 款式
    row["user_department_purchase_count"] = _count_level(hist, "department_name", department) if department else 0.0  # 部门
    row["user_colour_purchase_count"] = _count_level(hist, "colour_group_name", colour) if colour else 0.0  # 颜色
    row["user_product_type_purchase_count"] = _count_level(hist, "product_type_name", product_type) if product_type else 0.0  # 类型
    row["user_item_recency_days:float"] = _recency_days(as_of_day, _last_purchase_date(hist, "item_id", item_id))  # SKU 间隔
    row["user_style_recency_days:float"] = _recency_days(as_of_day, _last_purchase_date(hist, "product_code", product_code))  # 款式
    row["user_department_recency_days:float"] = _recency_days(as_of_day, _last_purchase_date(hist, "department_name", department))  # 部门
    row["user_colour_recency_days:float"] = _recency_days(as_of_day, _last_purchase_date(hist, "colour_group_name", colour))  # 颜色
    priced = hist.loc[hist["mean_price"].notna()] if not hist.empty else hist  # 有效价
    if priced.empty:  # 用户无价格
        row["user_mean_price:float"] = float("nan")  # 缺失
    else:  # 有价格
        spend = (priced["mean_price"] * priced["quantity"]).sum()  # spend
        qty = float(priced["quantity"].sum())  # 件数
        row["user_mean_price:float"] = float(spend / qty) if qty > 0.0 else float("nan")  # 用户均价
    candidate_price = _item_mean_price(global_hist, item_id)  # 候选全局均价
    row["candidate_mean_price:float"] = candidate_price  # 候选价
    if pd.isna(row["user_mean_price:float"]) or pd.isna(candidate_price):  # 任一侧缺失
        row["price_diff:float"] = float("nan")  # 差值缺失
    else:  # 两侧都有
        row["price_diff:float"] = float(candidate_price) - float(row["user_mean_price:float"])  # 价差
    row["department_preference_match:float"] = _weighted_share(hist, "department_name", department) if department else 0.0  # 部门匹配
    row["colour_preference_match:float"] = _weighted_share(hist, "colour_group_name", colour) if colour else 0.0  # 颜色匹配
    row["product_type_preference_match:float"] = _weighted_share(hist, "product_type_name", product_type) if product_type else 0.0  # 类型匹配
    bought_style = row["user_style_purchase_count"] > 0.0 if product_code else False  # 买过同款
    bought_sku = row["user_item_purchase_count"] > 0.0  # 买过 SKU
    row["user_bought_same_style:float"] = 1.0 if bought_style else 0.0  # 同款
    row["candidate_same_style_new_color:float"] = 1.0 if (bought_style and not bought_sku and product_code) else 0.0  # 同款新色
    dominant_channel = _dominant_channel(hist)  # 用户主渠道
    for days in POPULARITY_WINDOWS:  # 全局 / 人群 / 渠道热度
        window = _window_hist(global_hist, as_of_day, days)  # 全体窗口
        suffix = f"_{days}d"  # 后缀
        row[f"item_global_purchase_count{suffix}"] = float(window.loc[window["item_id"] == item_id, "quantity"].sum())  # 全局
        channel_window = window.loc[window["sales_channel_mode"] == dominant_channel]  # 用户主渠道窗
        row[f"item_channel_purchase_count{suffix}"] = float(channel_window.loc[channel_window["item_id"] == item_id, "quantity"].sum())  # 渠道
        if cohort_hist is not None and not cohort_hist.empty:  # 有人群历史
            cohort_window = _window_hist(cohort_hist, as_of_day, days)  # 人群窗
            row[f"item_cohort_purchase_count{suffix}"] = float(cohort_window.loc[cohort_window["item_id"] == item_id, "quantity"].sum())  # 人群
        else:  # 无 cohort
            row[f"item_cohort_purchase_count{suffix}"] = 0.0  # 零
    return row  # 返回


def build_cross_feature_table(  # 批量 user-item-as_of
    pairs: pd.DataFrame,  # 主键 + 可选 split
    events: pd.DataFrame,  # 原始事件
    item_metadata: pd.DataFrame,  # SKU 属性
    *,
    user_cohorts: pd.DataFrame | None = None,  # age_bucket
) -> pd.DataFrame:  # 特征表
    if pairs.empty:  # 无对
        raise ValueError("pairs must not be empty")  # 报错
    enriched = enrich_events(events, item_metadata)  # 带属性事件
    meta = item_metadata.set_index("item_id", drop=False)  # 索引便于查
    cohort_map = {}  # user -> bucket
    if user_cohorts is not None and not user_cohorts.empty:  # 有人群
        cohort_map = user_cohorts.set_index("user_id")["age_bucket"].astype(str).to_dict()  # 映射
    rows: list[dict[str, object]] = []  # 收集
    for as_of, group in pairs.groupby("as_of_date", sort=True):  # 按预测日批处理
        as_of_day = _as_day(as_of)  # 自然日
        global_hist = history_as_of(enriched, as_of_day)  # 全体历史
        by_user = {user_id: frame for user_id, frame in global_hist.groupby("user_id", sort=True)}  # 用户切片
        cohort_users: dict[str, set[str]] = {}  # bucket -> users
        for user_id in group["user_id"].unique():  # 本批用户
            bucket = cohort_map.get(canonical_user_id(user_id))  # 分桶
            if bucket:  # 有效
                cohort_users.setdefault(bucket, set()).add(canonical_user_id(user_id))  # 收集
        cohort_hists: dict[str, pd.DataFrame] = {}  # bucket -> hist
        for bucket, users in cohort_users.items():  # 各人群
            cohort_hists[bucket] = global_hist[global_hist["user_id"].isin(users)].copy()  # 切片
        for record in group.itertuples(index=False):  # 每个候选
            user_id = canonical_user_id(record.user_id)  # 用户
            item_id = canonical_item_id(record.item_id)  # SKU
            if item_id not in meta.index:  # 未见 SKU
                raise ValueError(f"item_id {item_id!r} missing from item metadata")  # 拒绝静默
            item_row = meta.loc[item_id]  # 属性
            bucket = cohort_map.get(user_id)  # 人群
            cohort_hist = cohort_hists.get(bucket) if bucket else None  # 人群历史
            split = getattr(record, "split", None)  # 可选
            rows.append(  # 一行
                compute_cross_feature_row(
                    user_id,  # 用户
                    item_id,  # SKU
                    as_of_day,  # 预测日
                    user_hist=by_user.get(user_id, enriched.iloc[0:0]),  # 用户历史或空
                    item_row=item_row,  # 商品
                    global_hist=global_hist,  # 全体
                    cohort_hist=cohort_hist,  # 人群
                    split=str(split) if split is not None and str(split) != "nan" else None,  # 划分
                )
            )  # 追加
    frame = pd.DataFrame(rows)  # 组装
    return frame.sort_values(["as_of_date", "user_id", "item_id"], kind="mergesort").reset_index(drop=True)  # 稳定


def assert_cross_features_ignore_future_events(  # 混入标签周不得改变 as-of 交叉特征
    events: pd.DataFrame,  # 全量事件
    pairs: pd.DataFrame,  # 至少一行
    item_metadata: pd.DataFrame,  # 商品属性
    *,
    user_cohorts: pd.DataFrame | None = None,  # 可选人群
) -> None:  # 违规则抛出
    if pairs.empty:  # 无对
        raise ValueError("pairs must not be empty for leakage check")  # 无法测
    sample = pairs.iloc[0]  # 取一行
    as_of_day = _as_day(sample["as_of_date"])  # 预测日
    user_id = canonical_user_id(sample["user_id"])  # 用户
    item_id = canonical_item_id(sample["item_id"])  # SKU
    enriched = enrich_events(events, item_metadata)  # enrichment
    future = enriched[(enriched["user_id"] == user_id) & (enriched["date"] > as_of_day)]  # 未来
    if future.empty:  # 需要未来对照
        raise ValueError("leakage check requires future events after as_of")  # 报错
    baseline = build_cross_feature_table(pairs.iloc[[0]], events, item_metadata, user_cohorts=user_cohorts)  # 基准
    future_raw = events[
        (events["user_id"].map(canonical_user_id) == user_id)
        & (pd.to_datetime(events["date"]).dt.normalize() > as_of_day)
    ]  # 原始未来行
    with_future = build_cross_feature_table(  # 重复追加未来行
        pairs.iloc[[0]],
        pd.concat([events, future_raw], ignore_index=True),
        item_metadata,
        user_cohorts=user_cohorts,
    )
    if not baseline.equals(with_future):  # 不一致
        diff_cols = [col for col in baseline.columns if not baseline[col].equals(with_future[col])]  # 变化列
        raise AssertionError(f"as-of cross features changed after appending future events: {diff_cols[:5]}")  # 失败


def write_cross_features_parquet(features: pd.DataFrame, output_dir: Path) -> Path:  # 按 as_of_date 分区
    output_dir = Path(output_dir)  # 规范化
    if features.empty:  # 空
        raise ValueError("cannot write empty cross features table")  # 拒绝
    if output_dir.exists():  # 清旧
        shutil.rmtree(output_dir)  # 删除
    output_dir.mkdir(parents=True, exist_ok=True)  # 重建
    frame = features.copy()  # 拷贝
    frame[PARTITION_COL] = pd.to_datetime(frame["as_of_date"]).dt.strftime("%Y-%m-%d")  # 分区键
    frame.to_parquet(output_dir, partition_cols=[PARTITION_COL], index=False, engine="pyarrow")  # 写出
    return output_dir  # 返回


def build_cross_features(  # 数据准备入口
    *,
    output_dir: Path,  # 分区 parquet 根
    events_dir: Path | None = None,  # 本 run 事件
    transactions_path: Path | None = None,  # 或交易 CSV
    articles_path: Path | None = DEFAULT_ARTICLES,  # 商品属性
    labels_dir: Path | None = None,  # 默认从标签读对
    candidates_path: Path | None = None,  # 或显式候选
    customers_path: Path | None = DEFAULT_CUSTOMERS,  # 人群分桶
) -> Path:  # 写出目录
    events = load_events_for_labels(events_dir=events_dir, transactions_path=transactions_path)  # 事件
    if articles_path is None:  # 必须有 articles
        raise ValueError("articles_path is required to build cross features")  # 报错
    item_metadata = load_item_metadata(articles_path)  # SKU 属性
    pairs = load_cross_feature_pairs(labels_dir=labels_dir, candidates_path=candidates_path)  # 候选对
    user_cohorts = _load_user_cohorts(customers_path)  # 人群
    features = build_cross_feature_table(pairs, events, item_metadata, user_cohorts=user_cohorts)  # 特征
    written = write_cross_features_parquet(features, output_dir)  # 落盘
    print(f"saved cross features: {written} ({len(features):,} rows, schema {CROSS_FEATURE_SCHEMA_VERSION})")  # 提示
    return written  # 返回
