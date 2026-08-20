"""Point-in-time user behavior features per snapshot; history includes as_of day only."""  # as-of 用户行为统计

from __future__ import annotations  # 延迟注解

import math  # 熵
import shutil  # 覆盖分区
from pathlib import Path  # 路径

import pandas as pd  # 聚合

from fashionrec.industrial.data.item_features import UNKNOWN_TOKEN, clean_category_token  # 类别 token
from fashionrec.industrial.data.labels import load_events_for_labels  # 事件来源
from fashionrec.industrial.data.snapshots import (  # 快照日历
    PARTITION_COL,
    SnapshotSpec,
    snapshot_specs_from_split,
)
from fashionrec.industrial.data.split import TimeSplitResult  # 切分
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # ID


USER_FEATURE_SCHEMA_VERSION = "hm.user_features.v1"  # 特征语义
BEHAVIOR_WINDOWS = (1, 7, 28, 84, 182, 365)  # 计划窗口（天）
DEFAULT_ARTICLES = Path("data/raw/articles.csv")  # 商品主数据
PRICE_BAND_BOUNDS = (0.05, 0.15)  # 低/中/高价格带


def _as_day(value: pd.Timestamp | str) -> pd.Timestamp:  # 自然日
    return pd.Timestamp(value).normalize()


def _entropy(counts: pd.Series) -> float:  # 分布熵
    total = float(counts.sum())  # 总量
    if total <= 0.0:  # 空
        return 0.0  # 零熵
    probs = counts.astype(float) / total  # 概率
    return float(-(probs * probs.map(lambda p: math.log(p) if p > 0.0 else 0.0)).sum())  # 熵


def _price_band(price: float) -> str:  # 离散价格带，不做连续距离
    if price < PRICE_BAND_BOUNDS[0]:  # 低价
        return "low"  # 低
    if price < PRICE_BAND_BOUNDS[1]:  # 中价
        return "mid"  # 中
    return "high"  # 高


def load_item_metadata(articles_path: Path) -> pd.DataFrame:  # item_id -> 品类/颜色/款式
    articles_path = Path(articles_path)  # 规范化
    if not articles_path.is_file():  # 缺文件
        raise FileNotFoundError(f"articles file not found: {articles_path}")  # 报错
    usecols = ["article_id", "product_code", "colour_group_name", "department_name", "product_type_name"]  # 所需列
    articles = pd.read_csv(  # 读 CSV
        articles_path,
        dtype={"article_id": "string", "product_code": "string"},
        usecols=lambda name: name in usecols,
    )
    if "article_id" not in articles.columns:  # 主键
        raise ValueError("articles must contain article_id")  # 报错
    for col in ("product_code", "colour_group_name", "department_name", "product_type_name"):  # 可选列
        if col not in articles.columns:  # 缺列
            articles[col] = pd.NA  # 补空
    meta = pd.DataFrame(  # 规范化
        {
            "item_id": articles["article_id"].map(canonical_item_id),  # SKU
            "product_code": articles["product_code"].astype("string").fillna("").str.strip(),  # 款式
            "colour_group_name": articles["colour_group_name"].map(clean_category_token),  # 颜色
            "department_name": articles["department_name"].map(clean_category_token),  # 部门
            "product_type_name": articles["product_type_name"].map(clean_category_token),  # 类型
        }
    )
    return meta.drop_duplicates("item_id", keep="first")  # 一 SKU 一行


def enrich_events(events: pd.DataFrame, item_metadata: pd.DataFrame) -> pd.DataFrame:  # 事件 + 商品属性
    required = {"user_id", "item_id", "date", "quantity"}  # 最少列
    missing = required.difference(events.columns)  # 缺列
    if missing:  # schema 不对
        raise ValueError(f"events missing columns: {sorted(missing)}")  # 报错
    frame = events.copy()  # 不改调用方
    frame["user_id"] = frame["user_id"].map(canonical_user_id)  # 用户
    frame["item_id"] = frame["item_id"].map(canonical_item_id)  # 商品
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()  # 自然日
    frame["quantity"] = pd.to_numeric(frame.get("quantity", 1), errors="coerce").fillna(1).astype("int64")  # 件数
    if "mean_price" not in frame.columns:  # 无价格
        frame["mean_price"] = float("nan")  # 允许空
    else:  # 有价格
        frame["mean_price"] = pd.to_numeric(frame["mean_price"], errors="coerce")  # 数值
    if "sales_channel_mode" not in frame.columns:  # 无渠道
        frame["sales_channel_mode"] = 0  # 占位
    frame["sales_channel_mode"] = pd.to_numeric(frame["sales_channel_mode"], errors="coerce").fillna(0).astype("int64")  # 渠道
    meta = item_metadata.copy()  # 商品表
    frame = frame.merge(meta, on="item_id", how="left")  # 左连接
    for col in ("product_code", "colour_group_name", "department_name", "product_type_name"):  # 空款式/类别
        if col in frame.columns:  # 存在
            if col == "product_code":  # 款式保留原字符串
                frame[col] = frame[col].astype("string").fillna("").str.strip()  # 空串
            else:  # 类别
                frame[col] = frame[col].map(clean_category_token).fillna(UNKNOWN_TOKEN)  # unknown
    frame["price_band"] = frame["mean_price"].map(  # 价格带
        lambda value: _price_band(float(value)) if pd.notna(value) else UNKNOWN_TOKEN  # 缺失 unknown
    )
    return frame.sort_values(["user_id", "date", "item_id"], kind="mergesort").reset_index(drop=True)  # 稳定


def history_as_of(events: pd.DataFrame, as_of: pd.Timestamp | str) -> pd.DataFrame:  # 只含 as_of 及之前
    as_of_day = _as_day(as_of)  # 预测日
    if events.empty:  # 无列或无行
        return events.copy()  # 直接返回
    frame = events.copy()  # 拷贝
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()  # 自然日
    return frame[frame["date"] <= as_of_day].copy()  # 含当天


def assert_user_features_ignore_future_events(  # 混入标签周事件不得改变 as-of 特征
    events: pd.DataFrame,  # 全量事件（可含未来）
    *,
    user_id: str,
    as_of: pd.Timestamp | str,
    item_metadata: pd.DataFrame,
    windows: tuple[int, ...] = BEHAVIOR_WINDOWS,
) -> None:  # 违规则抛出
    as_of_day = _as_day(as_of)  # 预测日
    user = canonical_user_id(user_id)  # 规范
    enriched = enrich_events(events, item_metadata)  #  enrichment
    user_events = enriched[enriched["user_id"] == user]  # 该用户
    if user_events.empty:  # 无事件
        raise ValueError(f"user {user!r} has no events for leakage check")  # 无法测
    future = user_events[user_events["date"] > as_of_day]  # 标签周
    if future.empty:  # 没有未来对照
        raise ValueError("leakage check requires future events after as_of")  # 需要未来行
    baseline = compute_user_feature_row(  # 只用历史
        user,
        user_events,
        as_of_day,
        split="valid",
        windows=windows,
    )
    with_future = compute_user_feature_row(  # 输入含未来，但 as_of 必须截断
        user,
        pd.concat([user_events, future], ignore_index=True),
        as_of_day,
        split="valid",
        windows=windows,
    )
    if baseline != with_future:  # 不一致 = 泄漏
        changed = [key for key in baseline if baseline.get(key) != with_future.get(key)]  # 变化列
        raise AssertionError(f"as-of user features changed after appending future events: {changed[:5]}")  # 失败


def _window_hist(hist: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:  # [as_of-days+1, as_of]
    if hist.empty:  # 无历史列
        return hist.copy()  # 空窗
    start = as_of - pd.Timedelta(days=days - 1)  # 窗口起
    return hist[(hist["date"] >= start) & (hist["date"] <= as_of)].copy()  # 含 as_of


def _weighted_repeat_rate(window: pd.DataFrame, hist: pd.DataFrame, *, column: str) -> tuple[float, float, float]:  # 复购/新款/同款
    if window.empty:  # 窗口无购买
        return 0.0, 0.0, 0.0  # 全零
    total_qty = float(window["quantity"].sum())  # 总件数
    if total_qty <= 0.0:  # 无件数
        return 0.0, 0.0, 0.0  # 全零
    repeat_qty = 0.0  # 复购 SKU
    same_style_qty = 0.0  # 同款
    for row in window.itertuples(index=False):  # 逐事件
        qty = float(row.quantity)  # 件数
        earlier_items = hist[(hist["date"] < row.date) & (hist["item_id"] == row.item_id)]  # 更早买过同 SKU
        if not earlier_items.empty:  # 复购
            repeat_qty += qty  # 累计
        code = str(getattr(row, "product_code", "") or "").strip()  # 款式
        if code:  # 有款式
            earlier_styles = hist[(hist["date"] < row.date) & (hist["product_code"] == code)]  # 更早买过同款
            if not earlier_styles.empty:  # 同款
                same_style_qty += qty  # 累计
    repeat_rate = repeat_qty / total_qty  # 复购率
    same_style_rate = same_style_qty / total_qty  # 同款率
    new_item_rate = 1.0 - repeat_rate  # 新 SKU 率
    return repeat_rate, new_item_rate, same_style_rate  # 返回


def _window_features(hist: pd.DataFrame, as_of: pd.Timestamp, days: int) -> dict[str, float]:  # 单窗口统计
    window = _window_hist(hist, as_of, days)  # 窗口内
    suffix = f"_{days}d"  # 列后缀
    out: dict[str, float] = {}  # 结果
    purchase_count = float(window["quantity"].sum()) if not window.empty else 0.0  # 购买件数
    active_days = float(window["date"].nunique()) if not window.empty else 0.0  # 活跃购物日
    out[f"purchase_count{suffix}"] = purchase_count  # 件数
    out[f"active_days{suffix}"] = active_days  # 天数
    out[f"basket_size{suffix}"] = purchase_count / active_days if active_days > 0.0 else 0.0  # 日均件数
    if window.empty or window["mean_price"].notna().sum() == 0:  # 无有效价格
        out[f"monetary_sum{suffix}"] = 0.0  # 金额和
        out[f"monetary_mean{suffix}"] = 0.0  # 均价
    else:  # 有价格
        priced = window.loc[window["mean_price"].notna()].copy()  # 有效价
        priced["spend"] = priced["mean_price"] * priced["quantity"]  # 近似 spend
        out[f"monetary_sum{suffix}"] = float(priced["spend"].sum())  # 总和
        out[f"monetary_mean{suffix}"] = float(priced["spend"].sum() / float(priced["quantity"].sum()))  # 均值
    total_qty = float(window["quantity"].sum()) if not window.empty else 0.0  # 渠道分母
    if total_qty > 0.0:  # 有购买
        for channel in (1, 2):  # H&M 两渠道
            qty = float(window.loc[window["sales_channel_mode"] == channel, "quantity"].sum())  # 渠道件数
            out[f"channel_{channel}_share{suffix}"] = qty / total_qty  # 占比
    else:  # 无购买
        out[f"channel_1_share{suffix}"] = 0.0  # 零
        out[f"channel_2_share{suffix}"] = 0.0  # 零
    if window.empty:  # 空窗口
        out[f"department_entropy{suffix}"] = 0.0  # 熵
        out[f"colour_entropy{suffix}"] = 0.0  # 熵
        out[f"price_band_entropy{suffix}"] = 0.0  # 熵
        out[f"style_diversity{suffix}"] = 0.0  # 款式数
        out[f"repeat_rate{suffix}"] = 0.0  # 复购
        out[f"new_item_rate{suffix}"] = 0.0  # 新款
        out[f"same_style_rate{suffix}"] = 0.0  # 同款
        return out  # 返回
    weighted = window.groupby("department_name")["quantity"].sum()  # 部门分布
    out[f"department_entropy{suffix}"] = _entropy(weighted)  # 部门熵
    weighted = window.groupby("colour_group_name")["quantity"].sum()  # 颜色分布
    out[f"colour_entropy{suffix}"] = _entropy(weighted)  # 颜色熵
    weighted = window.groupby("price_band")["quantity"].sum()  # 价格带分布
    out[f"price_band_entropy{suffix}"] = _entropy(weighted)  # 价格带熵
    codes = window.loc[window["product_code"].astype(str).str.strip().ne(""), "product_code"]  # 非空款式
    out[f"style_diversity{suffix}"] = float(codes.nunique()) if not codes.empty else 0.0  # 款式多样性
    repeat_rate, new_item_rate, same_style_rate = _weighted_repeat_rate(window, hist, column="item_id")  # 复购结构
    out[f"repeat_rate{suffix}"] = repeat_rate  # 复购
    out[f"new_item_rate{suffix}"] = new_item_rate  # 新款
    out[f"same_style_rate{suffix}"] = same_style_rate  # 同款
    return out  # 单窗口结束


def compute_user_feature_row(  # 单用户单 as_of 特征
    user_id: str,
    user_events: pd.DataFrame,  # 可含未来；内部会截到 as_of
    as_of: pd.Timestamp | str,
    *,
    split: str,
    windows: tuple[int, ...] = BEHAVIOR_WINDOWS,
) -> dict[str, object]:  # 一行特征
    user_id = canonical_user_id(user_id)  # 规范
    as_of_day = _as_day(as_of)  # 预测日
    hist = history_as_of(user_events, as_of_day)  # 严格历史
    row: dict[str, object] = {  # 主键
        "user_id": user_id,  # 用户
        "as_of_date": as_of_day,  # 预测日
        "split": split,  # train/valid/test
        "feature_version": USER_FEATURE_SCHEMA_VERSION,  # 版本
    }  # 主键结束
    if hist.empty:  # 冷启动
        row["recency_days:float"] = float("nan")  # 无上次购买
        row["avg_shopping_interval_days:float"] = 0.0  # 无间隔
        for days in windows:  # 各窗口置零
            row.update(_window_features(hist, as_of_day, days))  # 零特征
        return row  # 返回
    shopping_days = sorted(hist["date"].unique())  # 购物日
    row["recency_days:float"] = float((as_of_day - shopping_days[-1]).days)  # 距上次购买
    if len(shopping_days) < 2:  # 只有一天
        row["avg_shopping_interval_days:float"] = 0.0  # 无间隔
    else:  # 多天
        gaps = [(shopping_days[idx + 1] - shopping_days[idx]).days for idx in range(len(shopping_days) - 1)]  # 间隔
        row["avg_shopping_interval_days:float"] = float(sum(gaps) / len(gaps))  # 平均间隔
    for days in windows:  # 多窗口
        row.update(_window_features(hist, as_of_day, days))  # 窗口特征
    return row  # 返回


def build_user_feature_table(  # 每个 snapshot × 有历史的用户
    events: pd.DataFrame,  # 原始事件
    specs: list[SnapshotSpec],  # as_of 日历
    item_metadata: pd.DataFrame,  # 商品属性
    *,
    windows: tuple[int, ...] = BEHAVIOR_WINDOWS,
) -> pd.DataFrame:  # 特征表
    if not specs:  # 无快照
        raise ValueError("specs must not be empty")  # 报错
    enriched = enrich_events(events, item_metadata)  # 带属性事件
    by_user = {user_id: frame for user_id, frame in enriched.groupby("user_id", sort=True)}  # 按用户
    rows: list[dict[str, object]] = []  # 收集
    for spec in specs:  # 每个 as_of
        as_of = _as_day(spec.as_of_date)  # 预测日
        for user_id, user_events in by_user.items():  # 每个用户
            if user_events[user_events["date"] <= as_of].empty:  # 无历史
                continue  # 跳过
            rows.append(  # 一行
                compute_user_feature_row(
                    user_id,  # 用户
                    user_events,  # 全量用户事件
                    as_of,  # 预测日
                    split=spec.split,  # 划分
                    windows=windows,  # 窗口
                )
            )  # 追加
    if not rows:  # 空
        raise ValueError("no user feature rows generated")  # 报错
    frame = pd.DataFrame(rows)  # 表
    return frame.sort_values(["as_of_date", "user_id", "split"], kind="mergesort").reset_index(drop=True)  # 稳定


def write_user_features_parquet(features: pd.DataFrame, output_dir: Path) -> Path:  # 按 as_of_date 分区
    output_dir = Path(output_dir)  # 规范化
    if features.empty:  # 空
        raise ValueError("cannot write empty user features table")  # 拒绝
    if output_dir.exists():  # 清旧
        shutil.rmtree(output_dir)  # 删除
    output_dir.mkdir(parents=True, exist_ok=True)  # 重建
    frame = features.copy()  # 拷贝
    frame[PARTITION_COL] = pd.to_datetime(frame["as_of_date"]).dt.strftime("%Y-%m-%d")  # 分区键
    frame.to_parquet(output_dir, partition_cols=[PARTITION_COL], index=False, engine="pyarrow")  # 写出
    return output_dir  # 返回


def build_user_features(  # 数据准备入口
    *,
    split: TimeSplitResult,  # 时间切分
    output_dir: Path,  # 分区 parquet 根
    horizon_days: int = 7,  # 与标签窗口一致，用于 specs
    events_dir: Path | None = None,  # 本 run 事件
    transactions_path: Path | None = None,  # 或交易 CSV
    articles_path: Path | None = DEFAULT_ARTICLES,  # 商品属性
    windows: tuple[int, ...] = BEHAVIOR_WINDOWS,  # 行为窗口
) -> Path:  # 写出目录
    events = load_events_for_labels(events_dir=events_dir, transactions_path=transactions_path)  # 事件
    item_metadata = load_item_metadata(articles_path) if articles_path is not None else pd.DataFrame()  # 属性
    if item_metadata.empty:  # 无 articles
        raise ValueError("articles_path is required to build user features")  # 需要商品属性
    specs = snapshot_specs_from_split(split, horizon_days=horizon_days)  # 快照
    features = build_user_feature_table(events, specs, item_metadata, windows=windows)  # 特征
    written = write_user_features_parquet(features, output_dir)  # 落盘
    print(f"saved user features: {written} ({len(features):,} rows, schema {USER_FEATURE_SCHEMA_VERSION})")  # 提示
    return written  # 返回
