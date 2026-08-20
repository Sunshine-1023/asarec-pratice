"""Aggregate raw transaction rows into user-day-item events."""  # 同行同日同 SKU 合成事件，数量不直接进 AP

from __future__ import annotations  # 延迟注解

import argparse  # CLI
import shutil  # 覆盖写出时清掉旧分区
from pathlib import Path  # 路径

import pandas as pd  # 聚合

from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # ID 规范


EVENT_SCHEMA_VERSION = "hm.user_day_item.v1"  # 事件表语义版本；购物篮阶段再升级
PARTITION_COL = "year_month"  # Hive 分区列，YYYY-MM
EVENT_COLUMNS = (  # 写出列顺序
    "user_id",  # 规范化用户 ID
    "item_id",  # 十位商品 ID
    "date",  # 自然日，无日内顺序
    "quantity",  # 原始交易行数；标签仍按去重 item 集合
    "mean_price",  # 非空价格均值
    "min_price",  # 非空价格最小
    "max_price",  # 非空价格最大
    "sales_channel_mode",  # 出现最多的渠道，打平取更小 id
    "channel_count",  # 不同渠道个数
)  # 列结束
REQUIRED_COLUMNS = ("t_dat", "customer_id", "article_id", "price", "sales_channel_id")  # raw 必需列
GROUP_KEYS = ("user_id", "item_id", "date")  # 事件主键


def _missing_mask(series: pd.Series) -> pd.Series:  # 空值、空白、<NA>
    text = series.astype("string")  # 统一成 pandas 字符串
    return text.isna() | text.str.strip().eq("") | text.eq("<NA>")  # 缺失掩码


def _require_present(series: pd.Series, name: str) -> None:  # 缺主键立即失败
    n_missing = int(_missing_mask(series).sum())  # 缺失行数
    if n_missing:  # 有缺失
        raise ValueError(f"{n_missing} rows have missing {name}")  # 不静默丢行


def aggregate_user_day_item_events(transactions: pd.DataFrame) -> pd.DataFrame:  # 纯函数：行 -> 事件
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in transactions.columns]  # 缺列
    if missing_cols:  # schema 不对
        raise ValueError(f"transactions missing columns: {missing_cols}")  # 报错
    if transactions.empty:  # 空表
        raise ValueError("transactions must not be empty")  # 无法聚合

    frame = transactions.loc[:, list(REQUIRED_COLUMNS)].copy()  # 只保留事件所需列
    _require_present(frame["customer_id"], "customer_id")  # 用户
    _require_present(frame["article_id"], "article_id")  # 商品
    frame["date"] = pd.to_datetime(frame["t_dat"], errors="coerce").dt.normalize()  # 去掉时分秒
    n_bad_dates = int(frame["date"].isna().sum())  # 空日期或无法解析
    if n_bad_dates:  # 有坏日期
        raise ValueError(f"{n_bad_dates} rows have missing or invalid t_dat")  # 报错
    frame["user_id"] = frame["customer_id"].map(canonical_user_id)  # 规范化用户
    frame["item_id"] = frame["article_id"].map(canonical_item_id)  # 十位商品 ID
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")  # 空价格允许变成 NaN
    frame["sales_channel_id"] = pd.to_numeric(frame["sales_channel_id"], errors="coerce")  # 渠道转数值
    n_bad_channel = int(frame["sales_channel_id"].isna().sum())  # 缺渠道无法算众数
    if n_bad_channel:  # 有坏渠道
        raise ValueError(f"{n_bad_channel} rows have missing or invalid sales_channel_id")  # 报错
    frame["sales_channel_id"] = frame["sales_channel_id"].astype(int)  # 渠道为整数

    keys = list(GROUP_KEYS)  # 分组键
    quantity = frame.groupby(keys, sort=False).size().rename("quantity")  # 行数即购买件数
    priced = frame.loc[frame["price"].notna(), keys + ["price"]]  # 空价格不进 min/mean/max
    if priced.empty:  # 全部价格为空
        price_stats = pd.DataFrame(  # 三个价格列都保持空
            {col: pd.Series(dtype="float64") for col in ("mean_price", "min_price", "max_price")},  # 空表
            index=quantity.index,  # 对齐事件键
        )  # 空价格统计结束
    else:  # 至少有一个有效价格
        price_stats = priced.groupby(keys)["price"].agg(  # 只在非空价格上聚合
            mean_price="mean",  # 均值
            min_price="min",  # 最小
            max_price="max",  # 最大
        )  # 价格统计结束

    channel_rows = (  # 每个事件下各渠道出现次数
        frame.groupby([*keys, "sales_channel_id"], sort=False).size().rename("n").reset_index()
    )  # 渠道计数结束
    channel_rows = channel_rows.sort_values(  # 先按次数降序，打平再按渠道 id 升序
        [*keys, "n", "sales_channel_id"],  # 排序键
        ascending=[True, True, True, False, True],  # 主键稳定，次数高优先，id 小优先
        kind="mergesort",  # 稳定排序
    )  # 排序结束
    mode = channel_rows.drop_duplicates(keys, keep="first").rename(  # 每个事件一行众数
        columns={"sales_channel_id": "sales_channel_mode"}
    )  # 众数结束
    channel_count = channel_rows.groupby(keys, sort=False)["sales_channel_id"].nunique().rename("channel_count")  # 渠道个数

    events = (  # 拼出事件表
        quantity.to_frame()  # 件数
        .join(price_stats, how="left")  # 全空价格的事件留下 NaN
        .join(channel_count, how="left")  # 渠道个数
        .reset_index()  # 主键变列
        .merge(mode[keys + ["sales_channel_mode"]], on=keys, how="left")  # 众数
    )  # 拼接结束
    events = events.sort_values(list(GROUP_KEYS), kind="mergesort").reset_index(drop=True)  # 确定顺序
    events["quantity"] = events["quantity"].astype("int64")  # 件数
    events["channel_count"] = events["channel_count"].astype("int64")  # 渠道数
    events["sales_channel_mode"] = events["sales_channel_mode"].astype("int64")  # 众数渠道
    return events.loc[:, list(EVENT_COLUMNS)]  # 固定列顺序


def write_events_parquet(events: pd.DataFrame, output_dir: Path) -> Path:  # 按月分区写出
    output_dir = Path(output_dir)  # 规范化
    if events.empty:  # 空事件表
        raise ValueError("cannot write empty events table")  # 拒绝空写出
    if output_dir.exists():  # 覆盖旧分区，避免残留月份
        shutil.rmtree(output_dir)  # 删除目录
    output_dir.mkdir(parents=True, exist_ok=True)  # 重建
    frame = events.copy()  # 不改调用方
    frame[PARTITION_COL] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m")  # 月份分区
    frame.to_parquet(  # Hive 风格 year_month=YYYY-MM
        output_dir,  # 目录
        partition_cols=[PARTITION_COL],  # 分区列
        index=False,  # 不写行号
        engine="pyarrow",  # 需要 pyarrow
    )  # 写出结束
    return output_dir  # 返回根目录


def build_events(*, transactions_path: Path, output_dir: Path) -> Path:  # 从 CSV 构建事件分区表
    transactions_path = Path(transactions_path)  # 规范化
    if not transactions_path.is_file():  # 输入不存在
        raise FileNotFoundError(f"transactions file not found: {transactions_path}")  # 报错
    transactions = pd.read_csv(  # 读交易，ID 强制字符串以免丢掉前导零
        transactions_path,  # 路径
        dtype={"customer_id": "string", "article_id": "string"},  # ID 列
    )  # 读取结束
    events = aggregate_user_day_item_events(transactions)  # 聚合
    written = write_events_parquet(events, output_dir)  # 按月写出
    print(f"saved events: {written} ({len(events):,} rows, schema {EVENT_SCHEMA_VERSION})")  # 提示
    return written  # 返回目录


def main(argv: list[str] | None = None) -> None:  # 模块级入口，供孤立调试
    parser = argparse.ArgumentParser(  # 参数
        description="Aggregate transactions into monthly-partitioned user-day-item events.",  # 说明
    )  # 解析器结束
    parser.add_argument("--transactions", type=Path, required=True)  # 输入 CSV
    parser.add_argument("--output-dir", type=Path, required=True)  # 事件目录
    args = parser.parse_args(argv)  # 解析
    build_events(transactions_path=args.transactions, output_dir=args.output_dir)  # 构建


if __name__ == "__main__":  # 直接运行
    main()  # 入口
