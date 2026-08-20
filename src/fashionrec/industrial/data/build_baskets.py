"""Group user-day-item events into unordered daily baskets."""  # 按天购物篮：同日商品是集合，不是序列

from __future__ import annotations  # 延迟注解

import argparse  # CLI
import shutil  # 覆盖旧分区
from pathlib import Path  # 路径

import pandas as pd  # 聚合

from fashionrec.industrial.data.build_events import aggregate_user_day_item_events  # 无事件文件时从交易现算
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id  # ID 规范


BASKET_SCHEMA_VERSION = "hm.daily_basket.v1"  # 购物篮语义版本
PARTITION_COL = "year_month"  # 按月分区
BASKET_COLUMNS = (  # 写出列
    "user_id",  # 用户
    "date",  # 购物日
    "item_ids",  # 去重 SKU，空格连接；排序只为可复现，不表示购买先后
    "n_items",  # 去重商品数
    "quantity_sum",  # 原始行数之和
)  # 列结束


def _unique_sorted_item_ids(values) -> str:  # 日内商品集合的确定序列化
    items = sorted({canonical_item_id(value) for value in values})  # 去重后按 ID 排序，避免文件行序噪音
    return " ".join(items)  # RecBole 风格的 token 序列字符串


def flatten_recent_baskets(  # 把完整购物日拼成模型可用的 item 序列
    baskets: list[list[str]],  # 从旧到新的购物日，每个元素是当日去重商品
    *,
    max_item_list_length: int,  # RecBole 序列上限
    max_shopping_days: int | None = None,  # 最多保留最近 N 个购物日；None 表示不额外按日截断
) -> list[str]:  # 展平后的历史商品
    if max_item_list_length < 1:  # 非法上限
        raise ValueError("max_item_list_length must be >= 1")  # 尽早失败
    selected = list(baskets)  # 拷贝，避免改调用方
    if max_shopping_days is not None:  # 先按购物日个数截断
        if max_shopping_days < 1:  # 非法
            raise ValueError("max_shopping_days must be >= 1")  # 报错
        selected = selected[-max_shopping_days:]  # 只留最近 N 日
    while selected and sum(len(day) for day in selected) > max_item_list_length:  # 超长则丢掉最旧的完整一天
        if len(selected) == 1:  # 单日已经超过上限，只能截这一天
            return list(selected[0][:max_item_list_length])  # 保序截断；这一天内部仍无因果
        selected = selected[1:]  # 丢掉最旧购物日
    flattened: list[str] = []  # 从旧到新拼接
    for day_items in selected:  # 逐日
        flattened.extend(day_items)  # 日内集合已在构建时定序
    return flattened  # 返回


def baskets_from_events(events: pd.DataFrame) -> pd.DataFrame:  # 事件表 -> 按天购物篮
    required = {"user_id", "item_id", "date", "quantity"}  # 最少列
    missing = required.difference(events.columns)  # 缺列
    if missing:  # schema 不对
        raise ValueError(f"events missing columns: {sorted(missing)}")  # 报错
    if events.empty:  # 空表
        raise ValueError("events must not be empty")  # 无法成篮

    frame = events.copy()  # 不改调用方
    frame["user_id"] = frame["user_id"].map(canonical_user_id)  # 规范化用户
    frame["item_id"] = frame["item_id"].map(canonical_item_id)  # 十位商品
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()  # 自然日
    grouped = frame.groupby(["user_id", "date"], sort=True, observed=True)  # 一用户一日一篮
    baskets = grouped.agg(  # 聚合
        item_ids=("item_id", _unique_sorted_item_ids),  # 去重 SKU
        n_items=("item_id", "nunique"),  # 商品数
        quantity_sum=("quantity", "sum"),  # 件数
    ).reset_index()  # 主键变列
    baskets["n_items"] = baskets["n_items"].astype("int64")  # 整数
    baskets["quantity_sum"] = baskets["quantity_sum"].astype("int64")  # 整数
    return baskets.loc[:, list(BASKET_COLUMNS)]  # 固定列


def baskets_from_interactions(interactions: pd.DataFrame) -> pd.DataFrame:  # RecBole 交互 -> 购物篮
    required = {"user_id:token", "item_id:token", "timestamp:float"}  # inter schema
    missing = required.difference(interactions.columns)  # 缺列
    if missing:  # schema 不对
        raise ValueError(f"interactions missing columns: {sorted(missing)}")  # 报错
    if interactions.empty:  # 空表
        raise ValueError("interactions must not be empty")  # 无法成篮

    frame = interactions.loc[:, list(required)].copy()  # 只留三列
    frame["user_id"] = frame["user_id:token"].map(canonical_user_id)  # 用户
    frame["item_id"] = frame["item_id:token"].map(canonical_item_id)  # 商品
    frame["date"] = pd.to_datetime(frame["timestamp:float"], unit="s").dt.normalize()  # 用日历日而不是精确秒
    frame["quantity"] = 1  # 一行算一件；同日同 SKU 多行会在 quantity_sum 里累加
    return baskets_from_events(frame.loc[:, ["user_id", "item_id", "date", "quantity"]])  # 复用事件聚合


def write_baskets_parquet(baskets: pd.DataFrame, output_dir: Path) -> Path:  # 按月分区写出
    output_dir = Path(output_dir)  # 规范化
    if baskets.empty:  # 空表
        raise ValueError("cannot write empty baskets table")  # 拒绝
    if output_dir.exists():  # 清掉旧月份
        shutil.rmtree(output_dir)  # 删除
    output_dir.mkdir(parents=True, exist_ok=True)  # 重建
    frame = baskets.copy()  # 不改调用方
    frame[PARTITION_COL] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m")  # 月份
    frame.to_parquet(  # Hive 分区
        output_dir,  # 目录
        partition_cols=[PARTITION_COL],  # 分区列
        index=False,  # 不写行号
        engine="pyarrow",  # parquet
    )  # 写出结束
    return output_dir  # 返回


def build_baskets(  # 从事件目录或原始交易构建购物篮
    *,
    output_dir: Path,  # 写出目录
    events_dir: Path | None = None,  # 已有事件 parquet
    transactions_path: Path | None = None,  # 没有事件文件时从 CSV 现算
) -> Path:  # 返回购物篮目录
    if events_dir is None and transactions_path is None:  # 两个输入都没有
        raise ValueError("build_baskets requires events_dir or transactions_path")  # 报错
    if events_dir is not None:  # 优先读本 run 刚写出的事件
        events_dir = Path(events_dir)  # 规范化
        if not events_dir.exists():  # 目录不存在
            raise FileNotFoundError(f"events dir not found: {events_dir}")  # 报错
        events = pd.read_parquet(events_dir)  # 回读分区表
    else:  # 从交易现算事件，不强制落盘子
        transactions_path = Path(transactions_path)  # 规范化
        if not transactions_path.is_file():  # 文件不存在
            raise FileNotFoundError(f"transactions file not found: {transactions_path}")  # 报错
        transactions = pd.read_csv(  # 读 CSV
            transactions_path,  # 路径
            dtype={"customer_id": "string", "article_id": "string"},  # 保前导零
        )  # 读取结束
        events = aggregate_user_day_item_events(transactions)  # 先变成事件
    baskets = baskets_from_events(events)  # 按天成篮
    written = write_baskets_parquet(baskets, output_dir)  # 按月写出
    print(f"saved baskets: {written} ({len(baskets):,} rows, schema {BASKET_SCHEMA_VERSION})")  # 提示
    return written  # 返回


def main(argv: list[str] | None = None) -> None:  # 模块入口
    parser = argparse.ArgumentParser(description="Build monthly-partitioned daily baskets.")  # 解析器
    parser.add_argument("--output-dir", type=Path, required=True)  # 输出
    parser.add_argument("--events-dir", type=Path, default=None)  # 事件目录
    parser.add_argument("--transactions", type=Path, default=None)  # 交易 CSV
    args = parser.parse_args(argv)  # 解析
    build_baskets(  # 构建
        output_dir=args.output_dir,  # 输出
        events_dir=args.events_dir,  # 事件
        transactions_path=args.transactions,  # 交易
    )  # 结束


if __name__ == "__main__":  # 直接运行
    main()  # 入口
