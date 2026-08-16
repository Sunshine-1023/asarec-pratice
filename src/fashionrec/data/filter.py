"""Create an optional train-fitted item sample over the experiment window."""  # 只用训练窗拟合商品集合，再应用到完整实验窗口

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 命令行参数解析
from pathlib import Path  # 路径对象

import pandas as pd  # 数据处理

from fashionrec.data.time import week_window_start



RAW_DIR = Path("data/raw")  # 原始数据目录
FILTERED_DIR = Path("data/raw/filtered")  # 过滤后数据输出目录

TOP_ITEMS = 30_000  # 保留的热门商品数量上限
MIN_USER_PURCHASES = 5  # 用户最少购买次数阈值
MAX_USER_BEHAVIORS = 100  # 每用户保留的最大行为条数（与序列模型 MAX_ITEM_LIST_LENGTH 对齐）
WEEKS = 6  # 时间窗口周数（与 split 的 total_weeks 一致）
VALID_WEEKS = 1  # 验证标签周数
TEST_WEEKS = 1  # 测试标签周数
CHUNK_SIZE = 500_000  # 分块读取 CSV 的行数


def _normalize_article_id(series: pd.Series) -> pd.Series:  # 将商品 ID 规范化为 10 位字符串
    return series.astype(str).str.zfill(10)  # 转字符串并左侧补零至 10 位


def _valid_window_start(  # 计算验证标签周起点，作为商品热度拟合截止点
    max_date: pd.Timestamp,
    valid_weeks: int,
    test_weeks: int,
) -> pd.Timestamp:
    max_day = pd.Timestamp(max_date).normalize()
    return max_day - pd.Timedelta(days=(valid_weeks + test_weeks) * 7 - 1)


def _select_top_item_ids(item_counts: dict[str, int], top_items: int) -> set[str]:
    if top_items < 1:
        raise ValueError("top_items must be >= 1")
    ordered = sorted(item_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return {item_id for item_id, _count in ordered[:top_items]}


def fit_train_item_universe(  # 纯函数：测试/复用时明确验证商品集合只由 train 决定
    transactions: pd.DataFrame,
    *,
    window_start: pd.Timestamp,
    valid_start: pd.Timestamp,
    top_items: int,
) -> set[str]:
    frame = transactions.copy()
    frame["t_dat"] = pd.to_datetime(frame["t_dat"])
    frame["article_id"] = _normalize_article_id(frame["article_id"])
    train = frame[
        (frame["t_dat"].dt.normalize() >= pd.Timestamp(window_start).normalize())
        & (frame["t_dat"].dt.normalize() < pd.Timestamp(valid_start).normalize())
    ]
    counts = {str(item_id): int(count) for item_id, count in train["article_id"].value_counts().items()}
    return _select_top_item_ids(counts, top_items)


def filter_transactions(  # 过滤交易记录并写入 CSV
    input_path: Path | None = None,  # 输入交易文件路径
    output_dir: Path | None = None,  # 输出目录
    top_items: int = TOP_ITEMS,  # 保留的热门商品数
    min_user_purchases: int = MIN_USER_PURCHASES,  # 用户最少购买次数
    max_user_behaviors: int = MAX_USER_BEHAVIORS,  # 每用户最大行为数
    weeks: int = WEEKS,  # 时间窗口周数
    valid_weeks: int = VALID_WEEKS,  # 验证标签周数
    test_weeks: int = TEST_WEEKS,  # 测试标签周数
) -> Path:  # 返回输出文件路径
    input_path = input_path or RAW_DIR / "transactions_train.csv"  # 默认输入路径
    output_dir = output_dir or FILTERED_DIR  # 默认输出目录
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录（含父目录）
    output_path = output_dir / "transactions_train.csv"  # 输出文件完整路径

    print(f"Reading {input_path} ...")  # 打印读取提示

    # 第一遍：确定最近 N 周的日期下界
    max_date = None  # 全局最大日期，初始为空
    for chunk in pd.read_csv(input_path, usecols=["t_dat"], chunksize=CHUNK_SIZE):  # 分块只读日期列
        chunk_max = pd.to_datetime(chunk["t_dat"]).max()  # 当前块的最大日期
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)  # 更新全局最大日期

    if max_date is None:
        raise ValueError(f"No transactions found in {input_path}")
    if weeks < 1 or valid_weeks < 1 or test_weeks < 1:
        raise ValueError("weeks, valid_weeks, and test_weeks must all be >= 1")
    if valid_weeks + test_weeks >= weeks:
        raise ValueError("valid_weeks + test_weeks must be smaller than weeks")

    cutoff = week_window_start(max_date, weeks)  # 计算时间窗口起始日期
    valid_start = _valid_window_start(max_date, valid_weeks, test_weeks)  # train 拟合截止点（不含）
    print(f"Date range: {cutoff.date()} ~ {max_date.date()} (last {weeks} weeks)")  # 打印日期范围
    print(f"Fit Top-item universe on train only: [{cutoff.date()}, {valid_start.date()})")

    # 第二遍：只统计 train 窗口内各商品购买次数；valid/test 不参与商品集合拟合
    item_counts: dict[str, int] = {}  # 商品 ID 到购买次数的映射
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):  # 分块读取完整交易
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])  # 转换日期列
        dates = chunk["t_dat"].dt.normalize()
        chunk = chunk[(dates >= cutoff) & (dates < valid_start)]  # 仅保留 train 记录
        chunk["article_id"] = _normalize_article_id(chunk["article_id"])  # 规范化商品 ID
        counts = chunk["article_id"].value_counts()  # 统计本块各商品出现次数
        for item_id, count in counts.items():  # 遍历本块商品计数
            item_counts[item_id] = item_counts.get(item_id, 0) + int(count)  # 累加到全局计数

    top_item_ids = _select_top_item_ids(item_counts, top_items)
    if not top_item_ids:
        raise ValueError("No train-window items are available for Top-item sampling")
    print(f"Top {len(top_item_ids):,} items selected (unique train items: {len(item_counts):,})")

    # 第三遍：把冻结的 train-fitted 商品集合应用到完整 train/valid/test 窗口
    filtered_chunks: list[pd.DataFrame] = []  # 存放各过滤后数据块
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):  # 分块读取交易
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])  # 转换日期列
        chunk = chunk[chunk["t_dat"].dt.normalize() >= cutoff]  # 保留窗口内记录
        chunk["article_id"] = _normalize_article_id(chunk["article_id"])  # 规范化商品 ID
        chunk = chunk[chunk["article_id"].isin(top_item_ids)]  # 只保留热门商品
        if not chunk.empty:  # 若本块非空
            filtered_chunks.append(chunk)  # 追加到列表

    if not filtered_chunks:  # 若无任何过滤结果
        pd.DataFrame(  # 创建空 DataFrame
            columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]  # 指定列名
        ).to_csv(output_path, index=False)  # 写入空 CSV
        print(f"Saved 0 transactions to {output_path}")  # 打印零条记录提示
        return output_path  # 提前返回输出路径

    df = pd.concat(filtered_chunks, ignore_index=True)  # 合并所有过滤块
    # 兼容保留 min_user_purchases/max_user_behaviors 参数，但用户资格在 split 后按 train 统计，
    # 历史长度只在构造序列或召回上下文时截断，二者都不得删除标签周用户/历史。
    _ = min_user_purchases, max_user_behaviors
    df = df.sort_values(["customer_id", "t_dat", "article_id"], kind="mergesort")
    df["t_dat"] = df["t_dat"].dt.strftime("%Y-%m-%d")  # 日期格式化为字符串
    df.to_csv(output_path, index=False)  # 写入 CSV
    print(f"Saved {len(df):,} transactions to {output_path}")  # 打印保存条数
    return output_path  # 返回输出文件路径


def filter_articles(  # 按交易中出现的商品过滤 articles.csv
    transactions_path: Path,  # 已过滤的交易文件路径
    input_path: Path | None = None,  # 原始 articles 文件路径
    output_dir: Path | None = None,  # 输出目录
) -> Path:  # 返回输出文件路径
    input_path = input_path or RAW_DIR / "articles.csv"  # 默认 articles 输入路径
    output_dir = output_dir or FILTERED_DIR  # 默认输出目录
    output_path = output_dir / "articles.csv"  # 输出文件路径

    item_ids = _normalize_article_id(  # 从交易中提取并规范化商品 ID
        pd.read_csv(transactions_path, usecols=["article_id"], dtype={"article_id": str})[  # 只读 article_id 列
            "article_id"  # 取 Series
        ]  # 结束列索引
    ).unique()  # 去重得到唯一商品 ID

    articles = pd.read_csv(input_path, dtype={"article_id": str})  # 读取全部商品表
    filtered = articles[articles["article_id"].isin(item_ids)]  # 只保留交易中出现的商品
    filtered.to_csv(output_path, index=False)  # 写入过滤结果

    print(f"Saved {len(filtered):,} articles to {output_path}")  # 打印保存条数
    return output_path  # 返回输出路径


def filter_customers(  # 按交易中出现的用户过滤 customers.csv
    transactions_path: Path,  # 已过滤的交易文件路径
    input_path: Path | None = None,  # 原始 customers 文件路径
    output_dir: Path | None = None,  # 输出目录
) -> Path:  # 返回输出文件路径
    input_path = input_path or RAW_DIR / "customers.csv"  # 默认 customers 输入路径
    output_dir = output_dir or FILTERED_DIR  # 默认输出目录
    output_path = output_dir / "customers.csv"  # 输出文件路径

    user_ids = pd.read_csv(transactions_path, usecols=["customer_id"])["customer_id"].unique()  # 提取唯一用户 ID

    customers = pd.read_csv(input_path)  # 读取全部用户表
    filtered = customers[customers["customer_id"].isin(user_ids)]  # 只保留交易中出现的用户
    filtered.to_csv(output_path, index=False)  # 写入过滤结果

    print(f"Saved {len(filtered):,} customers to {output_path}")  # 打印保存条数
    return output_path  # 返回输出路径


def run_filter(  # 依次执行交易、商品、用户三步过滤
    input_dir: Path | None = None,  # 原始数据目录
    output_dir: Path | None = None,  # 输出目录
    top_items: int = TOP_ITEMS,  # 热门商品数
    min_user_purchases: int = MIN_USER_PURCHASES,  # 用户最少购买次数
    max_user_behaviors: int = MAX_USER_BEHAVIORS,  # 每用户最大行为数
    weeks: int = WEEKS,  # 时间窗口周数
    valid_weeks: int = VALID_WEEKS,
    test_weeks: int = TEST_WEEKS,
) -> Path:  # 返回输出目录路径
    input_dir = input_dir or RAW_DIR  # 默认输入目录
    output_dir = output_dir or FILTERED_DIR  # 默认输出目录

    tx_path = filter_transactions(  # 过滤交易并获取输出路径
        input_path=input_dir / "transactions_train.csv",  # 交易输入文件
        output_dir=output_dir,  # 输出目录
        top_items=top_items,  # 热门商品数
        min_user_purchases=min_user_purchases,  # 最少购买次数
        max_user_behaviors=max_user_behaviors,  # 最大行为数
        weeks=weeks,  # 周数
        valid_weeks=valid_weeks,
        test_weeks=test_weeks,
    )  # 结束 filter_transactions 调用
    filter_articles(tx_path, input_path=input_dir / "articles.csv", output_dir=output_dir)  # 过滤商品表
    filter_customers(tx_path, input_path=input_dir / "customers.csv", output_dir=output_dir)  # 过滤用户表
    return output_dir  # 返回输出目录


def main() -> None:  # CLI 入口
    parser = argparse.ArgumentParser(description="Filter H&M dataset")  # 创建参数解析器
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)  # 输入目录参数
    parser.add_argument("--output-dir", type=Path, default=FILTERED_DIR)  # 输出目录参数
    parser.add_argument("--top-items", type=int, default=TOP_ITEMS)  # 热门商品数参数
    parser.add_argument("--min-user-purchases", type=int, default=MIN_USER_PURCHASES)  # 最少购买次数参数
    parser.add_argument("--max-user-behaviors", type=int, default=MAX_USER_BEHAVIORS)  # 最大行为数参数
    parser.add_argument("--weeks", type=int, default=WEEKS)  # 周数参数
    parser.add_argument("--valid-weeks", type=int, default=VALID_WEEKS)
    parser.add_argument("--test-weeks", type=int, default=TEST_WEEKS)
    args = parser.parse_args()  # 解析命令行参数

    run_filter(  # 执行完整过滤流程
        input_dir=args.input_dir,  # 传入输入目录
        output_dir=args.output_dir,  # 传入输出目录
        top_items=args.top_items,  # 传入热门商品数
        min_user_purchases=args.min_user_purchases,  # 传入最少购买次数
        max_user_behaviors=args.max_user_behaviors,  # 传入最大行为数
        weeks=args.weeks,  # 传入周数
        valid_weeks=args.valid_weeks,
        test_weeks=args.test_weeks,
    )  # 结束 run_filter 调用


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用 main
