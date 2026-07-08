"""Filter H&M raw data: last 3 months, top items, active users."""  # 过滤 H&M 原始数据：最近 3 个月、热门商品、活跃用户

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 命令行参数解析
from pathlib import Path  # 路径对象

import pandas as pd  # 数据处理


RAW_DIR = Path("data/raw")  # 原始数据目录
FILTERED_DIR = Path("data/raw/filtered")  # 过滤后数据输出目录

TOP_ITEMS = 30_000  # 保留的热门商品数量上限
MIN_USER_PURCHASES = 5  # 用户最少购买次数阈值
MAX_USER_BEHAVIORS = 50  # 每用户保留的最大行为条数
MONTHS = 3  # 时间窗口月数
CHUNK_SIZE = 500_000  # 分块读取 CSV 的行数


def _normalize_article_id(series: pd.Series) -> pd.Series:  # 将商品 ID 规范化为 10 位字符串
    return series.astype(str).str.zfill(10)  # 转字符串并左侧补零至 10 位


def _last_n_months_cutoff(dates: pd.Series, months: int) -> pd.Timestamp:  # 计算最近 N 个月的时间下界
    max_date = pd.to_datetime(dates).max()  # 取日期序列的最大值
    return max_date - pd.DateOffset(months=months)  # 最大日期减去 N 个月


def filter_transactions(  # 过滤交易记录并写入 CSV
    input_path: Path | None = None,  # 输入交易文件路径
    output_dir: Path | None = None,  # 输出目录
    top_items: int = TOP_ITEMS,  # 保留的热门商品数
    min_user_purchases: int = MIN_USER_PURCHASES,  # 用户最少购买次数
    max_user_behaviors: int = MAX_USER_BEHAVIORS,  # 每用户最大行为数
    months: int = MONTHS,  # 时间窗口月数
) -> Path:  # 返回输出文件路径
    input_path = input_path or RAW_DIR / "transactions_train.csv"  # 默认输入路径
    output_dir = output_dir or FILTERED_DIR  # 默认输出目录
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录（含父目录）
    output_path = output_dir / "transactions_train.csv"  # 输出文件完整路径

    print(f"Reading {input_path} ...")  # 打印读取提示

    # Pass 1: find date cutoff from last N months  # 第一遍：确定最近 N 个月的日期下界
    max_date = None  # 全局最大日期，初始为空
    for chunk in pd.read_csv(input_path, usecols=["t_dat"], chunksize=CHUNK_SIZE):  # 分块只读日期列
        chunk_max = pd.to_datetime(chunk["t_dat"]).max()  # 当前块的最大日期
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)  # 更新全局最大日期

    cutoff = max_date - pd.DateOffset(months=months)  # 计算时间窗口起始日期
    print(f"Date range: {cutoff.date()} ~ {max_date.date()} (last {months} months)")  # 打印日期范围

    # Pass 2: count item purchases in the time window  # 第二遍：统计窗口内各商品购买次数
    item_counts: dict[str, int] = {}  # 商品 ID 到购买次数的映射
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):  # 分块读取完整交易
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])  # 转换日期列
        chunk = chunk[chunk["t_dat"] >= cutoff]  # 保留窗口内记录
        chunk["article_id"] = _normalize_article_id(chunk["article_id"])  # 规范化商品 ID
        counts = chunk["article_id"].value_counts()  # 统计本块各商品出现次数
        for item_id, count in counts.items():  # 遍历本块商品计数
            item_counts[item_id] = item_counts.get(item_id, 0) + int(count)  # 累加到全局计数

    top_item_ids = {  # 取购买次数最多的 top_items 个商品 ID 集合
        str(item_id)  # 转为字符串
        for item_id in pd.Series(item_counts)  # 将计数字典转为 Series
        .sort_values(ascending=False)  # 按次数降序排列
        .head(top_items)  # 取前 top_items 个
        .index  # 取索引即商品 ID
    }
    print(f"Top {top_items} items selected (unique items in window: {len(item_counts):,})")  # 打印热门商品统计

    # Pass 3: count user purchases after item filter  # 第三遍：在商品过滤后统计用户购买次数
    user_counts: dict[str, int] = {}  # 用户 ID 到购买次数的映射
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):  # 分块读取交易
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])  # 转换日期列
        chunk = chunk[chunk["t_dat"] >= cutoff]  # 保留窗口内记录
        chunk["article_id"] = _normalize_article_id(chunk["article_id"])  # 规范化商品 ID
        chunk = chunk[chunk["article_id"].isin(top_item_ids)]  # 只保留热门商品
        counts = chunk["customer_id"].value_counts()  # 统计本块各用户出现次数
        for user_id, count in counts.items():  # 遍历本块用户计数
            user_counts[user_id] = user_counts.get(user_id, 0) + int(count)  # 累加到全局计数

    active_user_ids = {  # 购买次数达到阈值的用户 ID 集合
        user_id for user_id, count in user_counts.items() if count >= min_user_purchases  # 过滤低活跃用户
    }
    print(  # 打印活跃用户统计
        f"Active users (>={min_user_purchases} purchases): "  # 活跃用户数量前缀
        f"{len(active_user_ids):,} / {len(user_counts):,}"  # 活跃用户数 / 总用户数
    )

    # Pass 4: collect filtered transactions  # 第四遍：收集满足全部条件的交易
    filtered_chunks: list[pd.DataFrame] = []  # 存放各过滤后数据块
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):  # 分块读取交易
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])  # 转换日期列
        chunk = chunk[chunk["t_dat"] >= cutoff]  # 保留窗口内记录
        chunk["article_id"] = _normalize_article_id(chunk["article_id"])  # 规范化商品 ID
        chunk = chunk[chunk["article_id"].isin(top_item_ids)]  # 只保留热门商品
        chunk = chunk[chunk["customer_id"].isin(active_user_ids)]  # 只保留活跃用户
        if not chunk.empty:  # 若本块非空
            filtered_chunks.append(chunk)  # 追加到列表

    if not filtered_chunks:  # 若无任何过滤结果
        pd.DataFrame(  # 创建空 DataFrame
            columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]  # 指定列名
        ).to_csv(output_path, index=False)  # 写入空 CSV
        print(f"Saved 0 transactions to {output_path}")  # 打印零条记录提示
        return output_path  # 提前返回输出路径

    df = pd.concat(filtered_chunks, ignore_index=True)  # 合并所有过滤块
    before_truncate = len(df)  # 截断前的行数

    # Pass 5: keep only the most recent N behaviors per user  # 第五遍：每用户只保留最近 N 条行为
    df = df.sort_values(["customer_id", "t_dat"])  # 按用户和日期排序
    df = df.groupby("customer_id", sort=False).tail(max_user_behaviors)  # 每组取最后 max_user_behaviors 行
    df["t_dat"] = df["t_dat"].dt.strftime("%Y-%m-%d")  # 日期格式化为字符串
    df.to_csv(output_path, index=False)  # 写入 CSV

    truncated = before_truncate - len(df)  # 计算被截断删除的行数
    print(  # 打印截断统计
        f"Truncated to last {max_user_behaviors} behaviors per user "  # 截断说明前缀
        f"({truncated:,} rows removed)"  # 删除行数
    )
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
        ]
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
    months: int = MONTHS,  # 时间窗口月数
) -> Path:  # 返回输出目录路径
    input_dir = input_dir or RAW_DIR  # 默认输入目录
    output_dir = output_dir or FILTERED_DIR  # 默认输出目录

    tx_path = filter_transactions(  # 过滤交易并获取输出路径
        input_path=input_dir / "transactions_train.csv",  # 交易输入文件
        output_dir=output_dir,  # 输出目录
        top_items=top_items,  # 热门商品数
        min_user_purchases=min_user_purchases,  # 最少购买次数
        max_user_behaviors=max_user_behaviors,  # 最大行为数
        months=months,  # 月数
    )
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
    parser.add_argument("--months", type=int, default=MONTHS)  # 月数参数
    args = parser.parse_args()  # 解析命令行参数

    run_filter(  # 执行完整过滤流程
        input_dir=args.input_dir,  # 传入输入目录
        output_dir=args.output_dir,  # 传入输出目录
        top_items=args.top_items,  # 传入热门商品数
        min_user_purchases=args.min_user_purchases,  # 传入最少购买次数
        max_user_behaviors=args.max_user_behaviors,  # 传入最大行为数
        months=args.months,  # 传入月数
    )


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用 main
