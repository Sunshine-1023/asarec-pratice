"""Convert H&M transactions to RecBole ``hm.inter`` format."""  # 将 H&M 交易转换为 RecBole hm.inter 格式

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 命令行参数解析
from pathlib import Path  # 路径对象

import pandas as pd  # 数据处理


RAW_DIR = Path("data/raw")  # 原始数据目录
FILTERED_RAW_PATH = RAW_DIR / "filtered/transactions_train.csv"  # 过滤后交易文件路径
RAW_PATH = RAW_DIR / "transactions_train.csv"  # 未过滤交易文件路径
PROCESSED_DIR = Path("data/processed")  # 处理后数据目录
DATASET_DIR = PROCESSED_DIR / "hm"  # hm 数据集子目录
INTER_FILE = DATASET_DIR / "hm.inter"  # RecBole 交互文件路径

WEEKS = 6  # 时间窗口周数（与 split 的 total_weeks 一致）
MIN_USER_PURCHASES = 5  # 用户最少购买次数阈值


def _week_window_start(max_date: pd.Timestamp, weeks: int) -> pd.Timestamp:  # 计算含 max_date 的 N 周窗口起始日
    max_day = pd.Timestamp(max_date).normalize()  # 归一化到自然日
    return max_day - pd.Timedelta(days=weeks * 7 - 1)  # 含首尾共 weeks*7 天


def _default_input_path() -> Path:  # 选择默认输入交易文件路径
    # 优先使用采样数据以便快速端到端运行
    if FILTERED_RAW_PATH.exists():  # 若过滤后文件存在
        return FILTERED_RAW_PATH  # 返回过滤后路径
    return RAW_PATH  # 否则返回原始完整路径


def load_transactions(  # 加载并预处理交易为 RecBole 列格式
    path: Path | None = None,  # 输入 CSV 路径
    weeks: int = WEEKS,  # 时间窗口周数
    min_user_purchases: int = MIN_USER_PURCHASES,  # 用户最少购买次数
) -> pd.DataFrame:  # 返回含 RecBole 字段名的 DataFrame
    path = path or _default_input_path()  # 解析默认输入路径
    df = pd.read_csv(  # 读取交易 CSV
        path,  # 文件路径
        dtype={"customer_id": "string", "article_id": "string"},  # 指定 ID 列为字符串
        parse_dates=["t_dat"],  # 解析日期列
    )  # 结束 read_csv 调用
    df = df[["customer_id", "article_id", "t_dat"]]  # 只保留所需三列

    max_date = df["t_dat"].max()  # 数据最大日期
    min_date = _week_window_start(max_date, weeks)  # 计算窗口起始日期
    df = df[df["t_dat"].dt.normalize() >= min_date]  # 保留最近 weeks 周

    user_cnt = df["customer_id"].value_counts()  # 统计各用户交互次数
    valid_users = user_cnt[user_cnt >= min_user_purchases].index  # 达到阈值的用户索引
    df = df[df["customer_id"].isin(valid_users)]  # 只保留有效用户

    df = df.sort_values(["customer_id", "t_dat"])  # 按用户和时间排序
    df["timestamp"] = (df["t_dat"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)  # 转为 Unix 秒时间戳

    out = pd.DataFrame(  # 构造 RecBole 标准列名 DataFrame
        {  # 开始列名字典
            "user_id:token": df["customer_id"],  # 用户 ID 列
            "item_id:token": df["article_id"],  # 商品 ID 列
            "timestamp:float": df["timestamp"],  # 时间戳列
        }  # 结束列名字典
    )  # 结束 DataFrame 构造
    return out  # 返回结果


def build_inter_file(  # 构建并保存 hm.inter 文件
    transactions_path: Path | None = None,  # 输入交易路径
    output_path: Path | None = None,  # 输出 inter 路径
    weeks: int = WEEKS,  # 时间窗口周数
    min_user_purchases: int = MIN_USER_PURCHASES,  # 用户最少购买次数
) -> Path:  # 返回输出文件路径
    output_path = output_path or INTER_FILE  # 默认输出路径
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录

    out = load_transactions(  # 加载并转换交易
        path=transactions_path,  # 输入路径
        weeks=weeks,  # 周数
        min_user_purchases=min_user_purchases,  # 最少购买次数
    )  # 结束 load_transactions 调用
    out.to_csv(output_path, sep="\t", index=False)  # 以制表符分隔写入
    print(f"saved: {output_path}")  # 打印保存路径
    print(f"rows: {len(out):,}")  # 打印行数
    print(out.head())  # 打印前几行预览
    return output_path  # 返回输出路径


def main() -> None:  # CLI 入口
    parser = argparse.ArgumentParser(description="Build RecBole hm.inter from H&M transactions")  # 创建参数解析器
    parser.add_argument("--transactions-path", type=Path, default=None)  # 交易文件路径参数
    parser.add_argument("--output-path", type=Path, default=INTER_FILE)  # 输出路径参数
    parser.add_argument("--weeks", type=int, default=WEEKS)  # 周数参数
    parser.add_argument("--min-user-purchases", type=int, default=MIN_USER_PURCHASES)  # 最少购买次数参数
    args = parser.parse_args()  # 解析命令行参数

    build_inter_file(  # 构建 inter 文件
        transactions_path=args.transactions_path,  # 传入交易路径
        output_path=args.output_path,  # 传入输出路径
        weeks=args.weeks,  # 传入周数
        min_user_purchases=args.min_user_purchases,  # 传入最少购买次数
    )  # 结束 build_inter_file 调用


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用 main
