"""Split interactions into train / valid / test by recent-day windows."""  # 按最近天数窗口划分训练/验证/测试集

from pathlib import Path  # 路径对象

import pandas as pd  # 数据处理


PROCESSED_DIR = Path("data/processed")  # 处理后数据目录
DATASET_DIR = PROCESSED_DIR / "hm"  # hm 数据集子目录
INTER_FILE = DATASET_DIR / "hm.inter"  # 完整交互文件路径
TRAIN_INTER_FILE = DATASET_DIR / "hm.train.inter"  # 训练集交互文件路径
VALID_INTER_FILE = DATASET_DIR / "hm.valid.inter"  # 验证集交互文件路径
TEST_INTER_FILE = DATASET_DIR / "hm.test.inter"  # 测试集交互文件路径


def split_by_time(  # 按时间窗口切分交互数据
    inter_path: Path | None = None,  # 输入 hm.inter 路径
    valid_days: int = 7,  # 验证集天数
    test_days: int = 7,  # 测试集天数
    train_inter_path: Path | None = None,  # 训练集输出路径
    valid_inter_path: Path | None = None,  # 验证集输出路径
    test_inter_path: Path | None = None,  # 测试集输出路径
) -> tuple[Path, Path, Path]:  # 返回三个输出文件路径
    inter_path = inter_path or INTER_FILE  # 默认输入路径
    train_inter_path = train_inter_path or TRAIN_INTER_FILE  # 默认训练输出路径
    valid_inter_path = valid_inter_path or VALID_INTER_FILE  # 默认验证输出路径
    test_inter_path = test_inter_path or TEST_INTER_FILE  # 默认测试输出路径
    train_inter_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录

    df = pd.read_csv(inter_path, sep="\t")  # 读取交互文件
    df["datetime"] = pd.to_datetime(df["timestamp:float"], unit="s")  # 时间戳转 datetime
    df["date"] = df["datetime"].dt.floor("D")  # 向下取整到自然日

    max_date = df["date"].max()  # 数据最大日期
    test_start = max_date - pd.Timedelta(days=test_days - 1)  # 测试集起始日（含 max_date 共 test_days 天）
    valid_start = test_start - pd.Timedelta(days=valid_days)  # 验证集起始日

    train_df = df[df["date"] < valid_start]  # 训练集：验证起始日之前
    valid_df = df[(df["date"] >= valid_start) & (df["date"] < test_start)]  # 验证集：验证窗口内
    test_df = df[df["date"] >= test_start]  # 测试集：测试起始日及之后

    for split_df, output_path in (  # 遍历三个划分并写出
        (train_df, train_inter_path),  # 训练集
        (valid_df, valid_inter_path),  # 验证集
        (test_df, test_inter_path),  # 测试集
    ):
        split_df = split_df.sort_values(["user_id:token", "timestamp:float"])  # 按用户与时间排序
        split_df[["user_id:token", "item_id:token", "timestamp:float"]].to_csv(  # 只写 RecBole 三列
            output_path, sep="\t", index=False  # 制表符分隔、不写行索引
        )

    print(  # 打印日期窗口说明
        "Date windows: "  # 前缀
        f"train < {valid_start.date()}, "  # 训练集上界
        f"valid [{valid_start.date()}, {(test_start - pd.Timedelta(days=1)).date()}], "  # 验证集区间
        f"test [{test_start.date()}, {max_date.date()}]"  # 测试集区间
    )
    print(  # 打印各划分行数
        f"Rows - train: {len(train_df):,}, valid: {len(valid_df):,}, test: {len(test_df):,}"  # 行数统计
    )
    print(  # 打印各划分用户数
        f"Users - train: {train_df['user_id:token'].nunique():,}, "  # 训练用户数
        f"valid: {valid_df['user_id:token'].nunique():,}, "  # 验证用户数
        f"test: {test_df['user_id:token'].nunique():,}"  # 测试用户数
    )
    return train_inter_path, valid_inter_path, test_inter_path  # 返回三个输出路径


if __name__ == "__main__":  # 脚本直接运行时
    split_by_time()  # 使用默认参数执行划分
