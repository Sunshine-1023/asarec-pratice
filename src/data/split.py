"""Split interactions into train / valid / test by recent-week windows."""  # 按最近周窗口划分训练/验证/测试集

from pathlib import Path  # 路径对象

import pandas as pd  # 数据处理


PROCESSED_DIR = Path("data/processed")  # 处理后数据目录
DATASET_DIR = PROCESSED_DIR / "hm"  # hm 数据集子目录
INTER_FILE = DATASET_DIR / "hm.inter"  # 完整交互文件路径
TRAIN_INTER_FILE = DATASET_DIR / "hm.train.inter"  # 训练集交互文件路径
VALID_INTER_FILE = DATASET_DIR / "hm.valid.inter"  # 验证集交互文件路径
TEST_INTER_FILE = DATASET_DIR / "hm.test.inter"  # 测试集交互文件路径

TOTAL_WEEKS = 6  # 总数据窗口（周）
TRAIN_WEEKS = 4  # 训练集周数
VALID_WEEKS = 1  # 验证集周数（倒数第二周）
TEST_WEEKS = 1  # 测试集周数（最后一周）


def _week_window_start(max_date: pd.Timestamp, weeks: int) -> pd.Timestamp:  # 计算含 max_date 的 N 周窗口起始日
    max_day = pd.Timestamp(max_date).normalize()  # 归一化到自然日
    return max_day - pd.Timedelta(days=weeks * 7 - 1)  # 含首尾共 weeks*7 天


def split_by_time(  # 按时间窗口切分交互数据
    inter_path: Path | None = None,  # 输入 hm.inter 路径
    total_weeks: int = TOTAL_WEEKS,  # 总数据窗口（周）
    train_weeks: int = TRAIN_WEEKS,  # 训练集周数
    valid_weeks: int = VALID_WEEKS,  # 验证集周数
    test_weeks: int = TEST_WEEKS,  # 测试集周数
    train_inter_path: Path | None = None,  # 训练集输出路径
    valid_inter_path: Path | None = None,  # 验证集输出路径
    test_inter_path: Path | None = None,  # 测试集输出路径
) -> tuple[Path, Path, Path]:  # 返回三个输出文件路径
    if train_weeks + valid_weeks + test_weeks != total_weeks:  # 校验周数之和
        raise ValueError(  # 参数不合法
            f"train_weeks ({train_weeks}) + valid_weeks ({valid_weeks}) + "
            f"test_weeks ({test_weeks}) must equal total_weeks ({total_weeks})"
        )

    inter_path = inter_path or INTER_FILE  # 默认输入路径
    train_inter_path = train_inter_path or TRAIN_INTER_FILE  # 默认训练输出路径
    valid_inter_path = valid_inter_path or VALID_INTER_FILE  # 默认验证输出路径
    test_inter_path = test_inter_path or TEST_INTER_FILE  # 默认测试输出路径
    train_inter_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录

    df = pd.read_csv(inter_path, sep="\t")  # 读取交互文件
    df["datetime"] = pd.to_datetime(df["timestamp:float"], unit="s")  # 时间戳转 datetime
    df["date"] = df["datetime"].dt.floor("D")  # 向下取整到自然日

    max_date = df["date"].max()  # 数据最大日期
    window_start = _week_window_start(max_date, total_weeks)  # 6 周窗口起始日
    df = df[df["date"] >= window_start]  # 只保留最近 total_weeks 周

    test_start = max_date - pd.Timedelta(days=test_weeks * 7 - 1)  # 测试集起始日（最后 1 周）
    valid_start = test_start - pd.Timedelta(days=valid_weeks * 7)  # 验证集起始日（倒数第 2 周）

    train_df = df[df["date"] < valid_start]  # 训练集：前 4 周
    valid_df = df[(df["date"] >= valid_start) & (df["date"] < test_start)]  # 验证集：倒数第 2 周
    test_df = df[df["date"] >= test_start]  # 测试集：最后 1 周

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
        f"Data window: [{window_start.date()}, {max_date.date()}] ({total_weeks} weeks); "  # 总窗口
        f"train [{window_start.date()}, {(valid_start - pd.Timedelta(days=1)).date()}] ({train_weeks}w), "  # 训练区间
        f"valid [{valid_start.date()}, {(test_start - pd.Timedelta(days=1)).date()}] ({valid_weeks}w), "  # 验证区间
        f"test [{test_start.date()}, {max_date.date()}] ({test_weeks}w)"  # 测试区间
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
