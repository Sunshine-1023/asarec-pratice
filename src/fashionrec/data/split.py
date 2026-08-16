"""Split interactions into train / valid / test by recent-week windows."""  # 按最近周窗口划分训练/验证/测试集

from __future__ import annotations  # 启用延迟注解

from dataclasses import dataclass  # 切分结果
from pathlib import Path  # 路径对象

import pandas as pd  # 数据处理

from fashionrec.domain.ids import canonical_item_id, canonical_user_id  # 统一 ID 契约
from fashionrec.data.time import week_window_start


PROCESSED_DIR = Path("data/processed")  # 处理后数据目录
DATASET_DIR = PROCESSED_DIR / "hm"  # hm 数据集子目录
INTER_FILE = DATASET_DIR / "hm.inter"  # 完整交互文件路径
TRAIN_INTER_FILE = DATASET_DIR / "hm.train.inter"  # 训练集交互文件路径
MODEL_TRAIN_INTER_FILE = DATASET_DIR / "hm.model_train.inter"  # 模型拟合专用的 train 活跃用户子集
VALID_INTER_FILE = DATASET_DIR / "hm.valid.inter"  # 验证集交互文件路径
TEST_INTER_FILE = DATASET_DIR / "hm.test.inter"  # 测试集交互文件路径

TOTAL_WEEKS = 6  # 总数据窗口（周）
TRAIN_WEEKS = 4  # 训练集周数
VALID_WEEKS = 1  # 验证集周数（倒数第二周）
TEST_WEEKS = 1  # 测试集周数（最后一周）
INTERACTION_SORT_COLUMNS = ("user_id:token", "timestamp:float", "item_id:token")  # 确定性排序键，不依赖原始行序


@dataclass(frozen=True)  # 不可变切分结果
class TimeSplitResult:  # 时间切分输出
    train_path: Path  # 训练集路径
    valid_path: Path  # 验证集路径
    test_path: Path  # 测试集路径
    window_start: pd.Timestamp  # 总窗口起始日
    max_date: pd.Timestamp  # 数据最大日期
    train_end: pd.Timestamp  # 训练集最后一天（含）
    valid_start: pd.Timestamp  # 验证集起始日
    valid_end: pd.Timestamp  # 验证集最后一天（含）
    test_start: pd.Timestamp  # 测试集起始日
    test_end: pd.Timestamp  # 测试集最后一天（含）


def sort_interactions(df: pd.DataFrame) -> pd.DataFrame:  # 按用户、时间、商品 ID 稳定排序
    missing = [col for col in INTERACTION_SORT_COLUMNS if col not in df.columns]  # 检查排序列
    if missing:  # 缺列
        raise KeyError(f"sort_interactions missing columns: {missing}")  # 无法确定排序
    ordered = df.copy()  # 避免修改调用方 DataFrame
    ordered["user_id:token"] = ordered["user_id:token"].map(canonical_user_id)  # 用户 ID 统一为字符串
    ordered["item_id:token"] = ordered["item_id:token"].map(canonical_item_id)  # 商品 ID 统一为十位字符串
    ordered["timestamp:float"] = ordered["timestamp:float"].astype(float)  # 时间戳统一为浮点
    return ordered.sort_values(list(INTERACTION_SORT_COLUMNS), kind="mergesort").reset_index(drop=True)  # 稳定排序


def history_paths_for_eval(  # 评估划分允许使用的历史交互路径
    eval_split: str,  # valid 或 test
    train_path: Path | None = None,  # 训练集路径
    valid_path: Path | None = None,  # 验证集路径
) -> list[Path]:  # 返回只含预测时刻之前数据的路径
    if eval_split not in {"valid", "test"}:  # 非法划分
        raise ValueError("eval_split must be 'valid' or 'test'")  # 抛出错误
    train_path = train_path or TRAIN_INTER_FILE  # 默认训练路径
    valid_path = valid_path or VALID_INTER_FILE  # 默认验证路径
    if eval_split == "valid":  # 验证评估
        return [train_path]  # 只能用 train
    return [train_path, valid_path]  # 测试评估可用 train+valid，不能用 test


def assert_history_paths_allowed(  # 检查召回索引路径没有混入标签周
    eval_split: str,  # valid 或 test
    history_paths: list[Path],  # 实际用于建索引的路径
    train_path: Path | None = None,  # 训练集路径
    valid_path: Path | None = None,  # 验证集路径
    test_path: Path | None = None,  # 测试集路径
) -> None:  # 违规则抛出
    allowed = {path.resolve() for path in history_paths_for_eval(eval_split, train_path, valid_path)}  # 允许的历史路径
    forbidden = {  # 标签周及之后的路径
        (test_path or TEST_INTER_FILE).resolve(),  # test 永远不能进历史
    }  # 禁止路径
    if eval_split == "valid":  # 验证评估时 valid 本身也是标签周
        forbidden.add((valid_path or VALID_INTER_FILE).resolve())  # 禁止 valid
    resolved = [Path(path).resolve() for path in history_paths]  # 规范化实际路径
    leaked = [path for path in resolved if path in forbidden]  # 混入的未来文件
    unknown = [path for path in resolved if path not in allowed]  # 不在允许集合中的路径
    if leaked or unknown:  # 有泄漏或不允许的来源
        raise AssertionError(  # 测试与运行时都应失败
            f"eval_split={eval_split} history paths leak future data: leaked={leaked} unknown={unknown}"
        )  # 错误信息


def validate_time_split(  # 断言 train/valid/test 时间严格递增且无重叠
    train_df: pd.DataFrame,  # 训练集
    valid_df: pd.DataFrame,  # 验证集
    test_df: pd.DataFrame,  # 测试集
    timestamp_col: str = "timestamp:float",  # 时间戳列
) -> None:  # 违规则抛出
    if train_df.empty or valid_df.empty or test_df.empty:  # 任一划分空
        raise ValueError("train/valid/test splits must be non-empty")  # 无法验证因果性
    train_max = float(train_df[timestamp_col].max())  # 训练最大时间
    valid_min = float(valid_df[timestamp_col].min())  # 验证最小时间
    valid_max = float(valid_df[timestamp_col].max())  # 验证最大时间
    test_min = float(test_df[timestamp_col].min())  # 测试最小时间
    if not train_max < valid_min:  # 训练与验证重叠或倒置
        raise AssertionError(  # 必须严格小于
            f"max(train.timestamp)={train_max} is not < min(valid.timestamp)={valid_min}"
        )  # 错误信息
    if not valid_max < test_min:  # 验证与测试重叠或倒置
        raise AssertionError(  # 必须严格小于
            f"max(valid.timestamp)={valid_max} is not < min(test.timestamp)={test_min}"
        )  # 错误信息


def filter_interactions_as_of(  # 只保留预测时刻之前的交互
    df: pd.DataFrame,  # 交互表
    as_of_ts: float,  # 预测时刻（不含）
    timestamp_col: str = "timestamp:float",  # 时间戳列
) -> pd.DataFrame:  # 返回历史交互
    return df[df[timestamp_col] < as_of_ts].copy()  # 严格早于 as_of


def item_popularity_as_of(  # 预测时刻之前的商品热度，禁止读标签周
    df: pd.DataFrame,  # 交互表
    as_of_ts: float,  # 预测时刻
    item_col: str = "item_id:token",  # 商品列
    timestamp_col: str = "timestamp:float",  # 时间戳列
) -> dict[str, int]:  # 商品到购买次数
    hist = filter_interactions_as_of(df, as_of_ts, timestamp_col=timestamp_col)  # 截断到 as_of 之前
    if hist.empty:  # 无历史
        return {}  # 空热度
    counts = hist[item_col].map(canonical_item_id).value_counts()  # 统一 ID 后统计次数
    return {canonical_item_id(item): int(count) for item, count in counts.items()}  # 转为普通字典


def user_item_counts_as_of(  # 预测时刻之前的用户-商品购买次数
    df: pd.DataFrame,  # 交互表
    as_of_ts: float,  # 预测时刻
    user_col: str = "user_id:token",  # 用户列
    item_col: str = "item_id:token",  # 商品列
    timestamp_col: str = "timestamp:float",  # 时间戳列
) -> dict[tuple[str, str], int]:  # (用户, 商品) -> 次数
    hist = filter_interactions_as_of(df, as_of_ts, timestamp_col=timestamp_col)  # 截断历史
    if hist.empty:  # 无历史
        return {}  # 空偏好
    normalized = hist.copy()  # 避免修改调用方数据
    normalized[user_col] = normalized[user_col].map(canonical_user_id)  # 统一用户 ID
    normalized[item_col] = normalized[item_col].map(canonical_item_id)  # 统一商品 ID
    grouped = normalized.groupby([user_col, item_col]).size()  # 交叉计数
    return {(canonical_user_id(user), canonical_item_id(item)): int(count) for (user, item), count in grouped.items()}  # 普通字典


def split_bounds_dict(result: TimeSplitResult) -> dict[str, str]:  # 将切分边界转为可写入 manifest 的字符串
    return {  # ISO 日期
        "window_start": str(result.window_start.date()),  # 总窗口起
        "max_date": str(result.max_date.date()),  # 最大日期
        "train_end": str(result.train_end.date()),  # 训练止
        "valid_start": str(result.valid_start.date()),  # 验证起
        "valid_end": str(result.valid_end.date()),  # 验证止
        "test_start": str(result.test_start.date()),  # 测试起
        "test_end": str(result.test_end.date()),  # 测试止
    }  # 边界字典结束


def build_model_train_split(  # 只使用 train 统计活跃用户
    train_path: Path = TRAIN_INTER_FILE,
    output_path: Path = MODEL_TRAIN_INTER_FILE,
    min_user_purchases: int = 5,
) -> Path:
    if min_user_purchases < 1:
        raise ValueError("min_user_purchases must be >= 1")
    train = pd.read_csv(
        train_path,
        sep="\t",
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    train = sort_interactions(train)
    counts = train["user_id:token"].value_counts()
    eligible_users = set(counts[counts >= min_user_purchases].index)
    model_train = train[train["user_id:token"].isin(eligible_users)].copy()
    if model_train.empty:
        raise ValueError(
            f"No model-train interactions remain with min_user_purchases={min_user_purchases}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_train[["user_id:token", "item_id:token", "timestamp:float"]].to_csv(
        output_path,
        sep="\t",
        index=False,
    )
    print(
        f"Model train rows: {len(model_train):,}/{len(train):,}; "
        f"eligible users: {len(eligible_users):,} (train-only threshold >= {min_user_purchases})"
    )
    return output_path


def split_by_time(  # 按时间窗口切分交互数据
    inter_path: Path | None = None,  # 输入 hm.inter 路径
    total_weeks: int = TOTAL_WEEKS,  # 总数据窗口（周）
    train_weeks: int = TRAIN_WEEKS,  # 训练集周数
    valid_weeks: int = VALID_WEEKS,  # 验证集周数
    test_weeks: int = TEST_WEEKS,  # 测试集周数
    train_inter_path: Path | None = None,  # 训练集输出路径
    valid_inter_path: Path | None = None,  # 验证集输出路径
    test_inter_path: Path | None = None,  # 测试集输出路径
) -> TimeSplitResult:  # 返回路径与时间边界
    if train_weeks + valid_weeks + test_weeks != total_weeks:  # 校验周数之和
        raise ValueError(  # 参数不合法
            f"train_weeks ({train_weeks}) + valid_weeks ({valid_weeks}) + "  # 错误信息前半段
            f"test_weeks ({test_weeks}) must equal total_weeks ({total_weeks})"  # 错误信息后半段
        )  # 结束 ValueError 抛出

    inter_path = inter_path or INTER_FILE  # 默认输入路径
    train_inter_path = train_inter_path or TRAIN_INTER_FILE  # 默认训练输出路径
    valid_inter_path = valid_inter_path or VALID_INTER_FILE  # 默认验证输出路径
    test_inter_path = test_inter_path or TEST_INTER_FILE  # 默认测试输出路径
    train_inter_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录

    df = pd.read_csv(  # 读取交互文件并保留 ID 文本
        inter_path,
        sep="\t",
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    df["user_id:token"] = df["user_id:token"].map(canonical_user_id)  # 入口统一用户 ID
    df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 入口统一商品 ID
    df["datetime"] = pd.to_datetime(df["timestamp:float"], unit="s")  # 时间戳转 datetime
    df["date"] = df["datetime"].dt.floor("D")  # 向下取整到自然日

    max_date = df["date"].max()  # 数据最大日期
    window_start = week_window_start(max_date, total_weeks)  # 总窗口起始日
    df = df[df["date"] >= window_start]  # 只保留最近 total_weeks 周

    test_start = max_date - pd.Timedelta(days=test_weeks * 7 - 1)  # 测试集起始日（最后 1 周）
    valid_start = test_start - pd.Timedelta(days=valid_weeks * 7)  # 验证集起始日（倒数第 2 周）
    train_end = valid_start - pd.Timedelta(days=1)  # 训练集最后一天
    valid_end = test_start - pd.Timedelta(days=1)  # 验证集最后一天

    train_df = df[df["date"] < valid_start]  # 训练集：前 4 周
    valid_df = df[(df["date"] >= valid_start) & (df["date"] < test_start)]  # 验证集：倒数第 2 周
    test_df = df[df["date"] >= test_start]  # 测试集：最后 1 周
    validate_time_split(train_df, valid_df, test_df)  # 切分后立即检查无重叠

    for split_df, output_path in (  # 遍历三个划分并写出
        (train_df, train_inter_path),  # 训练集
        (valid_df, valid_inter_path),  # 验证集
        (test_df, test_inter_path),  # 测试集
    ):  # 开始遍历三个划分
        split_df = sort_interactions(split_df)  # 确定性排序，不依赖原始行顺序
        split_df[["user_id:token", "item_id:token", "timestamp:float"]].to_csv(  # 只写 RecBole 三列
            output_path, sep="\t", index=False  # 制表符分隔、不写行索引
        )  # 结束 to_csv 写入

    print(  # 打印日期窗口说明
        f"Data window: [{window_start.date()}, {max_date.date()}] ({total_weeks} weeks); "  # 总窗口
        f"train [{window_start.date()}, {train_end.date()}] ({train_weeks}w), "  # 训练区间
        f"valid [{valid_start.date()}, {valid_end.date()}] ({valid_weeks}w), "  # 验证区间
        f"test [{test_start.date()}, {max_date.date()}] ({test_weeks}w)"  # 测试区间
    )  # 结束日期窗口说明打印
    print(  # 打印各划分行数
        f"Rows - train: {len(train_df):,}, valid: {len(valid_df):,}, test: {len(test_df):,}"  # 行数统计
    )  # 结束行数统计打印
    print(  # 打印各划分用户数
        f"Users - train: {train_df['user_id:token'].nunique():,}, "  # 训练用户数
        f"valid: {valid_df['user_id:token'].nunique():,}, "  # 验证用户数
        f"test: {test_df['user_id:token'].nunique():,}"  # 测试用户数
    )  # 结束用户数统计打印
    return TimeSplitResult(  # 返回路径与边界
        train_path=train_inter_path,  # 训练路径
        valid_path=valid_inter_path,  # 验证路径
        test_path=test_inter_path,  # 测试路径
        window_start=window_start,  # 总窗口起
        max_date=max_date,  # 最大日期
        train_end=train_end,  # 训练止
        valid_start=valid_start,  # 验证起
        valid_end=valid_end,  # 验证止
        test_start=test_start,  # 测试起
        test_end=max_date,  # 测试止
    )  # 结果结束


if __name__ == "__main__":  # 脚本直接运行时
    split_by_time()  # 使用默认参数执行划分
