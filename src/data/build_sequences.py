"""Build causal RecBole sequence samples from time-based interaction splits."""  # 序列样本属于数据层，不依赖训练入口

from __future__ import annotations  # 延迟注解

import csv  # TSV 输出
from collections import defaultdict  # 用户历史
from pathlib import Path  # 路径

import pandas as pd  # 读取和排序交互

from src.domain.ids import canonical_item_id, canonical_user_id  # 统一 ID 契约


SOURCE_DIR = Path("data/processed/hm")  # 原始划分数据目录
TARGET_DIR = Path("data/processed/hm_seq")  # 序列化数据输出目录
TRAIN_HISTORY_FILE = SOURCE_DIR / "hm.train.inter"  # valid 预测可见的完整 train 历史
TRAIN_SPLIT_FILE = SOURCE_DIR / "hm.model_train.inter"  # 模型拟合使用的 train-only 活跃用户子集
VALID_SPLIT_FILE = SOURCE_DIR / "hm.valid.inter"  # 验证集划分文件
TEST_SPLIT_FILE = SOURCE_DIR / "hm.test.inter"  # 测试集划分文件
RECB_TRAIN_FILE = TARGET_DIR / "hm_seq.train.inter"  # RecBole 训练文件
RECB_VALID_FILE = TARGET_DIR / "hm_seq.valid.inter"  # RecBole 验证文件
RECB_TEST_FILE = TARGET_DIR / "hm_seq.test.inter"  # RecBole 测试文件


def load_history_map(path: Path) -> dict[str, list[str]]:  # 从完整历史文件构建用户序列
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["user_id:token", "item_id:token", "timestamp:float"],
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    frame["user_id:token"] = frame["user_id:token"].map(canonical_user_id)
    frame["item_id:token"] = frame["item_id:token"].map(canonical_item_id)
    frame = frame.sort_values(["user_id:token", "timestamp:float", "item_id:token"], kind="mergesort")
    history: dict[str, list[str]] = defaultdict(list)
    for user_id, item_id in frame[["user_id:token", "item_id:token"]].itertuples(index=False, name=None):
        history[user_id].append(item_id)
    return history


def convert_to_sequence_samples(  # 将一个划分转换为因果序列样本
    source_path: Path,  # 源交互文件
    target_path: Path,  # 输出序列文件
    history_map: dict[str, list[str]],  # 预测时刻之前的用户历史
    max_item_list_length: int,  # 最大序列长度
    rolling_within_split: bool,  # 当前划分内是否逐行推进历史
    advance_history_after_split: bool,  # 当前划分结束后是否批量推进历史
) -> int:  # 写入样本数
    if max_item_list_length < 1:  # 无效序列长度
        raise ValueError("max_item_list_length must be >= 1")  # 尽早失败
    df = pd.read_csv(  # 读取源交互
        source_path,
        sep="\t",
        usecols=["user_id:token", "item_id:token", "timestamp:float"],
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    df["user_id:token"] = df["user_id:token"].map(canonical_user_id)  # 统一用户 ID
    df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 统一商品 ID
    df = df.sort_values(["user_id:token", "timestamp:float", "item_id:token"], kind="mergesort")  # 稳定因果顺序

    target_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录
    rows_written = 0  # 样本计数
    split_items_by_user: dict[str, list[str]] = defaultdict(list)  # 划分结束后再追加的物品
    with target_path.open("w", newline="", encoding="utf-8") as handle:  # 写出 TSV
        writer = csv.writer(handle, delimiter="\t")  # TSV writer
        writer.writerow(  # RecBole 序列 schema
            [
                "user_id:token",
                "item_id_list:token_seq",
                "item_length:float",
                "item_id:token",
                "timestamp:float",
            ]
        )
        for user_id, item_id, timestamp in df.itertuples(index=False, name=None):  # 逐交互转换
            history = history_map[user_id]  # 当前用户已有历史
            if history:  # 首条行为没有可用历史，不生成样本
                sequence = history[-max_item_list_length:]  # 截断最近历史
                writer.writerow([user_id, " ".join(sequence), len(sequence), item_id, timestamp])  # 写样本
                rows_written += 1  # 计数
            if rolling_within_split:  # 训练集按交互滚动
                history.append(item_id)
            elif advance_history_after_split:  # valid 完成后供 test 使用，但 valid 内互不泄漏
                split_items_by_user[user_id].append(item_id)

    if advance_history_after_split:  # 批量推进历史
        for user_id, items in split_items_by_user.items():
            history_map[user_id].extend(items)
    return rows_written  # 返回样本数


def prepare_recbole_benchmark_files(  # 构建 train/valid/test 三个序列文件
    max_item_list_length: int,
    train_split_file: Path = TRAIN_SPLIT_FILE,
    valid_split_file: Path = VALID_SPLIT_FILE,
    test_split_file: Path = TEST_SPLIT_FILE,
    target_dir: Path = TARGET_DIR,
    train_history_file: Path | None = None,  # 自定义 fixture 默认复用 train_split
) -> tuple[Path, Path, Path]:
    """Build benchmark files without importing or initializing RecBole."""  # 纯数据准备边界
    resolved_train_history = (
        TRAIN_HISTORY_FILE
        if train_history_file is None and train_split_file == TRAIN_SPLIT_FILE
        else (train_history_file or train_split_file)
    )
    sources = (train_split_file, valid_split_file, test_split_file, resolved_train_history)  # 模型 train、两标签周与完整历史
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run preprocessing/splitting first.")

    targets = (  # 输出路径由 target_dir 唯一决定
        target_dir / "hm_seq.train.inter",
        target_dir / "hm_seq.valid.inter",
        target_dir / "hm_seq.test.inter",
    )
    target_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    history_map: dict[str, list[str]] = defaultdict(list)  # 跨划分历史

    train_rows = convert_to_sequence_samples(  # train 内滚动
        sources[0], targets[0], history_map, max_item_list_length, True, False
    )
    history_map = defaultdict(list, load_history_map(resolved_train_history))  # valid 使用完整 train 历史
    valid_rows = convert_to_sequence_samples(  # valid 内不滚动，结束后纳入 test 历史
        sources[1], targets[1], history_map, max_item_list_length, False, True
    )
    test_rows = convert_to_sequence_samples(  # test 只消费历史，不推进
        sources[2], targets[2], history_map, max_item_list_length, False, False
    )

    for label, path, rows in zip(("train", "valid", "test"), targets, (train_rows, valid_rows, test_rows)):
        print(f"Prepared benchmark {label} file: {path} ({rows:,} rows)")
    return targets


def read_max_item_list_length(config_path: Path) -> int:  # 从 RecBole YAML 读取序列上限
    marker = "MAX_ITEM_LIST_LENGTH:"
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            value = stripped.split(":", 1)[1].split("#", 1)[0].strip()
            return int(value)
    raise ValueError("MAX_ITEM_LIST_LENGTH not found in config.")


# 旧训练脚本曾公开使用这些私有名称；迁移期保留别名。
_convert_to_seq_samples = convert_to_sequence_samples
_read_max_item_list_length = read_max_item_list_length
