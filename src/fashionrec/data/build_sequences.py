"""Build causal RecBole sequence samples from time-based interaction splits."""  # 序列样本按购物日推进，同日目标共享历史

from __future__ import annotations  # 延迟注解

import csv  # TSV 输出
from collections import defaultdict  # 用户历史
from pathlib import Path  # 路径

import pandas as pd  # 读取和排序交互

from fashionrec.data.build_baskets import flatten_recent_baskets  # 按完整购物日截断历史
from fashionrec.domain.ids import canonical_item_id, canonical_user_id  # 统一 ID 契约


SOURCE_DIR = Path("data/processed/hm")  # 原始划分数据目录
TARGET_DIR = Path("data/processed/hm_seq")  # 序列化数据输出目录
TRAIN_HISTORY_FILE = SOURCE_DIR / "hm.train.inter"  # valid 预测可见的完整 train 历史
TRAIN_SPLIT_FILE = SOURCE_DIR / "hm.model_train.inter"  # 模型拟合使用的 train-only 活跃用户子集
VALID_SPLIT_FILE = SOURCE_DIR / "hm.valid.inter"  # 验证集划分文件
TEST_SPLIT_FILE = SOURCE_DIR / "hm.test.inter"  # 测试集划分文件
RECB_TRAIN_FILE = TARGET_DIR / "hm_seq.train.inter"  # RecBole 训练文件
RECB_VALID_FILE = TARGET_DIR / "hm_seq.valid.inter"  # RecBole 验证文件
RECB_TEST_FILE = TARGET_DIR / "hm_seq.test.inter"  # RecBole 测试文件


BasketHistory = dict[str, list[list[str]]]  # 用户 -> 从旧到新的购物日商品集合


def _add_date_column(frame: pd.DataFrame) -> pd.DataFrame:  # 时间戳落到日历日，同日不同秒仍同一篮
    out = frame.copy()  # 拷贝
    out["user_id:token"] = out["user_id:token"].map(canonical_user_id)  # 用户
    out["item_id:token"] = out["item_id:token"].map(canonical_item_id)  # 商品
    out["date"] = pd.to_datetime(out["timestamp:float"], unit="s").dt.normalize()  # 自然日
    out = out.drop_duplicates(["user_id:token", "date", "item_id:token"], keep="first")  # 同日同 SKU 只留一件
    return out  # 返回


def load_history_map(path: Path) -> BasketHistory:  # 从完整历史文件构建按日购物篮历史
    frame = pd.read_csv(  # 读交互
        path,
        sep="\t",
        usecols=["user_id:token", "item_id:token", "timestamp:float"],
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    frame = _add_date_column(frame)  # 规范化并去重
    history: BasketHistory = defaultdict(list)  # 用户购物日
    for user_id in sorted(frame["user_id:token"].unique()):  # 用户顺序稳定
        user_df = frame[frame["user_id:token"] == user_id]  # 该用户
        for _date, day_df in user_df.groupby("date", sort=True):  # 按日从旧到新
            items = sorted(day_df["item_id:token"].unique())  # 日内无先后，排序只为可复现
            history[user_id].append(items)  # 追加完整一天
    return history  # 返回


def convert_to_sequence_samples(  # 将一个划分转换为按日因果序列样本
    source_path: Path,  # 源交互文件
    target_path: Path,  # 输出序列文件
    history_map: BasketHistory,  # 预测日之前的用户购物篮
    max_item_list_length: int,  # 最大展平序列长度
    rolling_within_split: bool,  # 当前划分内是否按购物日推进历史
    advance_history_after_split: bool,  # 当前划分结束后是否批量并入后续历史
    max_shopping_days: int | None = None,  # 最多保留最近 N 个购物日
) -> int:  # 写入样本数
    if max_item_list_length < 1:  # 无效序列长度
        raise ValueError("max_item_list_length must be >= 1")  # 尽早失败
    df = pd.read_csv(  # 读取源交互
        source_path,
        sep="\t",
        usecols=["user_id:token", "item_id:token", "timestamp:float"],
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    df = _add_date_column(df)  # 按日历日成篮，去掉同日同 SKU 重复

    target_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录
    rows_written = 0  # 样本计数
    split_baskets_by_user: dict[str, list[list[str]]] = defaultdict(list)  # 划分结束后再追加的购物日
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
        for user_id in sorted(df["user_id:token"].unique()):  # 用户稳定顺序
            user_df = df[df["user_id:token"] == user_id]  # 该用户
            for _date, day_df in user_df.groupby("date", sort=True):  # 按购物日推进
                items = sorted(day_df["item_id:token"].unique())  # 当日目标集合
                timestamp = float(day_df["timestamp:float"].min())  # 当日时间戳，同日样本共用
                flattened = flatten_recent_baskets(  # 只用这一天之前的完整购物日
                    history_map[user_id],  # 已完成的历史篮
                    max_item_list_length=max_item_list_length,  # 展平上限
                    max_shopping_days=max_shopping_days,  # 购物日上限
                )
                if flattened:  # 没有历史的首个购物日不生成样本
                    sequence = " ".join(flattened)  # 同日所有目标共享这份历史
                    length = len(flattened)  # 长度
                    for item_id in items:  # 每个目标一行，历史完全相同
                        writer.writerow([user_id, sequence, length, item_id, timestamp])  # 写样本
                        rows_written += 1  # 计数
                if rolling_within_split:  # 训练集：过完这一天，下一日才看得到 A/B/C
                    history_map[user_id].append(list(items))  # 追加完整一天
                elif advance_history_after_split:  # valid：周内不推进，整周结束后给 test
                    split_baskets_by_user[user_id].append(list(items))  # 记下当日篮子

    if advance_history_after_split:  # 批量推进历史
        for user_id, days in split_baskets_by_user.items():  # 按日追加，不把整周揉成假序列
            history_map[user_id].extend(days)  # 并入后续划分
    return rows_written  # 返回样本数


def prepare_recbole_benchmark_files(  # 构建 train/valid/test 三个序列文件
    max_item_list_length: int,
    train_split_file: Path = TRAIN_SPLIT_FILE,
    valid_split_file: Path = VALID_SPLIT_FILE,
    test_split_file: Path = TEST_SPLIT_FILE,
    target_dir: Path = TARGET_DIR,
    train_history_file: Path | None = None,  # 自定义 fixture 默认复用 train_split
    max_shopping_days: int | None = None,  # 最近 N 个购物日；默认只按展平长度截断
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
    history_map: BasketHistory = defaultdict(list)  # 跨划分按日历史

    train_rows = convert_to_sequence_samples(  # train 内按购物日滚动
        sources[0],
        targets[0],
        history_map,
        max_item_list_length,
        True,
        False,
        max_shopping_days,
    )
    history_map = defaultdict(list, load_history_map(resolved_train_history))  # valid 使用完整 train 历史
    valid_rows = convert_to_sequence_samples(  # valid 周内不滚动，结束后纳入 test 历史
        sources[1],
        targets[1],
        history_map,
        max_item_list_length,
        False,
        True,
        max_shopping_days,
    )
    test_rows = convert_to_sequence_samples(  # test 只消费历史，不推进
        sources[2],
        targets[2],
        history_map,
        max_item_list_length,
        False,
        False,
        max_shopping_days,
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
