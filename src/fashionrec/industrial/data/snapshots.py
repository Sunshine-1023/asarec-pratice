"""Weekly prediction snapshots for next-basket labels."""  # 按周冻结 as_of，标签窗口严格在其后

from __future__ import annotations  # 延迟注解

import shutil  # 覆盖旧分区
from dataclasses import dataclass  # 快照说明
from pathlib import Path  # 路径

import pandas as pd  # 日期与聚合

from fashionrec.industrial.data.split import TimeSplitResult  # 与现有时间切分对齐
from fashionrec.shared.domain.ids import canonical_user_id  # 用户 ID


SNAPSHOT_SCHEMA_VERSION = "hm.snapshot.v1"  # 快照索引语义
PARTITION_COL = "as_of_date"  # 按预测日分区
SNAPSHOT_COLUMNS = (  # 样本索引列
    "user_id",  # 用户
    "as_of_date",  # 预测日，历史含当天，标签从次日开始
    "split",  # train / valid / test
    "label_start",  # 标签窗口首日（含）
    "label_end",  # 标签窗口末日（含）
    "n_label_items",  # 窗口内去重 SKU 数
    "n_history_items",  # as_of 当天及之前去重 SKU 数
    "is_cold_start",  # 预测日之前（含当天）没有历史
)  # 列结束


@dataclass(frozen=True, slots=True)  # 一次预测快照
class SnapshotSpec:  # as_of 与所属划分
    as_of_date: pd.Timestamp  # 预测日
    split: str  # train / valid / test


def _as_day(value: pd.Timestamp | str) -> pd.Timestamp:  # 规范到自然日
    return pd.Timestamp(value).normalize()  # 去掉时分秒


def label_window(as_of_date: pd.Timestamp | str, horizon_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:  # 开闭窗口
    if horizon_days < 1:  # 非法窗口
        raise ValueError("horizon_days must be >= 1")  # 报错
    as_of = _as_day(as_of_date)  # 预测日
    start = as_of + pd.Timedelta(days=1)  # 严格在 as_of 之后
    end = as_of + pd.Timedelta(days=horizon_days)  # 含第 horizon 天
    return start, end  # 返回 [start, end]


def weekly_as_of_dates(  # 标签窗口完全落在 [min_as_of 的次日, max_label_end] 内的周快照
    *,
    min_as_of: pd.Timestamp | str,  # 最早允许的预测日
    max_label_end: pd.Timestamp | str,  # 标签末日不得晚于此
    horizon_days: int,  # 未来窗口
    step_days: int = 7,  # weekly
) -> list[pd.Timestamp]:  # 从旧到新
    if step_days < 1:  # 非法步长
        raise ValueError("step_days must be >= 1")  # 报错
    first = _as_day(min_as_of)  # 下界
    last_end = _as_day(max_label_end)  # 上界
    as_of = last_end - pd.Timedelta(days=horizon_days)  # 最后一次还能装满 horizon 的预测日
    dates: list[pd.Timestamp] = []  # 收集
    while as_of >= first:  # 仍在训练/历史窗口内
        dates.append(as_of)  # 记录
        as_of = as_of - pd.Timedelta(days=step_days)  # 上一周
    return sorted(dates)  # 旧到新


def snapshot_specs_from_split(  # 训练周内 weekly，valid/test 各一次，避免标签泄漏
    split: TimeSplitResult,  # 时间切分
    *,
    horizon_days: int,  # 与 valid/test 周长对齐时应为 7
) -> list[SnapshotSpec]:  # 快照列表
    train_as_ofs = weekly_as_of_dates(  # 标签必须落在 train 内
        min_as_of=split.window_start,  # 总窗口起
        max_label_end=split.train_end,  # 不得超过训练末日
        horizon_days=horizon_days,  # 未来窗口
    )
    specs = [SnapshotSpec(as_of_date=day, split="train") for day in train_as_ofs]  # 训练快照
    specs.append(SnapshotSpec(as_of_date=_as_day(split.train_end), split="valid"))  # valid：历史到 train_end
    specs.append(SnapshotSpec(as_of_date=_as_day(split.valid_end), split="test"))  # test：历史到 valid_end
    return specs  # 返回


def build_snapshot_index(  # 每个 (user, as_of) 一行，仅保留标签窗口内有购买的用户
    events: pd.DataFrame,  # 需含 user_id, date, item_id
    specs: list[SnapshotSpec],  # 要物化的快照
    *,
    horizon_days: int,  # 标签窗口
) -> pd.DataFrame:  # 样本索引
    required = {"user_id", "item_id", "date"}  # 最少列
    missing = required.difference(events.columns)  # 缺列
    if missing:  # schema 不对
        raise ValueError(f"events missing columns: {sorted(missing)}")  # 报错
    if not specs:  # 没有快照
        raise ValueError("specs must not be empty")  # 报错

    frame = events.loc[:, ["user_id", "item_id", "date"]].copy()  # 只用索引所需列
    frame["user_id"] = frame["user_id"].map(canonical_user_id)  # 规范化
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()  # 自然日
    rows: list[dict[str, object]] = []  # 逐快照收集
    for spec in specs:  # 每个预测日
        as_of = _as_day(spec.as_of_date)  # 预测日
        start, end = label_window(as_of, horizon_days)  # 标签窗口
        label_events = frame[(frame["date"] >= start) & (frame["date"] <= end)]  # 窗口内购买
        if label_events.empty:  # 该周无人购买
            continue  # 不写空快照
        history = frame[frame["date"] <= as_of]  # 含当天的历史
        hist_n = history.groupby("user_id")["item_id"].nunique()  # 历史 SKU 数
        label_n = label_events.groupby("user_id")["item_id"].nunique()  # 标签 SKU 数
        for user_id, n_label in label_n.items():  # 只给窗口内买过的用户建样本
            n_hist = int(hist_n.get(user_id, 0))  # 没有历史则为 0
            rows.append(  # 一行样本索引
                {
                    "user_id": user_id,  # 用户
                    "as_of_date": as_of,  # 预测日
                    "split": spec.split,  # 划分
                    "label_start": start,  # 窗口起
                    "label_end": end,  # 窗口止
                    "n_label_items": int(n_label),  # 去重标签商品
                    "n_history_items": n_hist,  # 历史商品
                    "is_cold_start": n_hist == 0,  # 冷启动
                }
            )  # 行结束
    if not rows:  # 全部快照都没有正例
        raise ValueError("no snapshot users found in label windows")  # 报错
    index = pd.DataFrame(rows)  # 组装
    index = index.sort_values(["as_of_date", "user_id", "split"], kind="mergesort").reset_index(drop=True)  # 稳定顺序
    index["n_label_items"] = index["n_label_items"].astype("int64")  # 整数
    index["n_history_items"] = index["n_history_items"].astype("int64")  # 整数
    return index.loc[:, list(SNAPSHOT_COLUMNS)]  # 固定列


def write_snapshots_parquet(index: pd.DataFrame, output_dir: Path) -> Path:  # 按 as_of_date 分区写出
    output_dir = Path(output_dir)  # 规范化
    if index.empty:  # 空表
        raise ValueError("cannot write empty snapshot index")  # 拒绝
    if output_dir.exists():  # 清旧分区
        shutil.rmtree(output_dir)  # 删除
    output_dir.mkdir(parents=True, exist_ok=True)  # 重建
    frame = index.copy()  # 不改调用方
    frame[PARTITION_COL] = pd.to_datetime(frame["as_of_date"]).dt.strftime("%Y-%m-%d")  # 分区键
    frame.to_parquet(output_dir, partition_cols=[PARTITION_COL], index=False, engine="pyarrow")  # 写出
    return output_dir  # 返回
