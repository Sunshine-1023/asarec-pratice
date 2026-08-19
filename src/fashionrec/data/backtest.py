"""Rolling time-window backtest splits. Official test is window 0 only."""  # 多窗口回测；正式报告只用窗口 0 的 test

from __future__ import annotations  # 延迟注解

import json  # 窗口清单
from dataclasses import dataclass  # 窗口说明
from datetime import datetime, timezone  # 生成时间
from pathlib import Path  # 路径
from typing import Any  # 清单附加字段

import pandas as pd  # 日期

from fashionrec.data.split import (  # 复用官方切分公式
    SplitBounds,  # 单窗口边界
    TimeSplitResult,  # 写出路径
    compute_split_bounds,  # 由锚点算边界
    split_bounds_from_bounds,  # manifest 日期
    split_by_time,  # 按边界切 hm.inter
)
from fashionrec.data.time import as_naive_utc_day  # 锚点去时区


BACKTEST_SCHEMA_VERSION = "hm.backtest.v1"  # 回测窗口语义
DEFAULT_N_WINDOWS = 3  # 与 experiment.yaml 默认一致
DEFAULT_SHIFT_WEEKS = 1  # 每个窗口锚点往回一周，使 window i 的 test = window i-1 的 valid


@dataclass(frozen=True)  # 不可变窗口
class BacktestWindow:  # 一个回测窗口
    window_id: int  # 0 为官方切分
    bounds: SplitBounds  # 日期边界
    official_test: bool  # 只有窗口 0 的 test 写入正式报告
    shift_weeks: int  # 相对官方锚点回移的周数


def required_preprocess_weeks(  # hm.inter 需要覆盖最早窗口的完整历史
    *,
    train_weeks: int,  # 单窗口训练周
    valid_weeks: int,  # 单窗口验证周
    test_weeks: int,  # 单窗口测试周
    n_windows: int,  # 回测窗口数
    shift_weeks: int = DEFAULT_SHIFT_WEEKS,  # 相邻窗口位移
) -> int:  # 预处理应保留的周数
    if n_windows < 1:  # 至少官方窗口
        raise ValueError("n_windows must be >= 1")  # 无法枚举
    if shift_weeks < 1:  # 位移必须为正
        raise ValueError("shift_weeks must be >= 1")  # 无法回移
    protocol = train_weeks + valid_weeks + test_weeks  # 单窗口协议周数
    return protocol + (n_windows - 1) * shift_weeks  # 最早窗口仍能装满 history+valid+test


def enumerate_backtest_windows(  # 从官方锚点往回生成窗口；不训练、不评官方 test 多次
    max_date: pd.Timestamp,  # 官方窗口最后一天
    *,
    train_weeks: int,  # 训练周
    valid_weeks: int,  # 验证周
    test_weeks: int,  # 测试周
    n_windows: int,  # 窗口数
    shift_weeks: int = DEFAULT_SHIFT_WEEKS,  # 位移周数
) -> list[BacktestWindow]:  # 窗口 0 在前
    if n_windows < 1:  # 非法
        raise ValueError("n_windows must be >= 1")  # 报错
    if shift_weeks < 1:  # 非法位移
        raise ValueError("shift_weeks must be >= 1")  # 报错
    anchor0 = pd.Timestamp(max_date).normalize()  # 官方锚点
    windows: list[BacktestWindow] = []  # 结果
    for window_id in range(n_windows):  # 0, 1, 2, ...
        shift = window_id * shift_weeks  # 相对官方回移
        anchor = anchor0 - pd.Timedelta(days=shift * 7)  # 该窗口最后一天
        bounds = compute_split_bounds(  # 与 split_by_time 同一公式
            anchor,  # 锚点
            train_weeks=train_weeks,  # 训练
            valid_weeks=valid_weeks,  # 验证
            test_weeks=test_weeks,  # 测试
        )  # 边界
        windows.append(  # 记录窗口
            BacktestWindow(  # 说明
                window_id=window_id,  # 编号
                bounds=bounds,  # 日期
                official_test=window_id == 0,  # 仅窗口 0 是正式 test
                shift_weeks=shift,  # 回移周数
            )
        )  # 追加结束
    return windows  # 返回


def window_dir(output_dir: Path, window_id: int) -> Path:  # 单窗口产物目录
    return Path(output_dir) / "windows" / f"w{window_id}"  # w0 / w1 / w2


def window_split_paths(output_dir: Path, window_id: int) -> dict[str, Path]:  # 单窗口文件布局
    root = window_dir(output_dir, window_id)  # 窗口根
    return {  # 逻辑名到路径
        "root": root,  # 根
        "train": root / "hm.train.inter",  # 该窗口训练
        "valid": root / "hm.valid.inter",  # 该窗口验证（只用于选参）
        "test": root / "hm.test.inter",  # 该窗口测试；仅 w0 是官方报告
        "manifest": root / "manifest.json",  # 边界清单
        "snapshots": root / "snapshots",  # 可选 next-basket 索引
        "labels": root / "labels",  # 可选标签
    }  # 布局结束


def window_manifest_payload(window: BacktestWindow, paths: dict[str, Path]) -> dict[str, Any]:  # 单窗口 manifest
    payload: dict[str, Any] = {  # 可审计字段
        "schema_version": BACKTEST_SCHEMA_VERSION,  # 语义
        "window_id": window.window_id,  # 编号
        "official_test": window.official_test,  # 是否正式 test
        "shift_weeks": window.shift_weeks,  # 回移
        "valid_role": "selection",  # valid 只选参
        "test_role": "official_report" if window.official_test else "local_holdout",  # test 角色
    }  # 元信息结束
    payload.update(split_bounds_from_bounds(window.bounds))  # 日期边界
    payload["paths"] = {name: str(path) for name, path in paths.items()}  # 产物路径
    return payload  # 返回


def write_json(payload: dict[str, Any], path: Path) -> Path:  # 写出 UTF-8 JSON
    path = Path(path)  # 规范化
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 写出
    return path  # 返回路径


def build_backtest_windows(  # 从同一份 hm.inter 写出多个窗口切分
    *,
    inter_path: Path,  # 已按 required_preprocess_weeks 拉长的交互
    output_dir: Path,  # backtest 根目录
    train_weeks: int,  # 训练周
    valid_weeks: int,  # 验证周
    test_weeks: int,  # 测试周
    n_windows: int,  # 窗口数
    shift_weeks: int = DEFAULT_SHIFT_WEEKS,  # 位移
    max_date: pd.Timestamp | None = None,  # 官方锚点；默认读文件最大日
    extra: dict[str, Any] | None = None,  # 写入 windows.json 的附加字段
) -> list[tuple[BacktestWindow, TimeSplitResult]]:  # 窗口与切分结果
    inter_path = Path(inter_path)  # 规范化
    output_dir = Path(output_dir)  # 规范化
    if max_date is None:  # 未显式给锚点
        timestamps = pd.read_csv(inter_path, sep="\t", usecols=["timestamp:float"])  # 只读时间
        max_date = pd.to_datetime(timestamps["timestamp:float"], unit="s", utc=True).dt.tz_localize(None).dt.floor("D").max()
    max_date = as_naive_utc_day(max_date)  # 与 split 边界同一日历日
    windows = enumerate_backtest_windows(  # 生成窗口
        max_date,  # 官方锚点
        train_weeks=train_weeks,  # 训练
        valid_weeks=valid_weeks,  # 验证
        test_weeks=test_weeks,  # 测试
        n_windows=n_windows,  # 个数
        shift_weeks=shift_weeks,  # 位移
    )  # 枚举结束
    protocol_weeks = train_weeks + valid_weeks + test_weeks  # 单窗口周数
    written: list[tuple[BacktestWindow, TimeSplitResult]] = []  # 结果
    index_windows: list[dict[str, Any]] = []  # 总清单里的窗口摘要
    for window in windows:  # 每个窗口单独切分
        paths = window_split_paths(output_dir, window.window_id)  # 路径
        result = split_by_time(  # 用显式边界，不用文件 max_date
            inter_path=inter_path,  # 同一份更长的 hm.inter
            total_weeks=protocol_weeks,  # 协议周数，不是预处理周数
            train_weeks=train_weeks,  # 训练
            valid_weeks=valid_weeks,  # 验证
            test_weeks=test_weeks,  # 测试
            train_inter_path=paths["train"],  # 窗口训练
            valid_inter_path=paths["valid"],  # 窗口验证
            test_inter_path=paths["test"],  # 窗口测试
            bounds=window.bounds,  # 回测边界
        )  # 切分结束
        payload = window_manifest_payload(window, paths)  # 单窗口清单
        write_json(payload, paths["manifest"])  # 写出
        index_windows.append(payload)  # 汇总
        written.append((window, result))  # 记录
        print(  # 提示角色
            f"backtest window {window.window_id}: official_test={window.official_test} "
            f"test_role={payload['test_role']} "
            f"[{window.bounds.window_start.date()}, {window.bounds.test_end.date()}]"
        )  # 打印结束
    index: dict[str, Any] = {  # 总清单
        "schema_version": BACKTEST_SCHEMA_VERSION,  # 语义
        "n_windows": len(windows),  # 窗口数
        "official_test_window_id": 0,  # 正式 test 只评这一次
        "shift_weeks": shift_weeks,  # 位移
        "train_weeks": train_weeks,  # 训练周
        "valid_weeks": valid_weeks,  # 验证周
        "test_weeks": test_weeks,  # 测试周
        "preprocess_weeks": required_preprocess_weeks(  # 实际应保留的历史
            train_weeks=train_weeks,  # 训练
            valid_weeks=valid_weeks,  # 验证
            test_weeks=test_weeks,  # 测试
            n_windows=n_windows,  # 窗口
            shift_weeks=shift_weeks,  # 位移
        ),  # 预处理周数
        "generated_at": datetime.now(timezone.utc).isoformat(),  # UTC
        "windows": index_windows,  # 各窗口
    }  # 总清单结束
    if extra:  # 调用方附加字段
        index.update(extra)  # 合并
    write_json(index, output_dir / "windows.json")  # 写出总清单
    print(f"Wrote backtest index: {output_dir / 'windows.json'} ({len(windows)} windows)")  # 提示
    return written  # 返回
