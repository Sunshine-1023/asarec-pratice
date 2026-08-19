"""Tests for weekly snapshot calendars and sample indexes."""  # 快照索引测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 日期
import pytest  # 异常

from fashionrec.data.command import main as data_main  # 帮助
from fashionrec.data.command import processed_layout  # 布局
from fashionrec.data.snapshots import (  # 快照
    SnapshotSpec,  # 说明
    build_snapshot_index,  # 索引
    label_window,  # 窗口
    snapshot_specs_from_split,  # 从切分生成
    weekly_as_of_dates,  # 周历
)
from fashionrec.data.split import TimeSplitResult  # 切分结果


def _split() -> TimeSplitResult:  # 6 周窗口：4 训练 + 1 验证 + 1 测试
    dummy = Path("unused.inter")  # 路径不参与日历
    return TimeSplitResult(  # 与 split_by_time 相同边界语义
        train_path=dummy,
        valid_path=dummy,
        test_path=dummy,
        window_start=pd.Timestamp("2020-08-12"),
        max_date=pd.Timestamp("2020-09-22"),
        train_end=pd.Timestamp("2020-09-08"),
        valid_start=pd.Timestamp("2020-09-09"),
        valid_end=pd.Timestamp("2020-09-15"),
        test_start=pd.Timestamp("2020-09-16"),
        test_end=pd.Timestamp("2020-09-22"),
    )


def test_label_window_starts_the_day_after_as_of() -> None:  # 标签严格在 as_of 之后
    start, end = label_window("2020-09-15", 7)
    assert start == pd.Timestamp("2020-09-16")
    assert end == pd.Timestamp("2020-09-22")


def test_weekly_as_of_dates_fit_horizon_inside_train() -> None:  # 训练标签不得越过 train_end
    dates = weekly_as_of_dates(
        min_as_of="2020-08-12",
        max_label_end="2020-09-08",
        horizon_days=7,
    )
    assert dates[0] == pd.Timestamp("2020-08-18")  # 从最后一次能装满 horizon 的预测日往回每周一档
    assert dates[-1] == pd.Timestamp("2020-09-01")  # 09-01 + 7 天 = 09-08
    assert all((day + pd.Timedelta(days=7)) <= pd.Timestamp("2020-09-08") for day in dates)


def test_split_specs_keep_valid_and_test_labels_out_of_train() -> None:  # valid/test 各一次
    specs = snapshot_specs_from_split(_split(), horizon_days=7)
    train_as_ofs = [spec.as_of_date for spec in specs if spec.split == "train"]
    valid = next(spec for spec in specs if spec.split == "valid")
    test = next(spec for spec in specs if spec.split == "test")
    assert valid.as_of_date == pd.Timestamp("2020-09-08")  # 历史到 train_end
    assert test.as_of_date == pd.Timestamp("2020-09-15")  # 历史到 valid_end
    assert valid.as_of_date not in train_as_ofs  # 验证周购买不能当训练标签
    start, end = label_window(valid.as_of_date, 7)
    assert start == pd.Timestamp("2020-09-09")
    assert end == pd.Timestamp("2020-09-15")


def test_snapshot_index_marks_cold_start_and_only_users_with_labels() -> None:  # 只给窗口内买过的用户建样本
    events = pd.DataFrame(
        {
            "user_id": ["warm", "warm", "cold", "idle"],
            "item_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "date": [
                "2020-09-08",
                "2020-09-16",
                "2020-09-16",
                "2020-09-08",
            ],
        }
    )
    specs = [SnapshotSpec(as_of_date=pd.Timestamp("2020-09-15"), split="valid")]
    index = build_snapshot_index(events, specs, horizon_days=7)
    assert sorted(index["user_id"].tolist()) == ["cold", "warm"]  # idle 窗口内没买
    warm = index[index["user_id"] == "warm"].iloc[0]
    cold = index[index["user_id"] == "cold"].iloc[0]
    assert int(warm["n_history_items"]) == 1
    assert bool(warm["is_cold_start"]) is False
    assert int(cold["n_history_items"]) == 0
    assert bool(cold["is_cold_start"]) is True
    assert pd.Timestamp(warm["as_of_date"]).normalize() == pd.Timestamp("2020-09-15")


def test_processed_layout_includes_snapshot_and_label_dirs(tmp_path: Path) -> None:  # 布局预留目录
    layout = processed_layout(tmp_path)
    assert layout["snapshots"] == tmp_path / "snapshots"
    assert layout["labels"] == tmp_path / "labels"


def test_data_command_help_documents_opt_in_build_labels(capsys) -> None:  # 默认不落盘
    with pytest.raises(SystemExit) as exited:
        data_main(["--help"])
    assert exited.value.code == 0
    assert "--build-labels" in capsys.readouterr().out
