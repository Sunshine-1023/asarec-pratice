"""Tests for causal time splits and deterministic interaction order."""  # 时间切分与确定性排序测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 读写切分结果
import pytest  # 断言

from src.data.split import (  # 切分
    INTERACTION_SORT_COLUMNS,  # 排序键
    history_paths_for_eval,  # 历史路径
    sort_interactions,  # 确定性排序
    split_by_time,  # 切分
    validate_time_split,  # 因果检查
)  # 导入结束


def _unix(date: str, hour: int = 12) -> int:  # 把日期转成 Unix 秒（UTC，避免本地时区把日期挪一天）
    ts = pd.Timestamp(f"{date} {hour:02d}:00:00", tz="UTC")  # UTC 中午
    return int(ts.timestamp())  # 秒


def _write_inter(path: Path, rows: list[tuple[str, str, str, int]]) -> None:  # 写 inter：user, item, date, hour
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]  # 表头
    for user, item, date, hour in rows:  # 逐行
        lines.append(f"{user}\t{item}\t{_unix(date, hour)}")  # 时间戳
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写出


def _six_week_rows() -> list[tuple[str, str, str, int]]:  # 覆盖 6 周窗口的最小样本
    return [  # 训练 / 验证 / 测试都有记录
        ("u1", "b", "2020-08-13", 10),  # 窗口第一天，同一时间戳的 b
        ("u1", "a", "2020-08-13", 10),  # 同一用户同一时间戳的 a，输入时 b 在前
        ("u2", "x", "2020-09-08", 12),  # 训练最后一天
        ("u1", "v", "2020-09-09", 12),  # 验证第一天
        ("u2", "w", "2020-09-15", 12),  # 验证最后一天
        ("u1", "t", "2020-09-16", 12),  # 测试第一天
        ("u2", "z", "2020-09-22", 12),  # 测试最后一天 / 窗口最大日
    ]  # 行结束


def test_split_timestamps_are_strictly_increasing(tmp_path: Path) -> None:  # max(train)<min(valid)<min(test)
    source = tmp_path / "hm.inter"  # 输入
    _write_inter(source, _six_week_rows())  # 写入 6 周样本
    result = split_by_time(  # 切分
        inter_path=source,  # 输入
        train_inter_path=tmp_path / "train.inter",  # 训练
        valid_inter_path=tmp_path / "valid.inter",  # 验证
        test_inter_path=tmp_path / "test.inter",  # 测试
    )  # 切分结束
    train = pd.read_csv(result.train_path, sep="\t")  # 读训练
    valid = pd.read_csv(result.valid_path, sep="\t")  # 读验证
    test = pd.read_csv(result.test_path, sep="\t")  # 读测试
    validate_time_split(train, valid, test)  # 严格时间顺序
    assert float(train["timestamp:float"].max()) < float(valid["timestamp:float"].min())  # 训练 < 验证
    assert float(valid["timestamp:float"].max()) < float(test["timestamp:float"].min())  # 验证 < 测试
    assert str(result.valid_start.date()) == "2020-09-09"  # 验证起
    assert str(result.test_start.date()) == "2020-09-16"  # 测试起


def test_same_timestamp_order_is_deterministic(tmp_path: Path) -> None:  # 同一时间戳不依赖原始行序
    rows_ab = _six_week_rows()  # b 在 a 前
    rows_ba = [rows_ab[1], rows_ab[0], *rows_ab[2:]]  # 把同一时间戳的 a/b 对调
    outs = []  # 两次切分的训练文件
    for name, rows in ("first", rows_ab), ("second", rows_ba):  # 两种输入顺序
        source = tmp_path / f"{name}.inter"  # 输入
        _write_inter(source, rows)  # 写出
        result = split_by_time(  # 切分
            inter_path=source,  # 输入
            train_inter_path=tmp_path / f"{name}_train.inter",  # 训练
            valid_inter_path=tmp_path / f"{name}_valid.inter",  # 验证
            test_inter_path=tmp_path / f"{name}_test.inter",  # 测试
        )  # 切分结束
        outs.append(pd.read_csv(result.train_path, sep="\t"))  # 记录训练表
    pd.testing.assert_frame_equal(outs[0], outs[1])  # 两次输出必须逐行相同
    first_day = outs[0][outs[0]["user_id:token"] == "u1"]  # 用户 u1
    assert first_day.iloc[0]["item_id:token"] == "a"  # 同时间戳按 item_id 升序，a 在 b 前


def test_sort_interactions_uses_item_id_tiebreak() -> None:  # 排序键包含商品 ID
    df = pd.DataFrame(  # 两行同用户同时戳
        {  # 列
            "user_id:token": ["u", "u"],  # 用户
            "item_id:token": ["2", "1"],  # 商品
            "timestamp:float": [10.0, 10.0],  # 时间
        }  # 列结束
    )  # 表结束
    ordered = sort_interactions(df)  # 排序
    assert list(ordered["item_id:token"]) == ["0000000001", "0000000002"]  # 规范化 item_id 打破并列
    assert INTERACTION_SORT_COLUMNS[-1] == "item_id:token"  # 第三键是商品 ID


def test_history_paths_for_eval() -> None:  # valid 只用 train，test 用 train+valid
    train = Path("data/processed/hm/hm.train.inter")  # 训练
    valid = Path("data/processed/hm/hm.valid.inter")  # 验证
    assert history_paths_for_eval("valid", train, valid) == [train]  # 验证评估
    assert history_paths_for_eval("test", train, valid) == [train, valid]  # 测试评估
    with pytest.raises(ValueError):  # 非法划分
        history_paths_for_eval("train", train, valid)  # 不允许
