"""Rolling backtest windows: three splits from one inter, official test is window 0."""  # 多窗口回测测试

from __future__ import annotations  # 延迟注解

import json  # 读清单
from pathlib import Path  # 路径

import pandas as pd  # 日期与切分表
import pytest  # 断言

from fashionrec.data.backtest import (  # 回测
    BACKTEST_SCHEMA_VERSION,  # 语义
    build_backtest_windows,  # 落盘
    enumerate_backtest_windows,  # 枚举
    required_preprocess_weeks,  # 拉长周数
)
from fashionrec.data.command import main as data_main  # 帮助
from fashionrec.data.command import processed_layout  # 布局
from fashionrec.data.split import (  # 切分
    compute_split_bounds,  # 边界公式
    split_by_time,  # 官方切分
    validate_time_split,  # 无重叠
)


def _unix(date: str, hour: int = 12) -> int:  # UTC 中午，避免本地时区挪一天
    ts = pd.Timestamp(f"{date} {hour:02d}:00:00", tz="UTC")  # UTC
    return int(ts.timestamp())  # 秒


def _write_inter(path: Path, rows: list[tuple[str, str, str]]) -> None:  # user, item, date
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]  # 表头
    for user, item, date in rows:  # 逐行
        lines.append(f"{user}\t{item}\t{_unix(date)}")  # 时间戳
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写出


def _eight_week_rows() -> list[tuple[str, str, str]]:  # 3 个 6 周窗口需要 8 周历史
    return [  # 每周至少一行，覆盖 w2 的 train 到 w0 的 test
        ("u0", "early2", "2020-07-29"),  # w2 训练起点
        ("u0", "early1", "2020-08-05"),  # w1 窗口起点
        ("u1", "b", "2020-08-12"),  # w0 窗口起点
        ("u1", "mid", "2020-08-19"),  # w0 训练
        ("u2", "w2v", "2020-08-26"),  # w2 验证
        ("u2", "w2t", "2020-09-02"),  # w2 测试 / w1 验证
        ("u1", "v", "2020-09-09"),  # w1 测试 / w0 验证
        ("u1", "t", "2020-09-16"),  # w0 测试起点
        ("u2", "z", "2020-09-22"),  # 官方锚点
    ]  # 行结束


def _canonical_index(payload: dict) -> dict:  # 去掉生成时间再比
    copied = json.loads(json.dumps(payload))  # 深拷贝
    copied.pop("generated_at", None)  # 时间戳不参与比较
    return copied  # 返回


def test_required_preprocess_weeks_covers_earliest_window() -> None:  # 3 窗要多留 2 周
    assert required_preprocess_weeks(train_weeks=4, valid_weeks=1, test_weeks=1, n_windows=3) == 8  # 6+2
    assert required_preprocess_weeks(train_weeks=26, valid_weeks=1, test_weeks=1, n_windows=3) == 30  # 28+2
    with pytest.raises(ValueError, match="n_windows"):  # 至少一窗
        required_preprocess_weeks(train_weeks=4, valid_weeks=1, test_weeks=1, n_windows=0)  # 非法


def test_enumerate_three_windows_marks_only_window_zero_as_official() -> None:  # 正式 test 只一次
    windows = enumerate_backtest_windows(  # 3 窗
        pd.Timestamp("2020-09-22"),  # 官方锚点
        train_weeks=4,  # 训练
        valid_weeks=1,  # 验证
        test_weeks=1,  # 测试
        n_windows=3,  # 三个
    )  # 枚举结束
    assert [window.window_id for window in windows] == [0, 1, 2]  # 顺序
    assert [window.official_test for window in windows] == [True, False, False]  # 只有 w0
    assert [str(window.bounds.test_end.date()) for window in windows] == [  # 锚点每周回移
        "2020-09-22",  # 官方
        "2020-09-15",  # 上一周
        "2020-09-08",  # 再上一周
    ]
    assert str(windows[0].bounds.valid_start.date()) == "2020-09-09"  # 官方 valid
    assert str(windows[1].bounds.test_start.date()) == "2020-09-09"  # w1 test = w0 valid
    assert windows[0].bounds == compute_split_bounds(  # 窗口 0 等于官方公式
        pd.Timestamp("2020-09-22"),  # 锚点
        train_weeks=4,  # 训练
        valid_weeks=1,  # 验证
        test_weeks=1,  # 测试
    )  # 边界


def test_same_command_writes_three_window_manifests(tmp_path: Path) -> None:  # 同一调用生成 3 窗
    source = tmp_path / "hm.inter"  # 输入
    _write_inter(source, _eight_week_rows())  # 8 周样本
    output = tmp_path / "backtest"  # 产物
    written = build_backtest_windows(  # 一次写出
        inter_path=source,  # 同一份 inter
        output_dir=output,  # 目录
        train_weeks=4,  # 训练
        valid_weeks=1,  # 验证
        test_weeks=1,  # 测试
        n_windows=3,  # 三窗
        max_date=pd.Timestamp("2020-09-22"),  # 官方锚点
    )  # 构建结束
    assert len(written) == 3  # 三个窗口
    index = json.loads((output / "windows.json").read_text(encoding="utf-8"))  # 总清单
    assert index["schema_version"] == BACKTEST_SCHEMA_VERSION  # 语义
    assert index["n_windows"] == 3  # 窗数
    assert index["official_test_window_id"] == 0  # 正式 test
    assert index["preprocess_weeks"] == 8  # 需要 8 周历史
    again = json.loads((output / "windows.json").read_text(encoding="utf-8"))  # 再读一次
    assert _canonical_index(index)["windows"] == _canonical_index(again)["windows"]  # 边界稳定
    for window_id in range(3):  # 每个窗口都有清单和切分
        manifest = json.loads((output / "windows" / f"w{window_id}" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["window_id"] == window_id  # 编号
        assert manifest["official_test"] is (window_id == 0)  # 正式标记
        assert manifest["valid_role"] == "selection"  # valid 只选参
        assert manifest["test_role"] == ("official_report" if window_id == 0 else "local_holdout")  # test 角色
        assert "window_start" in manifest and "test_end" in manifest  # 边界齐全
        window, result = written[window_id]  # 切分结果
        train = pd.read_csv(result.train_path, sep="\t")  # 训练
        valid = pd.read_csv(result.valid_path, sep="\t")  # 验证
        test = pd.read_csv(result.test_path, sep="\t")  # 测试
        validate_time_split(train, valid, test)  # 无重叠
        assert result.bounds == window.bounds  # 写出边界与枚举一致


def test_window_zero_matches_official_split(tmp_path: Path) -> None:  # w0 与默认 split_by_time 相同
    source = tmp_path / "hm.inter"  # 输入
    _write_inter(source, _eight_week_rows())  # 含更早周，官方切分应忽略它们
    official = split_by_time(  # 按文件 max_date
        inter_path=source,  # 输入
        total_weeks=6,  # 协议 6 周
        train_weeks=4,  # 训练
        valid_weeks=1,  # 验证
        test_weeks=1,  # 测试
        train_inter_path=tmp_path / "official_train.inter",  # 官方训练
        valid_inter_path=tmp_path / "official_valid.inter",  # 官方验证
        test_inter_path=tmp_path / "official_test.inter",  # 官方测试
    )  # 官方切分
    written = build_backtest_windows(  # 回测
        inter_path=source,  # 同一文件
        output_dir=tmp_path / "backtest",  # 目录
        train_weeks=4,  # 训练
        valid_weeks=1,  # 验证
        test_weeks=1,  # 测试
        n_windows=3,  # 三窗
        max_date=official.max_date,  # 同一锚点
    )  # 回测结束
    window0 = written[0][1]  # 窗口 0
    assert window0.bounds == official.bounds  # 日期一致
    pd.testing.assert_frame_equal(  # 训练行一致
        pd.read_csv(official.train_path, sep="\t"),  # 官方
        pd.read_csv(window0.train_path, sep="\t"),  # 回测 w0
    )
    pd.testing.assert_frame_equal(  # 测试行一致
        pd.read_csv(official.test_path, sep="\t"),
        pd.read_csv(window0.test_path, sep="\t"),
    )


def test_shifted_window_cannot_see_official_test_week(tmp_path: Path) -> None:  # 早期窗口不得读官方 test
    source = tmp_path / "hm.inter"  # 输入
    _write_inter(source, _eight_week_rows())  # 8 周
    written = build_backtest_windows(  # 三窗
        inter_path=source,  # 输入
        output_dir=tmp_path / "backtest",  # 目录
        train_weeks=4,  # 训练
        valid_weeks=1,  # 验证
        test_weeks=1,  # 测试
        n_windows=3,  # 三窗
        max_date=pd.Timestamp("2020-09-22"),  # 锚点
    )  # 构建
    official_test_start = float(_unix("2020-09-16"))  # 官方 test 起点
    for window, result in written[1:]:  # w1 / w2
        for path in (result.train_path, result.valid_path, result.test_path):  # 该窗口全部历史+标签
            frame = pd.read_csv(path, sep="\t")  # 读出
            leaked = frame[frame["timestamp:float"] >= official_test_start]  # 官方 test 周
            assert leaked.empty, f"window {window.window_id} {path.name} leaked official test"  # 必须为空


def test_processed_layout_includes_backtest_dir(tmp_path: Path) -> None:  # 布局预留目录
    assert processed_layout(tmp_path)["backtest"] == tmp_path / "backtest"


def test_data_command_help_documents_opt_in_build_backtest(capsys) -> None:  # 默认不落盘
    with pytest.raises(SystemExit) as exited:
        data_main(["--help"])
    assert exited.value.code == 0
    assert "--build-backtest" in capsys.readouterr().out
