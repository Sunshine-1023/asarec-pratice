"""Tests for user-day-item event aggregation."""  # 同日同 SKU 事件测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 临时路径

import pandas as pd  # 写 CSV / 读 parquet
import pytest  # 异常

from fashionrec.data.build_events import (  # 事件构建
    EVENT_COLUMNS,  # 输出列
    aggregate_user_day_item_events,  # 纯函数
    build_events,  # CSV -> parquet
)
from fashionrec.data.command import main as data_main  # 检查 --build-events 开关
from fashionrec.data.command import processed_layout  # 产物布局


def _transactions(rows: list[dict[str, object]]) -> pd.DataFrame:  # 小 fixture
    return pd.DataFrame(rows)  # 保持调用方给出的字符串 ID


def test_duplicate_rows_become_quantity_two() -> None:  # 重复两行合成一件事件
    events = aggregate_user_day_item_events(
        _transactions(
            [
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0107360650",
                    "price": 0.02,
                    "sales_channel_id": 2,
                },
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0107360650",
                    "price": 0.04,
                    "sales_channel_id": 2,
                },
            ]
        )
    )
    assert len(events) == 1  # 一条事件
    row = events.iloc[0]  # 唯一行
    assert row["user_id"] == "user-a"  # 用户
    assert row["item_id"] == "0107360650"  # 前导零保留
    assert pd.Timestamp(row["date"]).normalize() == pd.Timestamp("2020-09-22")  # 自然日
    assert int(row["quantity"]) == 2  # 两行 -> 2
    assert row["mean_price"] == pytest.approx(0.03)  # (0.02+0.04)/2
    assert row["min_price"] == pytest.approx(0.02)  # 最小
    assert row["max_price"] == pytest.approx(0.04)  # 最大
    assert int(row["sales_channel_mode"]) == 2  # 单一渠道
    assert int(row["channel_count"]) == 1  # 一个渠道
    assert list(events.columns) == list(EVENT_COLUMNS)  # 列契约


def test_different_channels_count_and_mode_tie_breaks_to_smaller_id() -> None:  # 多渠道与打平
    events = aggregate_user_day_item_events(
        _transactions(
            [
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0100000001",
                    "price": 0.1,
                    "sales_channel_id": 2,
                },
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0100000001",
                    "price": 0.1,
                    "sales_channel_id": 1,
                },
            ]
        )
    )
    row = events.iloc[0]  # 唯一事件
    assert int(row["quantity"]) == 2  # 两行
    assert int(row["channel_count"]) == 2  # 两个渠道
    assert int(row["sales_channel_mode"]) == 1  # 次数相同取更小 id


def test_channel_mode_uses_majority_not_smaller_id() -> None:  # 多数渠道优先于更小 id
    events = aggregate_user_day_item_events(
        _transactions(
            [
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "1",
                    "price": 0.1,
                    "sales_channel_id": 2,
                },
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "1",
                    "price": 0.1,
                    "sales_channel_id": 2,
                },
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "1",
                    "price": 0.1,
                    "sales_channel_id": 1,
                },
            ]
        )
    )
    assert int(events.iloc[0]["sales_channel_mode"]) == 2  # 渠道 2 出现两次
    assert events.iloc[0]["item_id"] == "0000000001"  # 数字 ID 补齐十位


def test_same_day_different_skus_stay_separate_events() -> None:  # 同日不同 SKU 不合并
    events = aggregate_user_day_item_events(
        _transactions(
            [
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0100000001",
                    "price": 0.1,
                    "sales_channel_id": 1,
                },
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0100000002",
                    "price": 0.2,
                    "sales_channel_id": 1,
                },
            ]
        )
    )
    assert len(events) == 2  # 两件事件
    assert sorted(events["item_id"].tolist()) == ["0100000001", "0100000002"]  # 两个 SKU
    assert list(events["quantity"].astype(int)) == [1, 1]  # 各一行


def test_null_prices_are_excluded_from_stats_and_zero_is_kept() -> None:  # 空价格忽略，0 保留
    events = aggregate_user_day_item_events(
        _transactions(
            [
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0100000001",
                    "price": "",
                    "sales_channel_id": 1,
                },
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0100000001",
                    "price": 0,
                    "sales_channel_id": 1,
                },
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0100000002",
                    "price": "",
                    "sales_channel_id": 1,
                },
            ]
        )
    )
    first = events.loc[events["item_id"] == "0100000001"].iloc[0]  # 有 0 价格
    second = events.loc[events["item_id"] == "0100000002"].iloc[0]  # 全空
    assert int(first["quantity"]) == 2  # 空价格行仍计入件数
    assert first["min_price"] == pytest.approx(0.0)  # 0 参与统计
    assert first["max_price"] == pytest.approx(0.0)  # 只有 0
    assert pd.isna(second["mean_price"])  # 全空则为 null
    assert pd.isna(second["min_price"])  # 全空
    assert pd.isna(second["max_price"])  # 全空


def test_missing_user_or_date_raises() -> None:  # 缺主键报错，不丢行
    with pytest.raises(ValueError, match="customer_id"):  # 缺用户
        aggregate_user_day_item_events(
            _transactions(
                [
                    {
                        "t_dat": "2020-09-22",
                        "customer_id": "",
                        "article_id": "1",
                        "price": 0.1,
                        "sales_channel_id": 1,
                    }
                ]
            )
        )
    with pytest.raises(ValueError, match="t_dat"):  # 缺日期
        aggregate_user_day_item_events(
            _transactions(
                [
                    {
                        "t_dat": "",
                        "customer_id": "user-a",
                        "article_id": "1",
                        "price": 0.1,
                        "sales_channel_id": 1,
                    }
                ]
            )
        )


def test_events_parquet_is_partitioned_by_month(tmp_path: Path) -> None:  # 按月分区且可回读
    source = tmp_path / "transactions.csv"  # 输入
    output = tmp_path / "events"  # 输出
    pd.DataFrame(
        [
            {
                "t_dat": "2020-08-15",
                "customer_id": "user-a",
                "article_id": "0107360650",
                "price": 0.02,
                "sales_channel_id": 2,
            },
            {
                "t_dat": "2020-09-22",
                "customer_id": "user-a",
                "article_id": "0107360650",
                "price": 0.03,
                "sales_channel_id": 2,
            },
        ]
    ).to_csv(source, index=False)  # 写出 CSV
    written = build_events(transactions_path=source, output_dir=output)  # 构建
    assert written == output  # 返回目录
    assert (output / "year_month=2020-08").is_dir()  # 8 月分区
    assert (output / "year_month=2020-09").is_dir()  # 9 月分区
    loaded = pd.read_parquet(output)  # 回读
    loaded["item_id"] = loaded["item_id"].astype(str)  # parquet 可能是 dictionary
    assert sorted(loaded["item_id"].unique().tolist()) == ["0107360650"]  # 前导零仍在
    assert len(loaded) == 2  # 两个日期两条事件


def test_processed_layout_includes_events_dir(tmp_path: Path) -> None:  # 数据布局预留事件目录
    layout = processed_layout(tmp_path)  # 布局
    assert layout["events"] == tmp_path / "events"  # 约定路径


def test_data_command_help_documents_opt_in_build_events(capsys) -> None:  # 默认不构建，开关可见
    with pytest.raises(SystemExit) as exited:  # argparse --help
        data_main(["--help"])  # 只看帮助
    assert exited.value.code == 0  # 正常退出
    help_text = capsys.readouterr().out  # 帮助文本
    assert "--build-events" in help_text  # 可选步骤
