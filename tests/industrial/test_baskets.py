"""Tests for daily basket aggregation."""  # 按天购物篮测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 临时路径

import pandas as pd  # 表格
import pytest  # 异常

from fashionrec.industrial.data.baskets import (  # 购物篮
    BASKET_COLUMNS,  # 列契约
    baskets_from_events,  # 事件成篮
    build_baskets,  # CSV/事件 -> parquet
    flatten_recent_baskets,  # 按完整日截断
)
from fashionrec.industrial.data.events import aggregate_user_day_item_events  # 先聚合成事件
from fashionrec.industrial.data.command import main as data_main  # 帮助文本
from fashionrec.industrial.data.service import processed_layout  # 布局


def test_same_day_skus_become_one_basket_without_quantity_as_separate_items() -> None:  # 同日多 SKU 一篮
    events = aggregate_user_day_item_events(
        pd.DataFrame(
            [
                {
                    "t_dat": "2020-09-22",
                    "customer_id": "user-a",
                    "article_id": "0100000002",
                    "price": 0.2,
                    "sales_channel_id": 1,
                },
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
                    "article_id": "0100000001",
                    "price": 0.1,
                    "sales_channel_id": 2,
                },
            ]
        )
    )
    baskets = baskets_from_events(events)
    assert len(baskets) == 1  # 一天一篮
    row = baskets.iloc[0]
    assert row["user_id"] == "user-a"
    assert pd.Timestamp(row["date"]).normalize() == pd.Timestamp("2020-09-22")
    assert row["item_ids"] == "0100000001 0100000002"  # 去重；排序只为可复现
    assert int(row["n_items"]) == 2  # 两个 SKU
    assert int(row["quantity_sum"]) == 3  # 三行原始交易
    assert list(baskets.columns) == list(BASKET_COLUMNS)


def test_different_days_stay_separate_baskets() -> None:  # 跨日不合并
    events = aggregate_user_day_item_events(
        pd.DataFrame(
            [
                {
                    "t_dat": "2020-09-21",
                    "customer_id": "user-a",
                    "article_id": "0100000001",
                    "price": 0.1,
                    "sales_channel_id": 1,
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
    baskets = baskets_from_events(events)
    assert len(baskets) == 2  # 两天
    assert list(baskets["n_items"].astype(int)) == [1, 1]


def test_flatten_recent_baskets_drops_oldest_complete_days() -> None:  # 截断丢掉最旧的一整天
    history = [["A", "B", "C"], ["D"]]  # 第一天 3 件，第二天 1 件
    assert flatten_recent_baskets(history, max_item_list_length=3) == ["D"]  # 3+1>3，丢掉第一天
    assert flatten_recent_baskets(history, max_item_list_length=4) == ["A", "B", "C", "D"]  # 刚好放下
    assert flatten_recent_baskets(history, max_item_list_length=10, max_shopping_days=1) == ["D"]  # 只留最近 1 日


def test_baskets_parquet_keeps_leading_zeros_and_month_partitions(tmp_path: Path) -> None:  # 分区回读
    source = tmp_path / "transactions.csv"
    output = tmp_path / "baskets"
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
    ).to_csv(source, index=False)
    written = build_baskets(transactions_path=source, output_dir=output)
    assert written == output
    assert (output / "year_month=2020-08").is_dir()
    assert (output / "year_month=2020-09").is_dir()
    loaded = pd.read_parquet(output)
    assert sorted(loaded["item_ids"].astype(str).tolist()) == ["0107360650", "0107360650"]


def test_processed_layout_includes_baskets_dir(tmp_path: Path) -> None:  # 布局预留目录
    assert processed_layout(tmp_path)["baskets"] == tmp_path / "baskets"


def test_data_command_help_documents_opt_in_build_baskets(capsys) -> None:  # 默认不落盘
    with pytest.raises(SystemExit) as exited:
        data_main(["--help"])
    assert exited.value.code == 0
    assert "--build-baskets" in capsys.readouterr().out
