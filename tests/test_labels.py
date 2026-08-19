"""Tests for next-basket labels."""  # 未来 7 天去重标签测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格
import pytest  # 异常

from fashionrec.data.labels import (  # 标签
    LABEL_COLUMNS,  # 列契约
    build_labels,  # 落盘
    build_next_basket_labels,  # 纯函数
)
from fashionrec.data.snapshots import SnapshotSpec  # 快照
from fashionrec.data.split import TimeSplitResult  # 切分


def _events(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_labels_are_unique_user_item_and_exclude_as_of_day() -> None:  # 去重且不含预测当天
    events = _events(
        [
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-15", "quantity": 1},  # as_of 当天，只能进历史
            {"user_id": "u1", "item_id": "0000000002", "date": "2020-09-16", "quantity": 1},  # 窗口内
            {"user_id": "u1", "item_id": "0000000002", "date": "2020-09-18", "quantity": 2},  # 同 SKU 再买
            {"user_id": "u1", "item_id": "0000000003", "date": "2020-09-23", "quantity": 1},  # 窗口外
        ]
    )
    specs = [SnapshotSpec(as_of_date=pd.Timestamp("2020-09-15"), split="valid")]
    labels = build_next_basket_labels(events, specs, horizon_days=7)
    assert list(labels.columns) == list(LABEL_COLUMNS)
    assert len(labels) == 1  # 只有 SKU 2，且只一行
    row = labels.iloc[0]
    assert row["item_id"] == "0000000002"
    assert bool(row["label_purchase"]) is True
    assert int(row["label_quantity"]) == 3  # 1+2，不重复计入 AP
    assert bool(row["label_repeat"]) is False  # 历史里没有 2
    assert bool(row["label_new_to_user"]) is True
    assert pd.Timestamp(row["as_of_date"]).normalize() == pd.Timestamp("2020-09-15")


def test_repeat_and_new_to_user_are_exclusive() -> None:  # 复购与新购互斥
    events = _events(
        [
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-01", "quantity": 1},
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-16", "quantity": 1},
            {"user_id": "u1", "item_id": "0000000002", "date": "2020-09-16", "quantity": 1},
        ]
    )
    specs = [SnapshotSpec(as_of_date=pd.Timestamp("2020-09-15"), split="valid")]
    labels = build_next_basket_labels(events, specs, horizon_days=7)
    by_item = labels.set_index("item_id")
    assert bool(by_item.loc["0000000001", "label_repeat"]) is True
    assert bool(by_item.loc["0000000001", "label_new_to_user"]) is False
    assert bool(by_item.loc["0000000002", "label_repeat"]) is False
    assert bool(by_item.loc["0000000002", "label_new_to_user"]) is True
    assert not (labels["label_repeat"] & labels["label_new_to_user"]).any()


def test_same_style_new_color_requires_new_sku_of_known_product_code() -> None:  # 同款新色
    events = _events(
        [
            {"user_id": "u1", "item_id": "0100000001", "date": "2020-09-01", "quantity": 1},  # 款 100
            {"user_id": "u1", "item_id": "0100000002", "date": "2020-09-16", "quantity": 1},  # 同款新 SKU
            {"user_id": "u1", "item_id": "0100000001", "date": "2020-09-16", "quantity": 1},  # 复购旧 SKU
            {"user_id": "u1", "item_id": "0200000001", "date": "2020-09-16", "quantity": 1},  # 全新款
        ]
    )
    styles = pd.DataFrame(
        {
            "item_id": ["0100000001", "0100000002", "0200000001"],
            "product_code": ["100", "100", "200"],
        }
    )
    specs = [SnapshotSpec(as_of_date=pd.Timestamp("2020-09-15"), split="valid")]
    labels = build_next_basket_labels(events, specs, horizon_days=7, product_codes=styles)
    by_item = labels.set_index("item_id")
    assert bool(by_item.loc["0100000002", "label_same_style_new_color"]) is True
    assert bool(by_item.loc["0100000001", "label_same_style_new_color"]) is False  # 复购不是新色
    assert bool(by_item.loc["0200000001", "label_same_style_new_color"]) is False  # 全新款


def test_purchase_on_as_of_day_is_history_not_label() -> None:  # 防泄漏：当天进历史
    events = _events(
        [
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-15", "quantity": 1},
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-16", "quantity": 1},
        ]
    )
    specs = [SnapshotSpec(as_of_date=pd.Timestamp("2020-09-15"), split="valid")]
    labels = build_next_basket_labels(events, specs, horizon_days=7)
    assert len(labels) == 1
    assert bool(labels.iloc[0]["label_repeat"]) is True  # 15 日那次算历史


def test_build_labels_writes_partitioned_parquet(tmp_path: Path) -> None:  # 落盘并保前导零
    source = tmp_path / "transactions.csv"
    articles = tmp_path / "articles.csv"
    pd.DataFrame(
        [
            {
                "t_dat": "2020-09-01",
                "customer_id": "user-a",
                "article_id": "0107360650",
                "price": 0.02,
                "sales_channel_id": 2,
            },
            {
                "t_dat": "2020-09-16",
                "customer_id": "user-a",
                "article_id": "0107360651",
                "price": 0.03,
                "sales_channel_id": 2,
            },
        ]
    ).to_csv(source, index=False)
    pd.DataFrame(
        {
            "article_id": ["0107360650", "0107360651"],
            "product_code": ["107360", "107360"],
        }
    ).to_csv(articles, index=False)
    dummy = tmp_path / "unused.inter"
    dummy.write_text("user_id:token\titem_id:token\ttimestamp:float\n", encoding="utf-8")
    split = TimeSplitResult(
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
    snapshots_dir, labels_dir = build_labels(
        split=split,
        snapshots_dir=tmp_path / "snapshots",
        labels_dir=tmp_path / "labels",
        horizon_days=7,
        transactions_path=source,
        articles_path=articles,
    )
    loaded = pd.read_parquet(labels_dir)
    assert (tmp_path / "snapshots" / "as_of_date=2020-09-15").is_dir()  # test 快照 as_of=valid_end
    assert (tmp_path / "labels" / "as_of_date=2020-09-15").is_dir()
    loaded["as_of_date"] = pd.to_datetime(loaded["as_of_date"])
    test_labels = loaded[loaded["as_of_date"].dt.normalize() == pd.Timestamp("2020-09-15")]
    assert test_labels["item_id"].astype(str).tolist() == ["0107360651"]  # 前导零
    assert test_labels["split"].astype(str).tolist() == ["test"]
    assert bool(test_labels.iloc[0]["label_same_style_new_color"]) is True  # 同款新 SKU
    assert snapshots_dir.exists() and labels_dir.exists()


def test_build_labels_rejects_next_item_mode(tmp_path: Path) -> None:  # 1.3 只做 next-basket
    dummy = tmp_path / "x.inter"
    dummy.write_text("user_id:token\titem_id:token\ttimestamp:float\n", encoding="utf-8")
    split = TimeSplitResult(
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
    with pytest.raises(ValueError, match="next_basket"):
        build_labels(
            split=split,
            snapshots_dir=tmp_path / "s",
            labels_dir=tmp_path / "l",
            horizon_days=7,
            transactions_path=tmp_path / "missing.csv",
            target_mode="next_item",
        )
