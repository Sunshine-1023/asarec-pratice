"""Tests for point-in-time user behavior features (Task 2.3)."""  # as-of 用户行为统计

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格
import pytest  # 断言

from fashionrec.industrial.data.command import main as data_main
from fashionrec.industrial.data.service import processed_layout  # CLI / 布局
from fashionrec.industrial.data.snapshots import SnapshotSpec  # 快照
from fashionrec.industrial.data.split import TimeSplitResult  # 切分
from fashionrec.industrial.data.user_features import (  # 2.3
    BEHAVIOR_WINDOWS,
    USER_FEATURE_SCHEMA_VERSION,
    assert_user_features_ignore_future_events,
    build_user_feature_table,
    build_user_features,
    compute_user_feature_row,
    enrich_events,
    history_as_of,
    load_item_metadata,
)


def _articles(path: Path, rows: list[dict[str, object]]) -> Path:  # 最小 articles
    frame = pd.DataFrame(rows)
    for col in (
        "article_id",
        "product_code",
        "colour_group_name",
        "department_name",
        "product_type_name",
    ):
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame.loc[
        :,
        [
            "article_id",
            "product_code",
            "colour_group_name",
            "department_name",
            "product_type_name",
        ],
    ]
    frame.to_csv(path, index=False)
    return path


def _events(rows: list[dict[str, object]]) -> pd.DataFrame:  # 事件表
    return pd.DataFrame(rows)


def _split() -> TimeSplitResult:  # 与 test_snapshots 相同边界
    dummy = Path("unused.inter")
    return TimeSplitResult(
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


def test_history_as_of_includes_prediction_day() -> None:  # as_of 当天算历史
    events = _events(
        [
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-15", "quantity": 1},
            {"user_id": "u1", "item_id": "0000000002", "date": "2020-09-16", "quantity": 1},
        ]
    )
    hist = history_as_of(events, "2020-09-15")
    assert len(hist) == 1
    assert hist.iloc[0]["item_id"] == "0000000001"


def test_window_features_cover_planned_columns(tmp_path: Path) -> None:  # 窗口列齐全
    articles = _articles(
        tmp_path / "articles.csv",
        [
            {
                "article_id": "0100000001",
                "product_code": "100",
                "colour_group_name": "Blue",
                "department_name": "Jersey",
                "product_type_name": "T-shirt",
            },
            {
                "article_id": "0100000002",
                "product_code": "100",
                "colour_group_name": "Red",
                "department_name": "Jersey",
                "product_type_name": "T-shirt",
            },
            {
                "article_id": "0200000001",
                "product_code": "200",
                "colour_group_name": "Black",
                "department_name": "Shoes",
                "product_type_name": "Sneaker",
            },
        ],
    )
    meta = load_item_metadata(articles)
    events = enrich_events(
        _events(
            [
                {
                    "user_id": "u1",
                    "item_id": "0100000001",
                    "date": "2020-09-01",
                    "quantity": 2,
                    "mean_price": 0.02,
                    "sales_channel_mode": 1,
                },
                {
                    "user_id": "u1",
                    "item_id": "0100000002",
                    "date": "2020-09-08",
                    "quantity": 1,
                    "mean_price": 0.04,
                    "sales_channel_mode": 2,
                },
                {
                    "user_id": "u1",
                    "item_id": "0200000001",
                    "date": "2020-09-15",
                    "quantity": 1,
                    "mean_price": 0.20,
                    "sales_channel_mode": 2,
                },
            ]
        ),
        meta,
    )
    row = compute_user_feature_row("u1", events, "2020-09-15", split="valid", windows=(7, 28))
    assert row["feature_version"] == USER_FEATURE_SCHEMA_VERSION
    assert row["recency_days:float"] == 0.0  # 当天有购买
    assert row["purchase_count_7d"] == 1.0  # 7 日窗 [9/9,9/15] 仅含 9/15
    assert row["active_days_7d"] == 1.0
    assert row["basket_size_7d"] == 1.0
    assert row["channel_1_share_7d"] == 0.0
    assert row["channel_2_share_7d"] == 1.0
    assert row["style_diversity_7d"] == 1.0
    assert row["repeat_rate_7d"] == 0.0
    assert row["purchase_count_28d"] == 4.0  # 2+1+1
    assert row["same_style_rate_28d"] == 0.25  # 9/8 的 0100000002 与 9/1 同款
    assert row["department_entropy_28d"] > 0.0


def test_cold_start_user_gets_zero_windows() -> None:  # 无历史
    row = compute_user_feature_row("u-cold", _events([]), "2020-09-15", split="valid", windows=(7,))
    assert pd.isna(row["recency_days:float"])
    assert row["purchase_count_7d"] == 0.0
    assert row["avg_shopping_interval_days:float"] == 0.0


def test_build_user_feature_table_skips_users_without_history(tmp_path: Path) -> None:  # 无历史用户
    articles = _articles(
        tmp_path / "articles.csv",
        [{"article_id": "0000000001", "product_code": "1"}],
    )
    meta = load_item_metadata(articles)
    events = _events(
        [
            {"user_id": "warm", "item_id": "0000000001", "date": "2020-09-01", "quantity": 1},
            {"user_id": "future_only", "item_id": "0000000001", "date": "2020-09-20", "quantity": 1},
        ]
    )
    specs = [SnapshotSpec(as_of_date=pd.Timestamp("2020-09-15"), split="valid")]
    table = build_user_feature_table(events, specs, meta, windows=(7,))
    assert sorted(table["user_id"].tolist()) == ["warm"]
    assert len(table) == 1


def test_assert_user_features_ignore_future_events(tmp_path: Path) -> None:  # 防泄漏断言
    articles = _articles(
        tmp_path / "articles.csv",
        [
            {"article_id": "0000000001", "product_code": "100"},
            {"article_id": "0000000002", "product_code": "200"},
        ],
    )
    meta = load_item_metadata(articles)
    events = _events(
        [
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-01", "quantity": 1},
            {"user_id": "u1", "item_id": "0000000002", "date": "2020-09-16", "quantity": 5},
        ]
    )
    assert_user_features_ignore_future_events(
        events,
        user_id="u1",
        as_of="2020-09-15",
        item_metadata=meta,
        windows=(7,),
    )


def test_build_user_features_writes_partitioned_parquet(tmp_path: Path) -> None:  # 落盘
    source = tmp_path / "transactions.csv"
    articles = _articles(
        tmp_path / "articles.csv",
        [
            {"article_id": "0107360650", "product_code": "107360"},
            {"article_id": "0107360651", "product_code": "107360"},
        ],
    )
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
    output_dir = tmp_path / "user_features"
    written = build_user_features(
        split=_split(),
        output_dir=output_dir,
        horizon_days=7,
        transactions_path=source,
        articles_path=articles,
        windows=(7, 28),
    )
    assert written == output_dir
    loaded = pd.read_parquet(output_dir)
    loaded["as_of_date"] = pd.to_datetime(loaded["as_of_date"])
    test_rows = loaded[loaded["as_of_date"].dt.normalize() == pd.Timestamp("2020-09-15")]
    assert not test_rows.empty
    user_row = test_rows[test_rows["user_id"] == "user-a"].iloc[0]
    assert user_row["split"] == "test"
    assert user_row["purchase_count_7d"] >= 0.0
    assert (tmp_path / "user_features" / "as_of_date=2020-09-15").is_dir()


def test_behavior_windows_match_plan() -> None:  # 计划窗口
    assert BEHAVIOR_WINDOWS == (1, 7, 28, 84, 182, 365)


def test_processed_layout_includes_user_features(tmp_path: Path) -> None:  # 布局
    assert processed_layout(tmp_path)["user_features"] == tmp_path / "user_features"


def test_data_command_help_documents_opt_in_build_user_features(capsys) -> None:  # 默认关
    with pytest.raises(SystemExit) as exited:
        data_main(["--help"])
    assert exited.value.code == 0
    assert "--build-user-features" in capsys.readouterr().out
