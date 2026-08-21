"""Tests for user-item cross features (Task 2.4)."""  # 用户×商品交叉特征

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格
import pytest  # 断言

from fashionrec.industrial.data.command import main as data_main
from fashionrec.industrial.data.service import processed_layout  # CLI / 布局
from fashionrec.industrial.data.cross_features import (  # 2.4
    CROSS_FEATURE_SCHEMA_VERSION,
    PAIR_COLUMNS,
    assert_cross_features_ignore_future_events,
    build_cross_feature_table,
    build_cross_features,
    compute_cross_feature_row,
    enrich_events,
    load_cross_feature_pairs,
    load_item_metadata,
)
from fashionrec.industrial.data.labels import build_labels  # 标签对
from fashionrec.industrial.data.split import TimeSplitResult  # 切分


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
    frame.to_csv(path, index=False)
    return path


def _events(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _split() -> TimeSplitResult:
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


def test_cross_feature_row_counts_recency_and_style_flags(tmp_path: Path) -> None:  # 计数/间隔/同款
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
        ],
    )
    meta = load_item_metadata(articles)
    enriched = enrich_events(
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
                    "date": "2020-09-15",
                    "quantity": 1,
                    "mean_price": 0.04,
                    "sales_channel_mode": 2,
                },
            ]
        ),
        meta,
    )
    as_of = pd.Timestamp("2020-09-15")
    global_hist = enriched.copy()
    user_hist = enriched[enriched["user_id"] == "u1"].copy()
    item_row = meta.set_index("item_id").loc["0100000002"]
    row = compute_cross_feature_row(
        "u1",
        "0100000002",
        as_of,
        user_hist=user_hist,
        item_row=item_row,
        global_hist=global_hist,
        split="valid",
    )
    assert row["feature_version"] == CROSS_FEATURE_SCHEMA_VERSION
    assert row["user_item_purchase_count"] == 1.0  # 当天 SKU
    assert row["user_style_purchase_count"] == 3.0  # 2+1 同款
    assert row["user_item_recency_days:float"] == 0.0
    assert row["user_style_recency_days:float"] == 0.0
    assert row["user_bought_same_style:float"] == 1.0
    assert row["candidate_same_style_new_color:float"] == 0.0  # 已买过该 SKU
    assert row["department_preference_match:float"] == 1.0  # 全 Jersey
    assert row["price_diff:float"] == pytest.approx(0.04 - 0.026666666666666666, rel=1e-3)


def test_same_style_new_color_flag_for_unseen_sku(tmp_path: Path) -> None:  # 同款新色
    articles = _articles(
        tmp_path / "articles.csv",
        [
            {"article_id": "0100000001", "product_code": "100", "department_name": "Jersey", "colour_group_name": "Blue"},
            {"article_id": "0100000002", "product_code": "100", "department_name": "Jersey", "colour_group_name": "Red"},
        ],
    )
    meta = load_item_metadata(articles)
    enriched = enrich_events(
        _events(
            [
                {"user_id": "u1", "item_id": "0100000001", "date": "2020-09-01", "quantity": 1, "mean_price": 0.02, "sales_channel_mode": 1},
            ]
        ),
        meta,
    )
    as_of = pd.Timestamp("2020-09-15")
    row = compute_cross_feature_row(
        "u1",
        "0100000002",
        as_of,
        user_hist=enriched,
        item_row=meta.set_index("item_id").loc["0100000002"],
        global_hist=enriched,
    )
    assert row["user_bought_same_style:float"] == 1.0
    assert row["candidate_same_style_new_color:float"] == 1.0
    assert row["user_item_purchase_count"] == 0.0


def test_build_cross_feature_table_primary_keys(tmp_path: Path) -> None:  # 主键齐全
    articles = _articles(
        tmp_path / "articles.csv",
        [{"article_id": "0000000001", "product_code": "1", "department_name": "A", "colour_group_name": "Blue"}],
    )
    meta = load_item_metadata(articles)
    events = _events(
        [{"user_id": "u1", "item_id": "0000000001", "date": "2020-09-01", "quantity": 1, "mean_price": 0.02, "sales_channel_mode": 1}]
    )
    pairs = pd.DataFrame(
        [{"user_id": "u1", "item_id": "0000000001", "as_of_date": "2020-09-15", "split": "valid"}]
    )
    table = build_cross_feature_table(pairs, events, meta)
    for col in (*PAIR_COLUMNS, "feature_version"):
        assert col in table.columns
    assert len(table) == 1
    assert table.iloc[0]["item_global_purchase_count_7d"] == 0.0  # 7 日窗不含 9/1


def test_cohort_features_use_full_customer_cohort_not_candidate_batch(tmp_path: Path) -> None:
    articles = _articles(
        tmp_path / "articles.csv",
        [{"article_id": "0000000001", "product_code": "1", "department_name": "A", "colour_group_name": "Blue"}],
    )
    meta = load_item_metadata(articles)
    events = _events(
        [
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-10", "quantity": 2, "mean_price": 0.02, "sales_channel_mode": 1},
            {"user_id": "u2", "item_id": "0000000001", "date": "2020-09-10", "quantity": 3, "mean_price": 0.02, "sales_channel_mode": 1},
        ]
    )
    pairs = pd.DataFrame([{"user_id": "u1", "item_id": "0000000001", "as_of_date": "2020-09-10"}])
    cohorts = pd.DataFrame({"user_id": ["u1", "u2"], "age_bucket": ["25-34", "25-34"]})
    table = build_cross_feature_table(pairs, events, meta, user_cohorts=cohorts)
    assert float(table.iloc[0]["item_cohort_purchase_count_7d"]) == 5.0


def test_assert_cross_features_ignore_future_events(tmp_path: Path) -> None:  # 防泄漏
    articles = _articles(
        tmp_path / "articles.csv",
        [
            {"article_id": "0000000001", "product_code": "100"},
            {"article_id": "0000000009", "product_code": "900"},
        ],
    )
    meta = load_item_metadata(articles)
    events = _events(
        [
            {"user_id": "u1", "item_id": "0000000001", "date": "2020-09-01", "quantity": 1, "mean_price": 0.02, "sales_channel_mode": 1},
            {"user_id": "u1", "item_id": "0000000009", "date": "2020-09-16", "quantity": 10, "mean_price": 0.50, "sales_channel_mode": 2},
        ]
    )
    pairs = pd.DataFrame([{"user_id": "u1", "item_id": "0000000001", "as_of_date": "2020-09-15"}])
    assert_cross_features_ignore_future_events(events, pairs, meta)


def test_build_cross_features_from_labels_writes_parquet(tmp_path: Path) -> None:  # 落盘
    source = tmp_path / "transactions.csv"
    articles = _articles(
        tmp_path / "articles.csv",
        [
            {"article_id": "0107360650", "product_code": "107360", "department_name": "Jersey", "colour_group_name": "Blue"},
            {"article_id": "0107360651", "product_code": "107360", "department_name": "Jersey", "colour_group_name": "Red"},
        ],
    )
    pd.DataFrame(
        [
            {"t_dat": "2020-09-01", "customer_id": "user-a", "article_id": "0107360650", "price": 0.02, "sales_channel_id": 2},
            {"t_dat": "2020-09-16", "customer_id": "user-a", "article_id": "0107360651", "price": 0.03, "sales_channel_id": 2},
        ]
    ).to_csv(source, index=False)
    split = _split()
    labels_dir = tmp_path / "labels"
    build_labels(
        split=split,
        snapshots_dir=tmp_path / "snapshots",
        labels_dir=labels_dir,
        horizon_days=7,
        transactions_path=source,
        articles_path=articles,
    )
    output_dir = tmp_path / "cross_features"
    written = build_cross_features(
        output_dir=output_dir,
        transactions_path=source,
        articles_path=articles,
        labels_dir=labels_dir,
        customers_path=None,  # 单测不依赖 raw customers
    )
    assert written == output_dir
    loaded = pd.read_parquet(output_dir)
    assert set(PAIR_COLUMNS).issubset(loaded.columns)
    assert loaded["feature_version"].iloc[0] == CROSS_FEATURE_SCHEMA_VERSION
    assert (output_dir / "as_of_date=2020-09-15").is_dir()


def test_load_pairs_from_candidates_csv(tmp_path: Path) -> None:  # 显式候选
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [{"user_id": "user-a", "item_id": "0107360651", "as_of_date": "2020-09-15"}]
    ).to_csv(path, index=False)
    pairs = load_cross_feature_pairs(candidates_path=path)
    assert len(pairs) == 1
    assert pairs.iloc[0]["item_id"] == "0107360651"


def test_processed_layout_includes_cross_features(tmp_path: Path) -> None:  # 布局
    assert processed_layout(tmp_path)["cross_features"] == tmp_path / "cross_features"


def test_data_command_help_documents_opt_in_build_cross_features(capsys) -> None:  # 默认关
    with pytest.raises(SystemExit) as exited:
        data_main(["--help"])
    assert exited.value.code == 0
    assert "--build-cross-features" in capsys.readouterr().out
