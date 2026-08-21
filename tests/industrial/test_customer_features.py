"""Tests for user static features from customers.csv."""  # 2.2 用户静态特征

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格
import pytest  # 断言

from fashionrec.industrial.data.service import processed_layout  # 布局
from fashionrec.industrial.data.customer_features import (  # 用户特征
    CUSTOMER_FEATURE_SCHEMA_VERSION,
    age_bucket_token,
    build_customer_feature_table,
    build_customer_features,
    load_customers_table,
    parse_age,
    parse_binary_flag,
    postal_frequency_bucket,
    postal_hash_bucket,
    write_customer_features_parquet,
)


def _customers_csv(path: Path, rows: list[dict[str, object]]) -> Path:  # 写 customers
    frame = pd.DataFrame(rows)  # 行
    for col in ("customer_id", "FN", "Active", "club_member_status", "fashion_news_frequency", "age", "postal_code"):
        if col not in frame.columns:  # 缺列
            frame[col] = pd.NA  # 补空
    frame = frame.loc[
        :,
        ["customer_id", "FN", "Active", "club_member_status", "fashion_news_frequency", "age", "postal_code"],
    ]
    frame.to_csv(path, index=False)  # 写出
    return path  # 返回


def _write_inter(path: Path, user_ids: list[str]) -> None:  # 最小交互
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]  # 表头
    for index, user_id in enumerate(user_ids):  # 每用户一行
        lines.append(f"{user_id}\t0000000001\t{index + 1}.0")  # 占位
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写出


def test_fn_active_three_state_mapping() -> None:  # FN/Active：1、0/空、missing 仅 unknown
    assert parse_binary_flag("1") == (1.0, 0.0)  # 明确 1
    assert parse_binary_flag("") == (0.0, 0.0)  # 空 = 0/空缺态
    assert parse_binary_flag("0") == (0.0, 0.0)  # 明确 0
    assert parse_binary_flag(pd.NA) == (0.0, 0.0)  # H&M 空单元格 = 未订阅
    assert parse_binary_flag(None) == (0.0, 0.0)  # 同上


def test_age_bucket_and_missing_indicator() -> None:  # 年龄分桶
    assert parse_age(30) == (30.0, "25_34", 0.0)
    assert parse_age("45") == (45.0, "45_54", 0.0)
    assert parse_age(pd.NA) == (0.0, "unknown", 1.0)  # 不填均值
    assert age_bucket_token(17) == "under_18"
    assert age_bucket_token(65) == "65_plus"


def test_postal_code_uses_hash_and_frequency_buckets_not_continuous() -> None:  # 邮编分桶
    assert postal_hash_bucket("12345") == postal_hash_bucket("12345")  # 稳定
    assert postal_hash_bucket("12345") != postal_hash_bucket("54321")  # 不同码不同桶
    assert postal_frequency_bucket(1) == "singleton"
    assert postal_frequency_bucket(3) == "rare"
    assert postal_frequency_bucket(20) == "medium"
    assert postal_frequency_bucket(100) == "common"


def test_build_customer_feature_table_maps_all_fields(tmp_path: Path) -> None:  # 全字段
    source = _customers_csv(
        tmp_path / "customers.csv",
        [
            {
                "customer_id": "user-a",
                "FN": "1",
                "Active": "",
                "club_member_status": "ACTIVE",
                "fashion_news_frequency": "Regularly",
                "age": "34",
                "postal_code": "12345",
            },
            {  # 目录长尾，不在交互里也要留
                "customer_id": "user-catalog",
                "age": "20",
                "postal_code": "99999",
            },
        ],
    )
    customers = load_customers_table(source)
    features = build_customer_feature_table(customers, extra_user_ids={"user-a"}, keep_full_customer_universe=True)
    assert set(features["user_id"]) == {"user-a", "user-catalog"}  # 全量目录
    row = features.set_index("user_id").loc["user-a"]
    assert float(row["age:float"]) == 34.0
    assert row["age_bucket:token"] == "25_34"
    assert float(row["age_missing:float"]) == 0.0
    assert float(row["FN:float"]) == 1.0
    assert float(row["FN_missing:float"]) == 0.0
    assert float(row["Active:float"]) == 0.0  # 空 = 0 态
    assert float(row["Active_missing:float"]) == 0.0
    assert row["club_member_status:token"] == "ACTIVE"
    assert float(row["club_member_status_missing:float"]) == 0.0
    assert row["fashion_news_frequency:token"] == "Regularly"
    assert float(row["fashion_news_frequency_missing:float"]) == 0.0
    assert row["postal_code:token"] == "12345"
    assert row["postal_code_hash_bucket:token"].startswith("h")
    assert row["postal_code_freq_bucket:token"] == "singleton"  # 只出现一次
    assert float(row["postal_code_missing:float"]) == 0.0
    assert row["feature_version"] == CUSTOMER_FEATURE_SCHEMA_VERSION


def test_missing_metadata_is_unknown_row_not_dropped() -> None:  # 未见用户补 unknown
    customers = pd.DataFrame(
        {
            "user_id": ["user-a"],
            "customer_id": ["user-a"],
            "FN": [pd.NA],
            "Active": [pd.NA],
            "club_member_status": [pd.NA],
            "fashion_news_frequency": [pd.NA],
            "age": [pd.NA],
            "postal_code": [pd.NA],
        }
    )
    features = build_customer_feature_table(
        customers,
        extra_user_ids={"user-a", "ghost-user"},
        keep_full_customer_universe=True,
    )
    by_id = features.set_index("user_id")
    assert "ghost-user" in by_id.index  # 不删
    ghost = by_id.loc["ghost-user"]
    assert float(ghost["is_unknown_customer:float"]) == 1.0
    assert float(ghost["FN_missing:float"]) == 1.0
    assert float(ghost["Active_missing:float"]) == 1.0
    assert float(ghost["age_missing:float"]) == 1.0
    assert ghost["postal_code_hash_bucket:token"] == "unknown"


def test_keep_full_customer_universe_false_still_backfills_unknown() -> None:  # 快速实验
    customers = pd.DataFrame(
        {
            "user_id": ["user-a", "user-b"],
            "customer_id": ["user-a", "user-b"],
            "FN": ["1", "0"],
            "Active": ["1", "0"],
            "club_member_status": ["ACTIVE", "ACTIVE"],
            "fashion_news_frequency": ["NONE", "NONE"],
            "age": ["30", "40"],
            "postal_code": ["111", "222"],
        }
    )
    features = build_customer_feature_table(
        customers,
        extra_user_ids={"user-a", "ghost"},
        keep_full_customer_universe=False,
    )
    assert set(features["user_id"]) == {"user-a", "ghost"}  # user-b 不在交互里被去掉
    assert float(features.set_index("user_id").loc["ghost", "is_unknown_customer:float"]) == 1.0


def test_postal_frequency_bucket_uses_full_catalog_counts() -> None:  # 频次在全表上算
    customers = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "customer_id": ["u1", "u2", "u3"],
            "FN": [pd.NA, pd.NA, pd.NA],
            "Active": [pd.NA, pd.NA, pd.NA],
            "club_member_status": [pd.NA, pd.NA, pd.NA],
            "fashion_news_frequency": [pd.NA, pd.NA, pd.NA],
            "age": [pd.NA, pd.NA, pd.NA],
            "postal_code": ["shared", "shared", "solo"],
        }
    )
    features = build_customer_feature_table(customers, extra_user_ids={"u1"})
    by_id = features.set_index("user_id")
    assert by_id.loc["u1", "postal_code_freq_bucket:token"] == "rare"  # shared 出现 2 次
    assert by_id.loc["u3", "postal_code_freq_bucket:token"] == "singleton"


def test_build_customer_features_writes_parquet(tmp_path: Path) -> None:  # 落盘
    customers = _customers_csv(
        tmp_path / "customers.csv",
        [{"customer_id": "user-a", "age": "25", "postal_code": "10001"}],
    )
    train = tmp_path / "train.inter"
    valid = tmp_path / "valid.inter"
    test = tmp_path / "test.inter"
    _write_inter(train, ["user-a", "missing-user"])
    _write_inter(valid, ["user-a"])
    _write_inter(test, ["user-a"])
    out = tmp_path / "customer_features" / "customers.parquet"
    build_customer_features(
        customers_path=customers,
        output_path=out,
        inter_paths=(train, valid, test),
        keep_full_customer_universe=True,
    )
    loaded = pd.read_parquet(out)
    assert set(loaded["user_id"]) == {"user-a", "missing-user"}
    assert loaded["feature_version"].unique().tolist() == [CUSTOMER_FEATURE_SCHEMA_VERSION]


def test_write_customer_features_parquet_accepts_directory(tmp_path: Path) -> None:  # 目录写出
    customers = pd.DataFrame(
        {
            "user_id": ["u1"],
            "customer_id": ["u1"],
            "FN": [pd.NA],
            "Active": [pd.NA],
            "club_member_status": [pd.NA],
            "fashion_news_frequency": [pd.NA],
            "age": [pd.NA],
            "postal_code": [pd.NA],
        }
    )
    features = build_customer_feature_table(customers)
    written = write_customer_features_parquet(features, tmp_path / "customer_features")
    assert written.name == "customers.parquet"
    assert written.parent == tmp_path / "customer_features"


def test_processed_layout_includes_customer_features(tmp_path: Path) -> None:  # 布局
    assert processed_layout(tmp_path)["customer_features"] == tmp_path / "customer_features" / "customers.parquet"


def test_load_customers_table_requires_customer_id(tmp_path: Path) -> None:  # 主键
    bad = tmp_path / "bad.csv"
    bad.write_text("FN\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="customer_id"):
        load_customers_table(bad)
