"""Tests for streaming raw data profile."""  # raw 数据体检测试

from __future__ import annotations  # 延迟注解

import json  # 读写出的 JSON
from pathlib import Path  # 临时路径

import pandas as pd  # 写小 CSV
import pytest  # 帮助与退出码

from fashionrec.data.profile import build_data_profile, main, write_data_profile  # 体检入口


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:  # 写小 fixture，保留字符串 ID
    pd.DataFrame(rows).to_csv(path, index=False)  # 写出


def test_profile_reads_ids_as_strings_and_join_coverage(tmp_path: Path) -> None:  # 字符串 ID、覆盖率、价格日期
    transactions = tmp_path / "transactions.csv"  # 交易
    customers = tmp_path / "customers.csv"  # 用户
    articles = tmp_path / "articles.csv"  # 商品
    _write_csv(  # 三笔交易：缺用户、空价格、非正价格、逆序日期
        transactions,
        [
            {
                "t_dat": "2020-09-22",
                "customer_id": "user-a",
                "article_id": "0107360650",
                "price": 0.02,
                "sales_channel_id": 2,
            },
            {
                "t_dat": "2020-09-21",
                "customer_id": "missing-user",
                "article_id": "0118450001",
                "price": "",
                "sales_channel_id": 1,
            },
            {
                "t_dat": "2020-09-20",
                "customer_id": "user-a",
                "article_id": "0107360650",
                "price": 0,
                "sales_channel_id": 2,
            },
        ],
    )
    _write_csv(customers, [{"customer_id": "user-a", "FN": "", "Active": "", "age": 30}])  # 只有一个用户
    _write_csv(  # 两个 SKU，一对一款式
        articles,
        [
            {"article_id": "0107360650", "product_code": "107360", "prod_name": "tee", "detail_desc": "a"},
            {"article_id": "0118450001", "product_code": "118450", "prod_name": "jean", "detail_desc": ""},
        ],
    )
    payload = build_data_profile(transactions, customers, articles, chunk_size=2)  # 强制分块
    txn = payload["files"]["transactions"]  # 交易节
    assert txn["n_rows"] == 3  # 三行
    assert txn["n_users"] == 2  # 两用户
    assert txn["n_items"] == 2  # 两 SKU
    assert txn["ids_read_as_string"] is True  # 强制字符串
    assert txn["n_floatish_article_id"] == 0  # 前导零未变成浮点
    assert txn["price"]["n_null"] == 1  # 空价格
    assert txn["price"]["n_non_positive"] == 1  # 零价格
    assert txn["dates"]["min"] == "2020-09-20"  # 最早
    assert txn["dates"]["max"] == "2020-09-22"  # 最晚
    assert txn["dates"]["n_decreasing_adjacent"] >= 1  # 文件按日期逆序
    assert payload["join_coverage"]["txn_users_in_customers"]["n_missing"] == 1  # missing-user
    assert payload["join_coverage"]["txn_items_in_articles"]["coverage"] == 1.0  # SKU 都能对上
    assert "user_ids" not in txn  # 写出前已删除 join 集合
    out = write_data_profile(payload, tmp_path / "profile.json")  # 写出应可 JSON 序列化
    loaded = json.loads(out.read_text(encoding="utf-8"))  # 再读
    assert loaded["files"]["articles"]["n_product_codes"] == 2  # 两款式
    first = json.loads(json.dumps({k: v for k, v in payload.items() if k != "generated_at"}))  # 去掉时间
    second_payload = build_data_profile(transactions, customers, articles, chunk_size=2)  # 再扫一次
    second = json.loads(json.dumps({k: v for k, v in second_payload.items() if k != "generated_at"}))
    assert first == second  # 同一输入除 generated_at 外一致


def test_profile_detects_illegal_article_product_mapping(tmp_path: Path) -> None:  # 同一 SKU 多个款式非法
    transactions = tmp_path / "transactions.csv"  # 交易
    customers = tmp_path / "customers.csv"  # 用户
    articles = tmp_path / "articles.csv"  # 商品
    _write_csv(
        transactions,
        [{"t_dat": "2020-09-22", "customer_id": "u1", "article_id": "0100000001", "price": 0.1}],
    )
    _write_csv(customers, [{"customer_id": "u1"}])
    _write_csv(  # 同一 article_id 两个 product_code
        articles,
        [
            {"article_id": "0100000001", "product_code": "100001"},
            {"article_id": "0100000001", "product_code": "100002"},
        ],
    )
    payload = build_data_profile(transactions, customers, articles, chunk_size=1)
    mapping = payload["files"]["articles"]["article_id_to_product_code"]
    assert mapping["n_article_ids_with_conflicting_product_code"] == 1  # 冲突一次


def test_profile_cli_writes_json(tmp_path: Path) -> None:  # CLI 写出 JSON
    transactions = tmp_path / "transactions.csv"
    customers = tmp_path / "customers.csv"
    articles = tmp_path / "articles.csv"
    output = tmp_path / "out" / "data_profile.json"
    _write_csv(
        transactions,
        [{"t_dat": "2020-01-01", "customer_id": "u1", "article_id": "1", "price": 1.0}],
    )
    _write_csv(customers, [{"customer_id": "u1"}])
    _write_csv(articles, [{"article_id": "1", "product_code": "1"}])
    main(  # 走 CLI
        [
            "--transactions",
            str(transactions),
            "--customers",
            str(customers),
            "--articles",
            str(articles),
            "--output",
            str(output),
            "--chunk-size",
            "10",
        ]
    )
    assert output.exists()  # 文件写出
    payload = json.loads(output.read_text(encoding="utf-8"))  # 可读
    assert payload["files"]["transactions"]["n_rows"] == 1  # 一行


def test_profile_missing_file_raises(tmp_path: Path) -> None:  # 缺文件立即失败
    with pytest.raises(FileNotFoundError, match="transactions"):  # 明确哪张表
        build_data_profile(tmp_path / "missing.csv", tmp_path / "c.csv", tmp_path / "a.csv")
