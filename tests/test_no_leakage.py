"""Leakage tests: recall indexes and as-of features must not read the label week."""  # 防泄漏测试

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格
import pytest  # 断言

from fashionrec.data.split import (  # 切分与 as-of 特征
    assert_history_has_no_future,  # 未来行必须失败
    assert_history_paths_allowed,  # 路径检查
    filter_interactions_as_of,  # 静默截断
    item_popularity_as_of,  # 热度
    user_item_counts_as_of,  # 用户偏好
    validate_time_split,  # 时间因果
)  # 导入结束
from fashionrec.data.user_features import (  # 2.3 as-of 用户行为
    assert_user_features_ignore_future_events,
    load_item_metadata,
)
from fashionrec.data.cross_features import assert_cross_features_ignore_future_events  # 2.4 交叉
from fashionrec.data.filter import fit_train_item_universe
from fashionrec.recall.popular import build_popular_index  # 热门索引


def _unix(date: str) -> int:  # 日期转 Unix 秒（UTC）
    return int(pd.Timestamp(f"{date} 12:00:00", tz="UTC").timestamp())  # 中午


def _write_inter(path: Path, rows: list[tuple[str, str, str]]) -> None:  # 写 user,item,date
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]  # 表头
    lines.extend(f"{user}\t{item}\t{_unix(date)}" for user, item, date in rows)  # 行
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写出


def test_validate_time_split_fails_when_future_mixed_into_train() -> None:  # 未来记录混入训练必须失败
    train = pd.DataFrame(  # 训练
        {"user_id:token": ["u"], "item_id:token": ["a"], "timestamp:float": [10.0]}  # 一行
    )  # 训练结束
    valid = pd.DataFrame(  # 验证
        {"user_id:token": ["u"], "item_id:token": ["b"], "timestamp:float": [20.0]}  # 一行
    )  # 验证结束
    test = pd.DataFrame(  # 测试
        {"user_id:token": ["u"], "item_id:token": ["c"], "timestamp:float": [30.0]}  # 一行
    )  # 测试结束
    validate_time_split(train, valid, test)  # 正常切分应通过
    leaked = pd.concat([train, valid], ignore_index=True)  # 把验证行混进训练
    with pytest.raises(AssertionError, match="train"):  # 必须失败
        validate_time_split(leaked, valid, test)  # 检查泄漏


def test_valid_history_paths_reject_valid_and_test(tmp_path: Path) -> None:  # valid 索引不能读 valid/test 文件
    train = tmp_path / "hm.train.inter"  # 训练
    valid = tmp_path / "hm.valid.inter"  # 验证
    test = tmp_path / "hm.test.inter"  # 测试
    for path in (train, valid, test):  # 创建空文件以便 resolve
        path.write_text("user_id:token\titem_id:token\ttimestamp:float\n", encoding="utf-8")  # 表头
    assert_history_paths_allowed("valid", [train], train, valid, test)  # 只用 train 合法
    with pytest.raises(AssertionError, match="leak"):  # 混入 valid
        assert_history_paths_allowed("valid", [train, valid], train, valid, test)  # 应失败
    with pytest.raises(AssertionError, match="leak"):  # 混入 test
        assert_history_paths_allowed("test", [train, valid, test], train, valid, test)  # 应失败
    assert_history_paths_allowed("test", [train, valid], train, valid, test)  # test 可用 train+valid


def test_popular_index_does_not_see_future_items(tmp_path: Path) -> None:  # 热门索引只统计传入的历史文件
    train = tmp_path / "train.inter"  # 训练
    valid = tmp_path / "valid.inter"  # 验证（标签周）
    _write_inter(train, [("u1", "old_item", "2020-09-01")] * 2)  # 历史热门
    _write_inter(valid, [("u1", "future_hot", "2020-09-10")] * 20)  # 标签周爆款
    allowed = build_popular_index(train)  # 正确：valid 评估只用 train
    leaked = build_popular_index(train, valid)  # 错误：把标签周算进热门
    allowed_ids = {item for item, _score in allowed}  # 合法索引中的商品
    leaked_ids = {item for item, _score in leaked}  # 泄漏索引中的商品
    assert "old_item" in allowed_ids  # 历史商品在
    assert "future_hot" not in allowed_ids  # 标签周商品不得出现
    assert "future_hot" in leaked_ids  # 对照：一旦混入未来，测试能抓到


def test_as_of_features_ignore_label_week() -> None:  # 热度与用户偏好必须带 as_of_date
    as_of = float(_unix("2020-09-09"))  # 验证周开始，标签周不得进入
    df = pd.DataFrame(  # 历史 + 标签周
        {  # 列
            "user_id:token": ["u1", "u1", "u1"],  # 同一用户
            "item_id:token": ["hist", "hist", "label_only"],  # 历史商品与标签周商品
            "timestamp:float": [_unix("2020-09-01"), _unix("2020-09-08"), _unix("2020-09-09")],  # 时间
        }  # 列结束
    )  # 表结束
    popularity = item_popularity_as_of(df, as_of)  # 预测时刻之前的热度
    prefs = user_item_counts_as_of(df, as_of)  # 预测时刻之前的偏好
    assert popularity == {"hist": 2}  # 只统计历史两次
    assert prefs == {("u1", "hist"): 2}  # 标签周商品不进入用户偏好
    assert "label_only" not in popularity  # 明确不读标签周


def test_history_features_fail_when_future_rows_are_passed_as_history() -> None:  # 传 future 必须失败
    as_of = float(_unix("2020-09-09"))  # 验证周开始
    df = pd.DataFrame(  # 历史 + 标签周
        {  # 列
            "user_id:token": ["u1", "u1"],  # 同一用户
            "item_id:token": ["hist", "label_only"],  # 历史与未来
            "timestamp:float": [_unix("2020-09-01"), _unix("2020-09-09")],  # 时间
        }  # 列结束
    )  # 表结束
    with pytest.raises(AssertionError, match="future"):  # 硬失败
        assert_history_has_no_future(df, as_of)  # 混入标签周
    with pytest.raises(AssertionError, match="future"):  # 特征入口同样失败
        item_popularity_as_of(df, as_of, strict=True)  # 不得静默丢掉
    with pytest.raises(AssertionError, match="future"):  # 用户偏好
        user_item_counts_as_of(df, as_of, strict=True)  # 不得静默丢掉
    hist = filter_interactions_as_of(df, as_of)  # 先截断
    assert_history_has_no_future(hist, as_of)  # 截断后应通过
    assert item_popularity_as_of(hist, as_of, strict=True) == {"hist": 1}  # 只剩历史


def test_optional_item_sampling_is_fitted_on_train_only() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": ["u1"] * 8,
            "article_id": ["train_item", "train_item", *(["future_hot"] * 6)],
            "t_dat": ["2020-09-01", "2020-09-02", *(["2020-09-10"] * 6)],
        }
    )

    selected = fit_train_item_universe(
        transactions,
        window_start=pd.Timestamp("2020-08-13"),
        valid_start=pd.Timestamp("2020-09-09"),
        top_items=1,
    )

    assert selected == {"train_item"}
    assert "future_hot" not in selected


def test_user_behavior_features_ignore_label_week(tmp_path: Path) -> None:  # 2.3 混入标签周不得改 as-of
    articles = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000009"],
            "product_code": ["100", "900"],
            "colour_group_name": ["Blue", "Red"],
            "department_name": ["Jersey", "Shoes"],
            "product_type_name": ["T-shirt", "Sneaker"],
        }
    ).to_csv(articles, index=False)
    meta = load_item_metadata(articles)
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "0000000001",
                "date": "2020-09-08",
                "quantity": 2,
                "mean_price": 0.02,
                "sales_channel_mode": 1,
            },
            {
                "user_id": "u1",
                "item_id": "0000000009",
                "date": "2020-09-16",
                "quantity": 99,
                "mean_price": 0.50,
                "sales_channel_mode": 2,
            },
        ]
    )
    assert_user_features_ignore_future_events(
        events,
        user_id="u1",
        as_of="2020-09-15",
        item_metadata=meta,
        windows=(7, 28),
    )


def test_cross_features_ignore_label_week(tmp_path: Path) -> None:  # 2.4 标签周不得改变交叉特征
    articles = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "article_id": ["0000000001", "0000000009"],
            "product_code": ["100", "900"],
            "colour_group_name": ["Blue", "Red"],
            "department_name": ["Jersey", "Shoes"],
            "product_type_name": ["T-shirt", "Sneaker"],
        }
    ).to_csv(articles, index=False)
    meta = load_item_metadata(articles)
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "0000000001",
                "date": "2020-09-08",
                "quantity": 2,
                "mean_price": 0.02,
                "sales_channel_mode": 1,
            },
            {
                "user_id": "u1",
                "item_id": "0000000009",
                "date": "2020-09-16",
                "quantity": 99,
                "mean_price": 0.50,
                "sales_channel_mode": 2,
            },
        ]
    )
    pairs = pd.DataFrame([{"user_id": "u1", "item_id": "0000000001", "as_of_date": "2020-09-15"}])
    assert_cross_features_ignore_future_events(events, pairs, meta)
