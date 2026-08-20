"""Tests for multi-window popular recall (Task 3.2)."""  # 热门召回升级

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格
import pytest  # 断言

from fashionrec.recall.popular import (  # 热门
    WINDOW_WEEKS,
    build_popular_index,
    build_user_cohort_lookup,
    recall_popular,
)
from fashionrec.recall.window_scores import rank_normalize_counts  # rank 归一化
from collections import Counter  # 计数


def _unix(date: str) -> int:  # 日期转 Unix 秒
    return int(pd.Timestamp(f"{date} 12:00:00", tz="UTC").timestamp())


def _write_inter(path: Path, rows: list[tuple[str, str, str]]) -> None:  # user,item,date
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]
    lines.extend(f"{user}\t{item}\t{_unix(date)}" for user, item, date in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_window_weeks_use_plan_defaults() -> None:  # 1/2/4/12 周
    assert WINDOW_WEEKS == (1, 2, 4, 12)


def test_rank_normalize_prefers_hotter_items() -> None:  # rank 归一化
    scores = rank_normalize_counts(Counter({"0000000001": 10, "0000000002": 1}))
    assert scores["0000000001"] > scores["0000000002"]


def test_recent_item_outranks_old_when_using_multi_window(tmp_path: Path) -> None:  # 短窗趋势
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000001", "2020-09-01"),
            ("u1", "0000000001", "2020-09-02"),
            ("u1", "0000000002", "2020-08-01"),
        ],
    )
    index = build_popular_index(inter, customers_path=None, articles_path=None)
    ranked = list(index)
    assert ranked[0][0] == "0000000001"  # 近窗更热


def test_as_of_excludes_future_interactions(tmp_path: Path) -> None:  # as-of 防泄漏
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000001", "2020-09-01"),
            ("u1", "0000000009", "2020-09-10"),
        ],
    )
    index = build_popular_index(inter, as_of="2020-09-05", customers_path=None, articles_path=None)
    ids = {item for item, _score in index.global_ranked}
    assert "0000000001" in ids
    assert "0000000009" not in ids


def test_cold_start_uses_age_bucket_cohort(tmp_path: Path) -> None:  # 冷启动人群热门
    inter = tmp_path / "train.inter"
    customers = tmp_path / "customers.csv"
    _write_inter(
        inter,
        [
            ("young", "0000000001", "2020-09-01"),
            ("young", "0000000001", "2020-09-02"),
            ("old", "0000000002", "2020-09-01"),
        ],
    )
    pd.DataFrame(
        [
            {"customer_id": "young", "age": "25", "FN": "", "Active": "", "club_member_status": "", "fashion_news_frequency": "", "postal_code": ""},
            {"customer_id": "old", "age": "55", "FN": "", "Active": "", "club_member_status": "", "fashion_news_frequency": "", "postal_code": ""},
            {"customer_id": "cold", "age": "25", "FN": "", "Active": "", "club_member_status": "", "fashion_news_frequency": "", "postal_code": ""},
        ]
    ).to_csv(customers, index=False)
    index = build_popular_index(inter, customers_path=customers, articles_path=None)
    lookup = build_user_cohort_lookup(customers)
    cold = recall_popular(index, user_history=set(), user_id="cold", cohort_lookup=lookup, top_k=1)
    assert cold[0][0] == "0000000001"  # 与同 age_bucket 的 young 一致


def test_warm_user_uses_global_popular(tmp_path: Path) -> None:  # 有历史仍走 global
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000001", "2020-09-01"),
            ("u2", "0000000002", "2020-09-01"),
            ("u2", "0000000002", "2020-09-02"),
        ],
    )
    index = build_popular_index(inter, customers_path=None, articles_path=None)
    warm = recall_popular(index, user_history={"0000000009"}, user_id="u1", top_k=1)
    global_top = index.global_ranked[0][0]
    assert warm[0][0] == global_top  # 全局 Top-1
