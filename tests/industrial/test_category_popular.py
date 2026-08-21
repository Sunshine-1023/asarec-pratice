"""Tests for category popular recall (Task 3.2)."""  # 类别热门升级

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.industrial.recall.category_popular import (  # 类别热门
    WINDOW_WEEKS,
    build_category_popular_index,
    recall_category_popular,
)


def _unix(date: str) -> int:
    return int(pd.Timestamp(f"{date} 12:00:00", tz="UTC").timestamp())


def _write_inter(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]
    lines.extend(f"{user}\t{item}\t{_unix(date)}" for user, item, date in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_category_popular_uses_four_windows() -> None:  # 与 global 对齐 1/2/4/12
    assert WINDOW_WEEKS == (1, 2, 4, 12)


def test_category_popular_prefers_recent_in_same_bucket(tmp_path: Path) -> None:  # 桶内近窗优先
    inter = tmp_path / "train.inter"
    articles = tmp_path / "articles.csv"
    _write_inter(
        inter,
        [
            ("u1", "0100000001", "2020-09-01"),
            ("u1", "0100000001", "2020-09-02"),
            ("u1", "0100000002", "2020-08-01"),
        ],
    )
    pd.DataFrame(
        {
            "article_id": ["0100000001", "0100000002"],
            "product_type_name": ["T-shirt", "T-shirt"],
            "department_name": ["Jersey", "Jersey"],
            "section_name": ["Top", "Top"],
            "garment_group_name": ["Knit", "Knit"],
            "colour_group_name": ["Blue", "Red"],
        }
    ).to_csv(articles, index=False)
    index = build_category_popular_index(inter, item_file=tmp_path / "missing.item", articles_path=articles)
    recalled = recall_category_popular(["0100000002"], index, seed_items=1, top_k=3)
    assert recalled[0][0] == "0100000001"  # 同 product_type 桶内近窗更热


def test_category_popular_respects_as_of(tmp_path: Path) -> None:  # as-of 截断
    inter = tmp_path / "train.inter"
    articles = tmp_path / "articles.csv"
    _write_inter(
        inter,
        [
            ("u1", "0100000001", "2020-09-01"),
            ("u1", "0100000009", "2020-09-10"),
        ],
    )
    pd.DataFrame(
        {
            "article_id": ["0100000001", "0100000009"],
            "product_type_name": ["T-shirt", "T-shirt"],
            "department_name": ["Jersey", "Jersey"],
            "section_name": ["Top", "Top"],
            "garment_group_name": ["Knit", "Knit"],
            "colour_group_name": ["Blue", "Blue"],
        }
    ).to_csv(articles, index=False)
    index = build_category_popular_index(inter, item_file=tmp_path / "missing.item", articles_path=articles, as_of="2020-09-05")
    all_items = {item for field in index.buckets.values() for ranked in field.values() for item, _ in ranked}
    assert "0100000001" in all_items
    assert "0100000009" not in all_items
