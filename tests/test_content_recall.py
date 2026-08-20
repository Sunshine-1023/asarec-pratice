"""Tests for content recall (Task 3.4)."""  # 内容召回

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.recall.content import build_content_index, recall_content


def _unix(date: str) -> int:
    return int(pd.Timestamp(f"{date} 12:00:00", tz="UTC").timestamp())


def _write_inter(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]
    lines.extend(f"{user}\t{item}\t{_unix(date)}" for user, item, date in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_articles(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def test_content_prefers_similar_text_and_category(tmp_path: Path) -> None:  # 文本+类别相似
    inter = tmp_path / "train.inter"
    articles = tmp_path / "articles.csv"
    _write_inter(
        inter,
        [
            ("u1", "0100000001", "2020-09-01"),
            ("u1", "0100000002", "2020-08-01"),
        ],
    )
    _write_articles(
        articles,
        pd.DataFrame(
            {
                "article_id": ["0100000001", "0100000002", "0100000003"],
                "prod_name": ["blue cotton t-shirt", "red wool sweater", "blue cotton tee"],
                "detail_desc": ["soft jersey top", "warm knit", "soft jersey shirt"],
                "product_type_name": ["T-shirt", "Sweater", "T-shirt"],
                "product_group_name": ["Garment", "Garment", "Garment"],
                "colour_group_name": ["Blue", "Red", "Blue"],
                "section_name": ["Top", "Top", "Top"],
                "garment_group_name": ["Knit", "Knit", "Knit"],
                "department_name": ["Jersey", "Jersey", "Jersey"],
                "index_name": ["Ladies", "Ladies", "Ladies"],
                "index_group_name": ["Womens", "Womens", "Womens"],
            }
        ),
    )
    index = build_content_index(articles, inter_paths=inter)
    recalled = recall_content(["0100000001"], index, seed_items=1, top_k=3)
    assert recalled
    assert recalled[0][0] == "0100000003"  # 与 seed 文本/类别更接近


def test_content_cold_start_uses_popularity_fallback(tmp_path: Path) -> None:  # 冷启动 fallback
    inter = tmp_path / "train.inter"
    articles = tmp_path / "articles.csv"
    _write_inter(
        inter,
        [
            ("u1", "0100000001", "2020-09-01"),
            ("u1", "0100000001", "2020-09-02"),
            ("u2", "0100000002", "2020-09-01"),
        ],
    )
    _write_articles(
        articles,
        pd.DataFrame(
            {
                "article_id": ["0100000001", "0100000002"],
                "prod_name": ["item one", "item two"],
                "detail_desc": ["desc one", "desc two"],
                "product_type_name": ["A", "B"],
                "product_group_name": ["G", "G"],
                "colour_group_name": ["Blue", "Red"],
                "section_name": ["S", "S"],
                "garment_group_name": ["K", "K"],
                "department_name": ["D", "D"],
                "index_name": ["I", "I"],
                "index_group_name": ["IG", "IG"],
            }
        ),
    )
    index = build_content_index(articles, inter_paths=inter)
    recalled = recall_content([], index, top_k=2)
    assert recalled
    assert recalled[0][0] == "0100000001"  # 更热


def test_content_as_of_excludes_future_popularity(tmp_path: Path) -> None:  # as-of 防泄漏
    inter = tmp_path / "train.inter"
    articles = tmp_path / "articles.csv"
    _write_inter(
        inter,
        [
            ("u1", "0100000001", "2020-09-01"),
            ("u1", "0100000009", "2020-09-10"),
        ],
    )
    _write_articles(
        articles,
        pd.DataFrame(
            {
                "article_id": ["0100000001", "0100000009"],
                "prod_name": ["seed item", "future hot item"],
                "detail_desc": ["desc", "desc"],
                "product_type_name": ["A", "A"],
                "product_group_name": ["G", "G"],
                "colour_group_name": ["Blue", "Blue"],
                "section_name": ["S", "S"],
                "garment_group_name": ["K", "K"],
                "department_name": ["D", "D"],
                "index_name": ["I", "I"],
                "index_group_name": ["IG", "IG"],
            }
        ),
    )
    index = build_content_index(articles, inter_paths=inter, as_of="2020-09-05")
    recalled = recall_content([], index, top_k=2)
    ids = {item for item, _ in recalled}
    assert "0100000001" in ids
    assert "0100000009" not in ids
