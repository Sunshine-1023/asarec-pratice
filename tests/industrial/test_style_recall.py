"""Tests for style recall (Task 3.4)."""  # 款式召回

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.industrial.recall.style import build_style_index, recall_style


def _unix(date: str) -> int:
    return int(pd.Timestamp(f"{date} 12:00:00", tz="UTC").timestamp())


def _write_inter(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]
    lines.extend(f"{user}\t{item}\t{_unix(date)}" for user, item, date in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_articles(path: Path, rows: list[tuple[str, str]]) -> None:
    pd.DataFrame({"article_id": [r[0] for r in rows], "product_code": [r[1] for r in rows]}).to_csv(path, index=False)


def test_style_suggests_sibling_sku_under_same_product_code(tmp_path: Path) -> None:  # 同款新色
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
        [
            ("0100000001", "STYLE001"),
            ("0100000002", "STYLE001"),
            ("0100000003", "STYLE001"),
        ],
    )
    index = build_style_index(inter, articles_path=articles)
    recalled = recall_style(["0100000001"], index, seed_items=1, top_k=3)
    ids = {item for item, _ in recalled}
    assert "0100000001" not in ids  # 已购 SKU 不召回
    assert "0100000002" in ids or "0100000003" in ids  # 同款其它 SKU


def test_style_respects_as_of(tmp_path: Path) -> None:  # as-of 防泄漏
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
        [
            ("0100000001", "STYLE001"),
            ("0100000009", "STYLE001"),
        ],
    )
    index = build_style_index(inter, articles_path=articles, as_of="2020-09-05")
    ranked = index.code_to_items["STYLE001"]
    scores = dict(ranked)
    assert scores.get("0100000009", 0.0) == 0.0  # 未来销量不计入款式内热度
    assert scores.get("0100000001", 0.0) > 0.0  # 历史 SKU 仍有热度


def test_style_cold_start_returns_empty(tmp_path: Path) -> None:  # 无历史
    inter = tmp_path / "train.inter"
    articles = tmp_path / "articles.csv"
    _write_inter(inter, [("u1", "0100000001", "2020-09-01")])
    _write_articles(articles, [("0100000001", "STYLE001")])
    index = build_style_index(inter, articles_path=articles)
    assert recall_style([], index, top_k=5) == []
