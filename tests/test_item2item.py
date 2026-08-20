"""Tests for item2item similarity variants (Task 3.3)."""  # Item2Item 升级

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格
import pytest  # 断言

from fashionrec.recall.item2item import (  # Item2Item
    DEFAULT_SIMILARITY_MODE,
    SIMILARITY_MODES,
    build_item2item_index,
    recall_item2item,
)


def _unix(date: str) -> int:
    return int(pd.Timestamp(f"{date} 12:00:00", tz="UTC").timestamp())


def _write_inter(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]
    lines.extend(f"{user}\t{item}\t{_unix(date)}" for user, item, date in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hub_spoke_rows() -> list[tuple[str, str, str]]:  # 热门 A + 长尾 B/C/D
    return [
        ("u1", "0000000001", "2020-09-01"),
        ("u1", "0000000002", "2020-09-01"),
        ("u2", "0000000001", "2020-09-02"),
        ("u2", "0000000003", "2020-09-02"),
        ("u3", "0000000001", "2020-09-03"),
        ("u3", "0000000004", "2020-09-03"),
        ("u4", "0000000002", "2020-09-04"),
        ("u4", "0000000003", "2020-09-04"),
    ]


def test_default_mode_is_cosine_iuf() -> None:  # 默认变体
    assert DEFAULT_SIMILARITY_MODE == "cosine_iuf"
    assert len(SIMILARITY_MODES) == 5


def test_cosine_iuf_downweights_hub_item(tmp_path: Path) -> None:  # 降热门偏差
    inter = tmp_path / "train.inter"
    _write_inter(inter, _hub_spoke_rows())
    raw = build_item2item_index(inter, cooccur_weeks=52, top_sim_k=5, similarity_mode="raw_cooccur")
    cosine = build_item2item_index(inter, cooccur_weeks=52, top_sim_k=5, similarity_mode="cosine_iuf")
    raw_neighbors = raw["0000000001"]  # A 的邻居（原始计数）
    assert raw_neighbors["0000000002"] >= 1.0  # A-B 共现
    assert raw_neighbors["0000000003"] >= 1.0  # A-C 共现
    # 长尾 B-C 在余弦/IUF 下相对 A 的任意邻居更突出
    bc = cosine.get("0000000002", {}).get("0000000003", 0.0)
    ac = cosine.get("0000000001", {}).get("0000000003", 0.0)
    assert bc >= ac  # B-C 不应被 hub A 完全压制


def test_sequential_only_keeps_directed_next_item(tmp_path: Path) -> None:  # 有向序列
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000001", "2020-09-01"),
            ("u1", "0000000002", "2020-09-02"),
            ("u1", "0000000003", "2020-09-03"),
        ],
    )
    index = build_item2item_index(inter, cooccur_weeks=52, similarity_mode="sequential", top_sim_k=5)
    assert "0000000002" in index["0000000001"]  # 1->2
    assert "0000000003" in index["0000000002"]  # 2->3
    assert "0000000003" not in index.get("0000000001", {})  # 非相邻 1->3 不应出现


def test_time_decay_prefers_recent_pair(tmp_path: Path) -> None:  # 时间衰减
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000001", "2020-09-01"),
            ("u1", "0000000002", "2020-09-01"),
            ("u2", "0000000001", "2020-09-08"),
            ("u2", "0000000003", "2020-09-08"),
        ],
    )
    index = build_item2item_index(inter, cooccur_weeks=52, similarity_mode="time_decay", top_sim_k=5)
    assert index["0000000001"]["0000000003"] > index["0000000001"]["0000000002"]  # 更近的 A-C


def test_swing_mode_builds_neighbors(tmp_path: Path) -> None:  # Swing 可构建
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000002", "2020-09-01"),
            ("u1", "0000000003", "2020-09-01"),
            ("u2", "0000000002", "2020-09-02"),
            ("u2", "0000000003", "2020-09-02"),
            ("u3", "0000000002", "2020-09-03"),
            ("u3", "0000000004", "2020-09-03"),
        ],
    )
    index = build_item2item_index(inter, cooccur_weeks=52, similarity_mode="swing", top_sim_k=5)
    assert "0000000003" in index.get("0000000002", {})  # 两用户共购 B-C


def test_as_of_excludes_future_cooccurrence(tmp_path: Path) -> None:  # as-of 防泄漏
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000001", "2020-09-01"),
            ("u1", "0000000002", "2020-09-01"),
            ("u1", "0000000001", "2020-09-10"),
            ("u1", "0000000009", "2020-09-10"),
        ],
    )
    index = build_item2item_index(inter, cooccur_weeks=52, similarity_mode="raw_cooccur", as_of="2020-09-05")
    neighbors = index.get("0000000001", {})
    assert "0000000002" in neighbors  # 历史共现
    assert "0000000009" not in neighbors  # 标签周不得进入


def test_recall_item2item_aggregates_seed_neighbors(tmp_path: Path) -> None:  # 召回聚合
    inter = tmp_path / "train.inter"
    _write_inter(inter, _hub_spoke_rows())
    index = build_item2item_index(inter, cooccur_weeks=52, similarity_mode="raw_cooccur", top_sim_k=5)
    recalled = recall_item2item(["0000000002"], index, seed_items=1, top_k=3)
    assert recalled  # 非空
    assert recalled[0][0] == "0000000001"  # B 与 A 共现两次，强于 B-C


def test_invalid_similarity_mode_raises(tmp_path: Path) -> None:  # 非法变体
    inter = tmp_path / "train.inter"
    _write_inter(inter, [("u1", "0000000001", "2020-09-01")])
    with pytest.raises(ValueError, match="similarity_mode"):
        build_item2item_index(inter, similarity_mode="invalid")  # type: ignore[arg-type]
