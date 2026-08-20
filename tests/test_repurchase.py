"""Tests for repurchase recall (Task 3.4)."""  # 复购召回

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.recall.repurchase import build_repurchase_index, recall_repurchase


def _unix(date: str) -> int:
    return int(pd.Timestamp(f"{date} 12:00:00", tz="UTC").timestamp())


def _write_inter(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]
    lines.extend(f"{user}\t{item}\t{_unix(date)}" for user, item, date in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_repurchase_prefers_recent_and_frequent_sku(tmp_path: Path) -> None:  # 次数×近因
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000001", "2020-09-01"),
            ("u1", "0000000001", "2020-09-02"),
            ("u1", "0000000002", "2020-08-01"),
        ],
    )
    index = build_repurchase_index(inter)
    recalled = recall_repurchase("u1", ["0000000009"], index, top_k=3, as_of="2020-09-05")
    assert recalled[0][0] == "0000000001"  # 次数多且更近
    assert recalled[1][0] == "0000000002"  # 更早购买


def test_repurchase_as_of_excludes_future_purchases(tmp_path: Path) -> None:  # as-of 防泄漏
    inter = tmp_path / "train.inter"
    _write_inter(
        inter,
        [
            ("u1", "0000000001", "2020-09-01"),
            ("u1", "0000000009", "2020-09-10"),
        ],
    )
    index = build_repurchase_index(inter, as_of="2020-09-05")
    recalled = recall_repurchase("u1", [], index, top_k=5, as_of="2020-09-05")
    ids = {item for item, _ in recalled}
    assert "0000000001" in ids
    assert "0000000009" not in ids


def test_repurchase_cold_start_returns_empty(tmp_path: Path) -> None:  # 冷启动
    inter = tmp_path / "train.inter"
    _write_inter(inter, [("u1", "0000000001", "2020-09-01")])
    index = build_repurchase_index(inter)
    assert recall_repurchase("unknown", [], index, top_k=5) == []
