"""Tests for the data-layer causal sequence builder."""  # 序列构建不依赖训练入口

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.build_sequences import prepare_recbole_benchmark_files, read_max_item_list_length


def _write_inter(path: Path, rows: list[tuple[str, str, int]]) -> None:
    frame = pd.DataFrame(rows, columns=["user_id:token", "item_id:token", "timestamp:float"])
    frame.to_csv(path, sep="\t", index=False)


def test_sequence_builder_preserves_split_causality_and_canonical_ids(tmp_path: Path) -> None:
    train = tmp_path / "train.inter"
    valid = tmp_path / "valid.inter"
    test = tmp_path / "test.inter"
    _write_inter(train, [("u1", "1", 1), ("u1", "2", 2)])
    _write_inter(valid, [("u1", "3", 3), ("u1", "4", 4)])
    _write_inter(test, [("u1", "5", 5)])

    outputs = prepare_recbole_benchmark_files(3, train, valid, test, tmp_path / "seq")
    train_out, valid_out, test_out = (pd.read_csv(path, sep="\t", dtype="string") for path in outputs)

    assert train_out["item_id_list:token_seq"].tolist() == ["0000000001"]
    assert train_out["item_id:token"].tolist() == ["0000000002"]
    assert valid_out["item_id_list:token_seq"].tolist() == ["0000000001 0000000002"] * 2  # valid 内不互相泄漏
    assert test_out["item_id_list:token_seq"].tolist() == ["0000000002 0000000003 0000000004"]  # test 可用完整 valid 历史


def test_read_max_item_list_length_ignores_inline_comment(tmp_path: Path) -> None:
    config = tmp_path / "model.yaml"
    config.write_text("MAX_ITEM_LIST_LENGTH: 100  # history cap\n", encoding="utf-8")
    assert read_max_item_list_length(config) == 100
