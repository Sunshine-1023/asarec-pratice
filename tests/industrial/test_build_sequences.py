"""Tests for the data-layer causal sequence builder."""  # 序列构建不依赖训练入口

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fashionrec.industrial.data.build_sequences import prepare_recbole_benchmark_files, read_max_item_list_length


DAY = 86_400  # 一天的秒数；真实 hm.inter 的 timestamp 也是按自然日对齐的


def _write_inter(path: Path, rows: list[tuple[str, str, int]]) -> None:
    frame = pd.DataFrame(rows, columns=["user_id:token", "item_id:token", "timestamp:float"])
    frame.to_csv(path, sep="\t", index=False)


def test_sequence_builder_preserves_split_causality_and_canonical_ids(tmp_path: Path) -> None:
    train = tmp_path / "train.inter"
    valid = tmp_path / "valid.inter"
    test = tmp_path / "test.inter"
    _write_inter(train, [("u1", "1", DAY), ("u1", "2", 2 * DAY)])
    _write_inter(valid, [("u1", "3", 3 * DAY), ("u1", "4", 4 * DAY)])
    _write_inter(test, [("u1", "5", 5 * DAY)])

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


def test_sequence_builder_fits_on_model_train_but_uses_complete_train_history(tmp_path: Path) -> None:
    model_train = tmp_path / "model_train.inter"
    full_train = tmp_path / "train.inter"
    valid = tmp_path / "valid.inter"
    test = tmp_path / "test.inter"
    _write_inter(model_train, [("eligible", "1", DAY), ("eligible", "2", 2 * DAY)])
    _write_inter(
        full_train,
        [("eligible", "1", DAY), ("eligible", "2", 2 * DAY), ("low_activity", "7", 2 * DAY)],
    )
    _write_inter(valid, [("eligible", "3", 3 * DAY), ("low_activity", "8", 3 * DAY)])
    _write_inter(test, [("low_activity", "9", 4 * DAY)])

    outputs = prepare_recbole_benchmark_files(
        3,
        model_train,
        valid,
        test,
        tmp_path / "seq",
        train_history_file=full_train,
    )
    train_out, valid_out, test_out = (pd.read_csv(path, sep="\t", dtype="string") for path in outputs)

    assert set(train_out["user_id:token"]) == {"eligible"}
    low_valid = valid_out[valid_out["user_id:token"] == "low_activity"].iloc[0]
    assert low_valid["item_id_list:token_seq"] == "0000000007"
    low_test = test_out[test_out["user_id:token"] == "low_activity"].iloc[0]
    assert low_test["item_id_list:token_seq"] == "0000000007 0000000008"


def test_same_day_targets_share_identical_history_and_next_day_sees_the_basket(tmp_path: Path) -> None:
    train = tmp_path / "train.inter"
    valid = tmp_path / "valid.inter"
    test = tmp_path / "test.inter"
    _write_inter(
        train,
        [
            ("u1", "9", DAY),
            ("u1", "3", 2 * DAY),
            ("u1", "1", 2 * DAY),
            ("u1", "2", 2 * DAY),
            ("u1", "4", 3 * DAY),
        ],
    )
    _write_inter(valid, [("u1", "5", 4 * DAY)])
    _write_inter(test, [("u1", "6", 5 * DAY)])

    outputs = prepare_recbole_benchmark_files(10, train, valid, test, tmp_path / "seq")
    train_out, valid_out, test_out = (pd.read_csv(path, sep="\t", dtype="string") for path in outputs)

    day2 = train_out[train_out["item_id:token"].isin(["0000000001", "0000000002", "0000000003"])]
    assert sorted(day2["item_id:token"].tolist()) == ["0000000001", "0000000002", "0000000003"]
    assert set(day2["item_id_list:token_seq"]) == {"0000000009"}  # 同日三个目标历史完全相同，不含彼此
    day3 = train_out[train_out["item_id:token"] == "0000000004"].iloc[0]
    assert day3["item_id_list:token_seq"] == "0000000009 0000000001 0000000002 0000000003"  # 下一日才看到 A/B/C
    assert valid_out["item_id_list:token_seq"].tolist() == [
        "0000000009 0000000001 0000000002 0000000003 0000000004"
    ]
    assert test_out["item_id_list:token_seq"].tolist() == [
        "0000000009 0000000001 0000000002 0000000003 0000000004 0000000005"
    ]


def test_valid_same_day_targets_do_not_enter_each_others_history(tmp_path: Path) -> None:
    train = tmp_path / "train.inter"
    valid = tmp_path / "valid.inter"
    test = tmp_path / "test.inter"
    _write_inter(train, [("u1", "1", DAY)])
    _write_inter(valid, [("u1", "3", 2 * DAY), ("u1", "2", 2 * DAY), ("u1", "4", 2 * DAY)])
    _write_inter(test, [("u1", "5", 3 * DAY)])

    outputs = prepare_recbole_benchmark_files(10, train, valid, test, tmp_path / "seq")
    _train_out, valid_out, test_out = (pd.read_csv(path, sep="\t", dtype="string") for path in outputs)
    assert valid_out["item_id_list:token_seq"].tolist() == ["0000000001"] * 3  # valid 周内不按商品推进
    assert test_out["item_id_list:token_seq"].tolist() == ["0000000001 0000000002 0000000003 0000000004"]


def test_history_truncation_drops_oldest_shopping_day(tmp_path: Path) -> None:
    train = tmp_path / "train.inter"
    valid = tmp_path / "valid.inter"
    test = tmp_path / "test.inter"
    _write_inter(
        train,
        [
            ("u1", "1", DAY),
            ("u1", "2", DAY),
            ("u1", "3", DAY),
            ("u1", "4", 2 * DAY),
            ("u1", "5", 3 * DAY),
        ],
    )
    _write_inter(valid, [("u1", "6", 4 * DAY)])
    _write_inter(test, [("u1", "7", 5 * DAY)])

    outputs = prepare_recbole_benchmark_files(3, train, valid, test, tmp_path / "seq")
    train_out, _valid_out, _test_out = (pd.read_csv(path, sep="\t", dtype="string") for path in outputs)
    day3 = train_out[train_out["item_id:token"] == "0000000005"].iloc[0]
    assert day3["item_id_list:token_seq"] == "0000000004"  # 丢掉第一天的 1/2/3，而不是切成 2 3 4
