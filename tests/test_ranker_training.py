"""Tests for LightGBM LambdaRank training (Task 4.2)."""  # 训练

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 产物

import pandas as pd  # 表
import pytest  # skip / 断言

try:
    import lightgbm  # noqa: F401
except (ImportError, OSError):
    pytest.skip("lightgbm/libomp not available", allow_module_level=True)

from fashionrec.ranking.train import (  # 训练 API
    RANKER_SCHEMA_VERSION,
    save_ranker_artifact,
    select_feature_columns,
    train_lambdarank,
)


def _group_table(*, split: str, users: list[str], signal_item: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for user in users:
        for item in ("0000000001", "0000000002", "0000000003"):
            signal = 1.0 if item == signal_item[user] else 0.0
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "snapshot_date": pd.Timestamp("2020-09-08" if split == "train" else "2020-09-15"),
                    "group_id": f"{user}@{'2020-09-08' if split == 'train' else '2020-09-15'}",
                    "split": split,
                    "history_len": 3,
                    "channel_count": 1,
                    "popular_present": 1,
                    "popular_score": signal,
                    "popular_rank": 1 if signal else 2,
                    "signal": signal,
                    "label": int(signal),
                    "relevance": int(signal),
                }
            )
    return pd.DataFrame(rows)


def test_select_feature_columns_drops_ids_and_labels() -> None:
    frame = _group_table(split="train", users=["u1"], signal_item={"u1": "0000000001"})
    columns = select_feature_columns(frame)
    assert "user_id" not in columns
    assert "relevance" not in columns
    assert "label" not in columns
    assert "signal" in columns


def test_train_lambdarank_uses_train_split_and_writes_artifact(tmp_path: Path) -> None:
    train = _group_table(split="train", users=["u1", "u2"], signal_item={"u1": "0000000001", "u2": "0000000002"})
    valid = _group_table(split="valid", users=["u1", "u2"], signal_item={"u1": "0000000001", "u2": "0000000002"})
    mixed = pd.concat([train, valid], ignore_index=True)
    model, schema, metrics = train_lambdarank(
        mixed,
        valid,
        n_estimators=40,
        early_stopping_rounds=10,
        learning_rate=0.2,
        seed=7,
    )
    assert schema.schema_version == RANKER_SCHEMA_VERSION
    assert schema.objective == "lambdarank"
    assert "signal" in schema.feature_columns
    assert metrics["n_train_rows"] == len(train)
    assert metrics["n_train_groups"] == 2
    assert metrics["best_iteration"] >= 1
    artifact = save_ranker_artifact(model, schema, metrics, tmp_path / "lambdarank")
    assert artifact.model_path.is_file()
    assert artifact.schema_path.is_file()
    payload = artifact.schema.to_json()
    assert payload["feature_columns"] == list(schema.feature_columns)
    assert set(payload["defaults"]) == set(schema.feature_columns)
