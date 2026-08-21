"""Tests for LambdaRank inference (Task 4.2)."""  # 推理

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 产物

import pandas as pd  # 表
import pytest  # skip

try:
    import lightgbm  # noqa: F401
except (ImportError, OSError):
    pytest.skip("lightgbm/libomp not available", allow_module_level=True)

from fashionrec.industrial.ranking.predict import load_ranker, rank_feature_frame
from fashionrec.industrial.ranking.train import save_ranker_artifact, train_lambdarank


def _group_table(*, split: str, users: list[str], signal_item: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    as_of = "2020-09-08" if split == "train" else "2020-09-15"
    for user in users:
        for item in ("0000000001", "0000000002", "0000000003"):
            signal = 1.0 if item == signal_item[user] else 0.0
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "snapshot_date": pd.Timestamp(as_of),
                    "group_id": f"{user}@{as_of}",
                    "split": split,
                    "signal": signal,
                    "label": int(signal),
                    "relevance": int(signal),
                }
            )
    return pd.DataFrame(rows)


def test_predict_prefers_signal_item_and_fills_missing_features(tmp_path: Path) -> None:
    train = _group_table(split="train", users=["u1", "u2"], signal_item={"u1": "0000000001", "u2": "0000000002"})
    valid = _group_table(split="valid", users=["u1", "u2"], signal_item={"u1": "0000000001", "u2": "0000000002"})
    model, schema, metrics = train_lambdarank(train, valid, n_estimators=40, learning_rate=0.2, seed=7)
    save_ranker_artifact(model, schema, metrics, tmp_path / "lambdarank")
    ranker = load_ranker(tmp_path / "lambdarank")
    scored = rank_feature_frame(valid, ranker, top_k=12)
    top = scored[scored["rank"] == 1][["user_id", "item_id"]]
    assert set(zip(top["user_id"], top["item_id"])) == {("u1", "0000000001"), ("u2", "0000000002")}

    missing_signal = valid.drop(columns=["signal"])
    scored_missing = rank_feature_frame(missing_signal, ranker, top_k=3)
    missing_rates = scored_missing.attrs["missing_rates"]
    assert missing_rates["signal"] == 1.0
    assert len(scored_missing) == 6


def test_ranker_inference_depends_only_on_saved_artifact(tmp_path: Path) -> None:
    train = _group_table(split="train", users=["u1", "u2"], signal_item={"u1": "0000000003", "u2": "0000000001"})
    model, schema, metrics = train_lambdarank(train, n_estimators=30, learning_rate=0.2, seed=3)
    save_ranker_artifact(model, schema, metrics, tmp_path / "lambdarank")
    infer = _group_table(split="test", users=["u1"], signal_item={"u1": "0000000003"})
    ranker = load_ranker(tmp_path / "lambdarank")
    ranked = ranker.bind_features(infer).rank(
        user_id="u1",
        user_history=set(),
        channel_candidates={},
        top_k=1,
    )
    assert ranked[0].item_id == "0000000003"
    assert ranked[0].rank == 1
