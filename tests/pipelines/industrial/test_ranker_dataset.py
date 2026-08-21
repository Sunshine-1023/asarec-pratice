"""Tests for causal LambdaRank table materialization orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fashionrec.domain.candidates import Candidate
from fashionrec.experiment.config import load_experiment_config
from fashionrec.industrial.ranking import dataset_materialization as command
from fashionrec.recall.generator import write_candidate_csv


def _events() -> pd.DataFrame:
    return command._normalize_events(
        pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u2"],
                "item_id": ["1", "2", "3", "4"],
                "date": ["2020-09-01", "2020-09-08", "2020-09-20", "2020-09-07"],
            }
        )
    )


def test_history_is_causal_and_truncated() -> None:
    history = command.build_history_as_of(_events(), "2020-09-10", max_items=1)
    assert history["u1"] == ["0000000002"]
    assert history["u2"] == ["0000000004"]
    assert "0000000003" not in history["u1"]


def test_history_deduplicates_same_day_sku_without_row_order() -> None:
    events = command._normalize_events(
        pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["1", "1", "2"],
                "date": ["2020-09-01", "2020-09-01", "2020-09-01"],
            }
        )
    )
    assert command.build_history_as_of(events, "2020-09-01", max_items=10)["u1"] == [
        "0000000001",
        "0000000002",
    ]


def test_train_batches_use_only_latest_configured_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = pd.DataFrame(
        {
            "user_id": ["u1"] * 6,
            "as_of_date": pd.date_range("2020-08-01", periods=6, freq="7D"),
            "split": ["train"] * 6,
        }
    )
    seen_as_of: list[pd.Timestamp] = []

    def fake_registry(*_args: object, as_of: object, **_kwargs: object) -> dict[str, object]:
        seen_as_of.append(pd.Timestamp(as_of))
        return {"popular": object()}

    def fake_generate(**kwargs: object) -> list[Candidate]:
        return [Candidate("u1", "1", "popular", 1.0, 1, str(kwargs["split"]))]

    monkeypatch.setattr(command, "build_rule_channel_registry", fake_registry)
    monkeypatch.setattr(command, "generate_candidates", fake_generate)
    monkeypatch.setattr(
        command,
        "load_snapshot_sequence_candidates",
        lambda _root, as_of: [Candidate("u1", "2", "sasrecf", float(pd.Timestamp(as_of).day), 1, "train")],
    )
    batches = command.build_train_candidate_batches(
        config=load_experiment_config("configs/industrial/experiment.yaml"),
        snapshots=snapshots,
        events=_events(),
        train_inter=Path("unused.inter"),
        item_file=Path("unused.item"),
        articles_path=Path("unused-articles.csv"),
        customers_path=Path("unused-customers.csv"),
        sequence_feature_dir=Path("snapshot-sequence"),
    )
    expected = [pd.Timestamp(value) for value in snapshots["as_of_date"].iloc[-4:]]
    assert seen_as_of == expected
    assert [batch.snapshot_date for batch in batches] == expected


def test_eval_candidates_are_restricted_to_unique_snapshot_users(tmp_path: Path) -> None:
    candidate_path = write_candidate_csv(
        [
            Candidate("u1", "1", "popular", 1.0, 1, "valid"),
            Candidate("u2", "2", "popular", 1.0, 1, "valid"),
        ],
        tmp_path / "valid.csv",
    )
    snapshots = pd.DataFrame(
        {"user_id": ["u1"], "as_of_date": ["2020-09-10"], "split": ["valid"]}
    )
    batch = command.load_eval_candidate_batch(
        split="valid",
        candidate_path=candidate_path,
        snapshots=snapshots,
        events=_events(),
        max_user_history=10,
    )
    assert {candidate.user_id for candidate in batch.candidates} == {"u1"}
    assert batch.snapshot_date == pd.Timestamp("2020-09-10")
    assert batch.history["u1"] == ["0000000001", "0000000002"]


def test_candidate_pairs_cover_every_candidate_and_snapshot() -> None:
    batches = [
        command.CandidateBatch(
            "train",
            pd.Timestamp("2020-09-01"),
            (
                Candidate("u1", "1", "popular", 1.0, 1, "train"),
                Candidate("u1", "2", "item2item", 0.5, 1, "train"),
            ),
            {"u1": []},
        ),
        command.CandidateBatch(
            "valid",
            pd.Timestamp("2020-09-08"),
            (Candidate("u1", "1", "sasrecf", 0.9, 1, "valid"),),
            {"u1": ["0000000002"]},
        ),
    ]
    pairs = command.candidate_pairs(batches)
    assert set(zip(pairs["item_id"], pairs["as_of_date"])) == {
        ("0000000001", pd.Timestamp("2020-09-01")),
        ("0000000002", pd.Timestamp("2020-09-01")),
        ("0000000001", pd.Timestamp("2020-09-08")),
    }


def test_materialization_fails_fast_when_protocol_artifacts_are_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="snapshots not found"):
        command.materialize_ranking_tables(
            config=load_experiment_config("configs/industrial/experiment.yaml"),
            data_dir=tmp_path / "data",
            candidate_dir=tmp_path / "candidates",
            output_dir=tmp_path / "ranking",
            diagnostics_dir=tmp_path / "evaluation",
            articles_path=tmp_path / "articles.csv",
            customers_path=tmp_path / "customers.csv",
        )


def test_materialize_ranking_tables_writes_all_splits_with_cross_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    candidate_dir = tmp_path / "candidates"
    output_dir = tmp_path / "ranking"
    diagnostics_dir = tmp_path / "evaluation"
    (data_dir / "customer_features").mkdir(parents=True)
    (data_dir / "item_features").mkdir(parents=True)
    candidate_dir.mkdir(parents=True)

    snapshots = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "as_of_date": ["2020-09-01", "2020-09-08", "2020-09-15"],
            "split": ["train", "valid", "test"],
        }
    )
    labels = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["2", "3", "4"],
            "as_of_date": ["2020-09-01", "2020-09-08", "2020-09-15"],
            "split": ["train", "valid", "test"],
            "label_purchase": [1, 1, 1],
        }
    )
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["1", "2", "3"],
            "date": ["2020-08-25", "2020-09-05", "2020-09-12"],
            "quantity": [1, 1, 1],
            "mean_price": [0.1, 0.2, 0.3],
            "sales_channel_mode": [1, 1, 2],
        }
    )
    user_features = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "as_of_date": ["2020-09-01", "2020-09-08", "2020-09-15"],
            "purchase_count_7d": [1.0, 2.0, 3.0],
        }
    )
    customer_features = pd.DataFrame({"user_id": ["u1"], "age_bucket:token": ["25-34"]})
    item_features = pd.DataFrame(
        {"item_id": ["1", "2", "3", "4", "5", "6"], "product_code:float": [1, 2, 3, 4, 5, 6]}
    )
    snapshots.to_parquet(data_dir / "snapshots", index=False)
    labels.to_parquet(data_dir / "labels", index=False)
    events.to_parquet(data_dir / "events", index=False)
    user_features.to_parquet(data_dir / "user_features", index=False)
    customer_features.to_parquet(data_dir / "customer_features" / "customers.parquet", index=False)
    item_features.to_parquet(data_dir / "item_features" / "items.parquet", index=False)

    articles = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "article_id": ["1", "2", "3", "4", "5", "6"],
            "product_code": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "colour_group_name": ["black"] * 6,
            "department_name": ["womenswear"] * 6,
            "product_type_name": ["shirt"] * 6,
        }
    ).to_csv(articles, index=False)
    customers = tmp_path / "customers.csv"
    pd.DataFrame({"customer_id": ["u1"], "age": [30]}).to_csv(customers, index=False)

    write_candidate_csv(
        [
            Candidate("u1", "3", "popular", 1.0, 1, "valid"),
            Candidate("u1", "5", "sasrecf", 0.5, 1, "valid"),
        ],
        candidate_dir / "valid.csv",
    )
    write_candidate_csv(
        [
            Candidate("u1", "4", "popular", 1.0, 1, "test"),
            Candidate("u1", "5", "sasrecf", 0.5, 1, "test"),
        ],
        candidate_dir / "test.csv",
    )
    train_batch = command.CandidateBatch(
        "train",
        pd.Timestamp("2020-09-01"),
        (
            Candidate("u1", "2", "popular", 1.0, 1, "train"),
            Candidate("u1", "5", "item2item", 0.5, 1, "train"),
            Candidate("u1", "6", "sasrecf", 0.8, 1, "train"),
        ),
        {"u1": ["0000000001"]},
    )
    monkeypatch.setattr(command, "build_train_candidate_batches", lambda **_kwargs: [train_batch])
    sequence_feature_dir = output_dir / "sasrecf_model_reuse"
    sequence_feature_dir.mkdir(parents=True)
    (sequence_feature_dir / "model_reuse_report.json").write_text(
        json.dumps(
            {
                "mode": "single_checkpoint_simple_reuse",
                "model_file": "/tmp/sasrecf_selected.pth",
                "causal_model": False,
                "history_as_of": True,
                "warning": "non-strict PIT test fixture",
            }
        ),
        encoding="utf-8",
    )

    summary = command.materialize_ranking_tables(
        config=load_experiment_config("configs/industrial/experiment.yaml"),
        data_dir=data_dir,
        candidate_dir=candidate_dir,
        output_dir=output_dir,
        diagnostics_dir=diagnostics_dir,
        articles_path=articles,
        customers_path=customers,
        sequence_feature_dir=sequence_feature_dir,
    )
    assert set(summary["splits"]) == {"train", "valid", "test"}
    for split in ("train", "valid", "test"):
        frame = pd.read_parquet(output_dir / f"{split}.parquet")
        assert len(frame) == (3 if split == "train" else 2)
        assert int(frame["label"].sum()) == 1
        assert "cross__user_item_purchase_count" in frame.columns
        assert "sasrecf_present" in frame.columns
        assert "sasrecf_score" in frame.columns
        assert "sasrecf_rank" in frame.columns
        assert int(frame["sasrecf_present"].sum()) == 1
    assert (output_dir / "train_candidates.parquet").is_file()
    assert (output_dir / "dataset_report.json").is_file()
    assert summary["sequence_evidence"]["causal_model"] is False
