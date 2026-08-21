"""Tests for reusing one Industrial SASRecF checkpoint in LambdaRank."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fashionrec.industrial.models.sasrecf.ranking_features import (
    materialize_reused_sequence_features,
    selected_train_snapshot_dates,
    snapshot_recall_path,
)
from fashionrec.shared.domain.candidates import Candidate


def test_selected_train_snapshot_dates_respects_limit() -> None:
    snapshots = pd.DataFrame(
        {
            "as_of_date": pd.date_range("2020-08-01", periods=6, freq="7D"),
            "split": ["train"] * 5 + ["valid"],
        }
    )
    assert selected_train_snapshot_dates(snapshots, limit=2) == [
        pd.Timestamp("2020-08-22"),
        pd.Timestamp("2020-08-29"),
    ]


def test_snapshot_recall_path_is_namespaced_by_as_of(tmp_path: Path) -> None:
    assert snapshot_recall_path(tmp_path, "2020-08-15") == (
        tmp_path / "2020-08-15" / "sasrecf_train.csv"
    )


def test_materialization_loads_one_model_and_scores_each_snapshot_as_of(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fashionrec.industrial.models.sasrecf import ranking_features

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "as_of_date": ["2020-08-08", "2020-08-15"],
            "split": ["train", "train"],
        }
    ).to_parquet(data_dir / "snapshots", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["1", "2", "999"],
            "date": ["2020-08-01", "2020-08-12", "2020-08-20"],
        }
    ).to_parquet(data_dir / "events", index=False)
    checkpoint = tmp_path / "sasrecf_selected.pth"
    checkpoint.write_bytes(b"stub")
    experiment = tmp_path / "experiment.yaml"
    payload = Path("configs/industrial/experiment.yaml").read_text(encoding="utf-8")
    payload = payload.replace("train_snapshot_limit: 4", "train_snapshot_limit: 2")
    experiment.write_text(payload, encoding="utf-8")

    load_calls: list[tuple[Path, int]] = []
    seen_histories: list[list[str]] = []

    class FakeScorer:
        def recommend(self, *, users, histories, top_k):
            assert users == ["u1"]
            assert top_k == 200
            seen_histories.append(list(histories["u1"]))
            return [Candidate("u1", "8", "sasrecf", 0.8, 1, "train")]

    def fake_load(model_file, *, batch_size):
        load_calls.append((Path(model_file), batch_size))
        return FakeScorer()

    monkeypatch.setattr(ranking_features, "load_sequence_scorer", fake_load)
    output_dir = tmp_path / "ranking" / "sasrecf_model_reuse"
    summary = materialize_reused_sequence_features(
        experiment_config=experiment,
        data_dir=data_dir,
        model_file=checkpoint,
        output_dir=output_dir,
        batch_size=64,
    )

    assert load_calls == [(checkpoint, 64)]
    assert seen_histories == [
        ["0000000001"],
        ["0000000001", "0000000002"],
    ]
    assert all("0000000999" not in history for history in seen_histories)
    for as_of in ("2020-08-08", "2020-08-15"):
        recall = pd.read_csv(snapshot_recall_path(output_dir, as_of))
        assert recall.loc[0, "split"] == "train"
        assert recall.loc[0, "channel"] == "sasrecf"
    report = json.loads((output_dir / "model_reuse_report.json").read_text(encoding="utf-8"))
    assert report["model_load_count"] == 1
    assert report["causal_model"] is False
    assert report["history_as_of"] is True
    assert report["model_file"] == str(checkpoint.resolve())
    assert summary["mode"] == "single_checkpoint_simple_reuse"
