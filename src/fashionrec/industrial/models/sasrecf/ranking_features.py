"""Reuse the single Industrial SASRecF checkpoint as LambdaRank evidence."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from fashionrec.experiment.config import load_experiment_config
from fashionrec.industrial.data.basket_history import history_from_events
from fashionrec.shared.domain.candidates import Candidate
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id


SCHEMA_VERSION = "hm.sasrecf_model_reuse_features.v1"
SEQUENCE_CHANNEL = "sasrecf"


def snapshot_key(value: pd.Timestamp | str) -> str:
    return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")


def selected_train_snapshot_dates(
    snapshots: pd.DataFrame,
    *,
    limit: int,
) -> list[pd.Timestamp]:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    required = {"as_of_date", "split"}
    missing = required.difference(snapshots.columns)
    if missing:
        raise ValueError(f"snapshots missing columns: {sorted(missing)}")
    train = snapshots[snapshots["split"].astype(str).str.lower().eq("train")]
    dates = sorted({pd.Timestamp(value).normalize() for value in train["as_of_date"]})
    return dates[-limit:]


def snapshot_recall_path(root: str | Path, as_of: pd.Timestamp | str) -> Path:
    return Path(root) / snapshot_key(as_of) / "sasrecf_train.csv"


def load_snapshot_sequence_candidates(
    root: str | Path,
    as_of: pd.Timestamp | str,
) -> list[Candidate]:
    from fashionrec.industrial.recall.generator import read_candidate_csv

    path = snapshot_recall_path(root, as_of)
    if not path.is_file():
        raise FileNotFoundError(
            f"reused SASRecF train recall not found: {path}. "
            "Run the industrial `ranker-sequence` stage first."
        )
    return read_candidate_csv(path)


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "item_id", "date"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    frame = events.loc[:, ["user_id", "item_id", "date"]].copy()
    frame["user_id"] = frame["user_id"].map(canonical_user_id)
    frame["item_id"] = frame["item_id"].map(canonical_item_id)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.drop_duplicates(["user_id", "date", "item_id"], keep="first").sort_values(
        ["user_id", "date", "item_id"], kind="mergesort"
    )


def _snapshot_users(snapshots: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    mask = snapshots["split"].astype(str).str.lower().eq("train")
    mask &= pd.to_datetime(snapshots["as_of_date"]).dt.normalize().eq(as_of)
    return sorted({canonical_user_id(value) for value in snapshots.loc[mask, "user_id"]})


class SequenceScorer(Protocol):
    def recommend(
        self,
        *,
        users: Sequence[str],
        histories: dict[str, list[str]],
        top_k: int,
    ) -> list[Candidate]: ...


@dataclass(slots=True)
class ReusedSASRecFScorer:
    """One loaded checkpoint reused across all historical ranking snapshots."""

    config: object
    model: object
    dataset: object
    interaction_type: type
    torch: object
    batch_size: int = 256

    @classmethod
    def from_checkpoint(cls, model_file: str | Path, *, batch_size: int = 256) -> "ReusedSASRecFScorer":
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        path = Path(model_file)
        if not path.is_file():
            raise FileNotFoundError(f"SASRecF checkpoint not found: {path}")

        from fashionrec.pytorch_compat import patch_recbole_compat

        patch_recbole_compat()
        import torch
        from recbole.data.interaction import Interaction
        from recbole.quick_start import load_data_and_model

        config, model, dataset, _train_data, _valid_data, _test_data = load_data_and_model(
            model_file=str(path)
        )
        model.eval()
        return cls(config, model, dataset, Interaction, torch, batch_size=batch_size)

    def recommend(
        self,
        *,
        users: Sequence[str],
        histories: dict[str, list[str]],
        top_k: int,
    ) -> list[Candidate]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        iid_field = self.dataset.iid_field
        list_suffix = str(self.config["LIST_SUFFIX"])
        item_seq_field = f"{iid_field}{list_suffix}"
        length_field = str(self.config["ITEM_LIST_LENGTH_FIELD"])
        max_length = int(self.config["MAX_ITEM_LIST_LENGTH"])
        token_to_id = self.dataset.field2token_id[iid_field]
        device = self.config["device"]
        resolved_top_k = min(top_k, int(self.dataset.item_num) - 1)
        rows: list[Candidate] = []

        prepared: list[tuple[str, list[int]]] = []
        for user_id in users:
            token_ids = [
                int(token_to_id[item_id])
                for item_id in histories.get(user_id, [])[-max_length:]
                if item_id in token_to_id
            ]
            if token_ids:
                prepared.append((canonical_user_id(user_id), token_ids[-max_length:]))

        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            sequence = self.torch.zeros((len(batch), max_length), dtype=self.torch.long)
            lengths = self.torch.tensor([len(item_ids) for _, item_ids in batch], dtype=self.torch.long)
            for row_index, (_user_id, item_ids) in enumerate(batch):
                sequence[row_index, : len(item_ids)] = self.torch.tensor(item_ids, dtype=self.torch.long)
            interaction = self.interaction_type(
                {item_seq_field: sequence, length_field: lengths}
            ).to(device)
            with self.torch.no_grad():
                scores = self.model.full_sort_predict(interaction)
            scores = scores.view(-1, self.dataset.item_num)
            scores[:, 0] = -np.inf
            top_scores, top_ids = self.torch.topk(scores, resolved_top_k)
            item_tokens = self.dataset.id2token(iid_field, top_ids.cpu().numpy())
            score_values = top_scores.cpu().numpy()
            for row_index, (user_id, _item_ids) in enumerate(batch):
                for rank in range(resolved_top_k):
                    token = item_tokens[row_index][rank]
                    if isinstance(token, bytes):
                        token = token.decode("utf-8")
                    rows.append(
                        Candidate(
                            user_id,
                            canonical_item_id(token),
                            SEQUENCE_CHANNEL,
                            float(score_values[row_index][rank]),
                            rank + 1,
                            "train",
                        )
                    )
        return rows


def load_sequence_scorer(model_file: str | Path, *, batch_size: int) -> SequenceScorer:
    return ReusedSASRecFScorer.from_checkpoint(model_file, batch_size=batch_size)


def _write_candidates(candidates: Sequence[Candidate], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "item_id", "channel", "score", "rank", "split"],
        )
        writer.writeheader()
        writer.writerows(candidate.as_dict() for candidate in candidates)
    return path


def materialize_reused_sequence_features(
    *,
    experiment_config: Path,
    data_dir: Path,
    model_file: Path,
    output_dir: Path,
    batch_size: int = 256,
) -> dict[str, object]:
    """Load one selected model once and score every configured train snapshot."""
    config = load_experiment_config(experiment_config)
    snapshots = pd.read_parquet(data_dir / "snapshots")
    events = _normalize_events(pd.read_parquet(data_dir / "events"))
    dates = selected_train_snapshot_dates(snapshots, limit=config.ranking.train_snapshot_limit)
    if not dates:
        raise ValueError("no train snapshots available for SASRecF LambdaRank features")

    scorer = load_sequence_scorer(model_file, batch_size=batch_size)
    summaries: list[dict[str, object]] = []
    for as_of in dates:
        users = _snapshot_users(snapshots, as_of)
        histories = history_from_events(
            events,
            max_items=config.data.max_user_history,
            as_of=as_of,
        )
        candidates = scorer.recommend(
            users=users,
            histories=histories,
            top_k=config.candidate.sequence_top_k,
        )
        output = _write_candidates(candidates, snapshot_recall_path(output_dir, as_of))
        summaries.append(
            {
                "as_of_date": snapshot_key(as_of),
                "users": len(users),
                "users_with_sequence_recall": len({candidate.user_id for candidate in candidates}),
                "rows": len(candidates),
                "recall": str(output),
            }
        )

    report = output_dir / "model_reuse_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "single_checkpoint_simple_reuse",
        "model_file": str(Path(model_file).resolve()),
        "model_load_count": 1,
        "causal_model": False,
        "history_as_of": True,
        "warning": (
            "The checkpoint parameters, item vocabulary and checkpoint selection may use data after "
            "historical LambdaRank train snapshots; offline ranking metrics are not strict PIT estimates."
        ),
        "snapshots": summaries,
    }
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["report"] = str(report)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fashionrec.industrial ranker-sequence",
        description="Reuse the single selected SASRecF checkpoint for LambdaRank train snapshots",
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    summary = materialize_reused_sequence_features(
        experiment_config=args.experiment_config,
        data_dir=args.data_dir,
        model_file=args.model_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
