"""Materialize industrial LambdaRank tables from fixed candidates and PIT features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fashionrec.industrial.recall.union import union_candidates
from fashionrec.industrial.data.cross_features import build_cross_feature_table
from fashionrec.industrial.data.user_features import load_item_metadata
from fashionrec.shared.domain.candidates import Candidate
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id
from fashionrec.industrial.evaluation.candidate_diagnostics import diagnose_users
from fashionrec.industrial.evaluation.experiment_report import save_candidate_diagnostics
from fashionrec.shared.experiment.config import ExperimentConfig, load_experiment_config
from fashionrec.industrial.ranking.dataset import RankingDataset, build_ranking_dataset, write_ranking_dataset
from fashionrec.industrial.recall.generator import generate_candidates, read_candidate_csv
from fashionrec.industrial.recall.channel_registry import build_rule_channel_registry
from fashionrec.industrial.recall.service import ALL_CHANNELS
from fashionrec.industrial.data.basket_history import history_from_events
from fashionrec.industrial.models.sasrecf.ranking_features import load_snapshot_sequence_candidates


SEQUENCE_CHANNEL = "sasrecf"


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    split: str
    snapshot_date: pd.Timestamp
    candidates: tuple[Candidate, ...]
    history: dict[str, list[str]]


def _as_day(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _read_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    return frame


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "item_id", "date"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    frame = events.copy()
    frame["user_id"] = frame["user_id"].map(canonical_user_id)
    frame["item_id"] = frame["item_id"].map(canonical_item_id)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.sort_values(["user_id", "date", "item_id"], kind="mergesort")


def build_history_as_of(
    events: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    max_items: int,
) -> dict[str, list[str]]:
    """Build deterministic item histories from complete user-day-item events."""
    if max_items < 1:
        raise ValueError("max_items must be >= 1")
    cutoff = _as_day(as_of)
    return history_from_events(events, max_items=max_items, as_of=cutoff)


def _snapshot_dates(snapshots: pd.DataFrame, split: str) -> list[pd.Timestamp]:
    selected = snapshots[snapshots["split"].astype(str).str.lower() == split]
    return sorted({_as_day(value) for value in selected["as_of_date"]})


def _snapshot_users(snapshots: pd.DataFrame, split: str, as_of: pd.Timestamp) -> list[str]:
    mask = snapshots["split"].astype(str).str.lower().eq(split)
    mask &= pd.to_datetime(snapshots["as_of_date"]).dt.normalize().eq(as_of)
    return sorted({canonical_user_id(value) for value in snapshots.loc[mask, "user_id"]})


def _channel_top_k(config: ExperimentConfig) -> dict[str, int]:
    candidate = config.candidate
    return {
        "popular": candidate.popular_top_k,
        "category_popular": candidate.category_popular_top_k,
        "item2item": candidate.item2item_top_k,
        "repurchase": candidate.repurchase_top_k,
        "style": candidate.style_top_k,
        "content": candidate.content_top_k,
    }


def build_train_candidate_batches(
    *,
    config: ExperimentConfig,
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    train_inter: Path,
    item_file: Path,
    articles_path: Path,
    customers_path: Path,
    sequence_feature_dir: Path | None = None,
) -> list[CandidateBatch]:
    dates = _snapshot_dates(snapshots, "train")[-config.ranking.train_snapshot_limit :]
    if not dates:
        raise ValueError("no train snapshots available for LambdaRank")
    batches: list[CandidateBatch] = []
    for as_of in dates:
        users = _snapshot_users(snapshots, "train", as_of)
        if not users:
            continue
        history = build_history_as_of(events, as_of, max_items=config.data.max_user_history)
        registry = build_rule_channel_registry(
            [train_inter],
            item_file=item_file,
            customers_path=customers_path,
            articles_path=articles_path,
            as_of=as_of,
        )
        generated = generate_candidates(
            eval_users=users,
            user_history=history,
            channels=registry,
            split="train",
            top_k_by_channel=_channel_top_k(config),
        )
        if config.ranking.use_sequence_features:
            if sequence_feature_dir is None:
                raise ValueError("ranking.use_sequence_features requires sequence_feature_dir")
            sequence_candidates = load_snapshot_sequence_candidates(sequence_feature_dir, as_of)
            bad_splits = sorted({candidate.split for candidate in sequence_candidates if candidate.split != "train"})
            if bad_splits:
                raise ValueError(f"snapshot SASRecF candidates must use split=train, found={bad_splits}")
            user_set = set(users)
            generated.extend(candidate for candidate in sequence_candidates if candidate.user_id in user_set)
        frozen = union_candidates(generated, config.ranking.top_k_for_training)
        if not frozen:
            raise ValueError(f"train candidate union is empty for snapshot {as_of.date()}")
        batches.append(CandidateBatch("train", as_of, tuple(frozen), history))
    if not batches:
        raise ValueError("no train candidate batches generated")
    return batches


def load_eval_candidate_batch(
    *,
    split: str,
    candidate_path: Path,
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    max_user_history: int,
) -> CandidateBatch:
    dates = _snapshot_dates(snapshots, split)
    if len(dates) != 1:
        raise ValueError(f"expected one {split} snapshot, found {dates}")
    as_of = dates[0]
    users = set(_snapshot_users(snapshots, split, as_of))
    candidates = tuple(row for row in read_candidate_csv(candidate_path) if row.user_id in users)
    if not candidates:
        raise ValueError(f"no {split} candidates matched snapshot users: {candidate_path}")
    history = build_history_as_of(events, as_of, max_items=max_user_history)
    return CandidateBatch(split, as_of, candidates, history)


def candidate_pairs(batches: list[CandidateBatch]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for batch in batches:
        seen: set[tuple[str, str]] = set()
        for candidate in batch.candidates:
            key = (candidate.user_id, candidate.item_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "user_id": candidate.user_id,
                    "item_id": candidate.item_id,
                    "as_of_date": batch.snapshot_date,
                    "split": batch.split,
                }
            )
    if not rows:
        raise ValueError("candidate pairs are empty")
    return pd.DataFrame(rows).drop_duplicates(["user_id", "item_id", "as_of_date"], keep="first")


def _customer_cohorts(customer_features: pd.DataFrame) -> pd.DataFrame:
    age_column = "age_bucket:token" if "age_bucket:token" in customer_features.columns else "age_bucket"
    if age_column not in customer_features.columns:
        return pd.DataFrame(columns=["user_id", "age_bucket"])
    return customer_features.loc[:, ["user_id", age_column]].rename(columns={age_column: "age_bucket"})


def _labels_for_batch(labels: pd.DataFrame, batch: CandidateBatch) -> pd.DataFrame:
    mask = labels["split"].astype(str).str.lower().eq(batch.split)
    mask &= pd.to_datetime(labels["as_of_date"]).dt.normalize().eq(batch.snapshot_date)
    return labels.loc[mask].copy()


def materialize_ranking_tables(
    *,
    config: ExperimentConfig,
    data_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    articles_path: Path,
    customers_path: Path,
    sequence_feature_dir: Path | None = None,
) -> dict[str, object]:
    snapshots = _read_required_parquet(data_dir / "snapshots", "snapshots")
    labels = _read_required_parquet(data_dir / "labels", "next-basket labels")
    events = _normalize_events(_read_required_parquet(data_dir / "events", "events"))
    user_features = _read_required_parquet(data_dir / "user_features", "PIT user features")
    customer_features = _read_required_parquet(
        data_dir / "customer_features" / "customers.parquet", "customer features"
    )
    item_features = _read_required_parquet(data_dir / "item_features" / "items.parquet", "item features")

    train_batches = build_train_candidate_batches(
        config=config,
        snapshots=snapshots,
        events=events,
        train_inter=data_dir / "hm" / "hm.train.inter",
        item_file=data_dir / "hm_seq" / "hm_seq.item",
        articles_path=articles_path,
        customers_path=customers_path,
        sequence_feature_dir=sequence_feature_dir,
    )
    valid_batch = load_eval_candidate_batch(
        split="valid",
        candidate_path=candidate_dir / "valid.csv",
        snapshots=snapshots,
        events=events,
        max_user_history=config.data.max_user_history,
    )
    test_batch = load_eval_candidate_batch(
        split="test",
        candidate_path=candidate_dir / "test.csv",
        snapshots=snapshots,
        events=events,
        max_user_history=config.data.max_user_history,
    )
    batches = [*train_batches, valid_batch, test_batch]

    pairs = candidate_pairs(batches)
    item_metadata = load_item_metadata(articles_path)
    cross_features = build_cross_feature_table(
        pairs,
        events,
        item_metadata,
        user_cohorts=_customer_cohorts(customer_features),
    )

    channels = [*ALL_CHANNELS, SEQUENCE_CHANNEL] if config.ranking.use_sequence_features else list(ALL_CHANNELS)
    frames: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    uncovered: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    for batch in batches:
        dataset = build_ranking_dataset(
            batch.candidates,
            channels=channels,
            snapshot_dates=batch.snapshot_date,
            history_lengths={user: len(items) for user, items in batch.history.items()},
            labels=_labels_for_batch(labels, batch),
            user_features=user_features,
            customer_features=customer_features,
            item_features=item_features,
            cross_features=cross_features,
        )
        frames[batch.split].append(dataset.frame)
        uncovered[batch.split] += dataset.n_uncovered_labels

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"schema_version": "hm.ranking_materialization.v1", "splits": {}}
    if config.ranking.use_sequence_features:
        if sequence_feature_dir is None:
            raise ValueError("ranking.use_sequence_features requires sequence_feature_dir")
        reuse_report = sequence_feature_dir / "model_reuse_report.json"
        if not reuse_report.is_file():
            raise FileNotFoundError(f"SASRecF model reuse report not found: {reuse_report}")
        reuse_payload = json.loads(reuse_report.read_text(encoding="utf-8"))
        summary["sequence_evidence"] = {
            "mode": reuse_payload.get("mode"),
            "model_file": reuse_payload.get("model_file"),
            "causal_model": reuse_payload.get("causal_model"),
            "history_as_of": reuse_payload.get("history_as_of"),
            "warning": reuse_payload.get("warning"),
        }
    for split, split_frames in frames.items():
        frame = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
        if frame.empty:
            raise ValueError(f"ranking table is empty for split={split}")
        dataset = RankingDataset(frame=frame, n_uncovered_labels=uncovered[split])
        path = write_ranking_dataset(dataset, output_dir / f"{split}.parquet")
        summary["splits"][split] = {
            "path": str(path),
            "rows": dataset.n_rows,
            "groups": len(dataset.group_sizes),
            "positives": dataset.n_positives,
            "uncovered_labels": dataset.n_uncovered_labels,
        }

    train_candidate_rows = [
        {**candidate.as_dict(), "snapshot_date": batch.snapshot_date}
        for batch in train_batches
        for candidate in batch.candidates
    ]
    pd.DataFrame(train_candidate_rows).to_parquet(output_dir / "train_candidates.parquet", index=False)

    _write_valid_candidate_diagnostics(
        valid_batch,
        labels=_labels_for_batch(labels, valid_batch),
        activity_tiers=config.evaluation.activity_tiers,
        diagnostics_dir=diagnostics_dir,
        union_top_k=config.candidate.union_top_k,
    )
    report = output_dir / "dataset_report.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["report"] = str(report)
    return summary


def _write_valid_candidate_diagnostics(
    batch: CandidateBatch,
    *,
    labels: pd.DataFrame,
    activity_tiers: dict[str, tuple[int, int | None]],
    diagnostics_dir: Path,
    union_top_k: int,
) -> None:
    actual_by_user = {
        canonical_user_id(user): {canonical_item_id(item) for item in group["item_id"]}
        for user, group in labels.groupby("user_id", sort=True)
    }
    by_user: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for candidate in batch.candidates:
        by_user.setdefault(candidate.user_id, {}).setdefault(candidate.channel, []).append(
            (candidate.item_id, candidate.score)
        )
    users = []
    for user_id, actual in actual_by_user.items():
        history = batch.history.get(user_id, [])
        users.append(
            {
                "user_id": user_id,
                "actual": actual,
                "history": history,
                "history_set": set(history),
                "channel_candidates": by_user.get(user_id, {}),
            }
        )
    diagnostics = diagnose_users(
        users,
        channels=[*ALL_CHANNELS, SEQUENCE_CHANNEL] if SEQUENCE_CHANNEL in {c.channel for c in batch.candidates} else list(ALL_CHANNELS),
        activity_tiers=activity_tiers,
        union_k_for_counts=union_top_k,
    )
    save_candidate_diagnostics(diagnostics_dir, diagnostics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fashionrec ranker-dataset",
        description="Build train/valid/test LambdaRank tables with single-checkpoint SASRecF reuse.",
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--articles-path", type=Path, default=Path("data/raw/articles.csv"))
    parser.add_argument("--customers-path", type=Path, default=Path("data/raw/customers.csv"))
    parser.add_argument("--sequence-feature-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_experiment_config(args.experiment_config)
    if config.ranking.library != "lightgbm" or config.ranking.objective != "lambdarank":
        raise ValueError("formal ranker pipeline currently supports only lightgbm + lambdarank")
    summary = materialize_ranking_tables(
        config=config,
        data_dir=args.data_dir,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        diagnostics_dir=args.diagnostics_dir,
        articles_path=args.articles_path,
        customers_path=args.customers_path,
        sequence_feature_dir=args.sequence_feature_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
