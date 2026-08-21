"""Offline evaluation for the fixed four-channel Baseline RRF protocol."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fashionrec.baseline.data.paths import ProcessedDataPaths
from fashionrec.baseline.data.split import (
    TEST_INTER_FILE,
    TRAIN_INTER_FILE,
    VALID_INTER_FILE,
    assert_history_paths_allowed,
    history_paths_for_eval,
)
from fashionrec.baseline.ranking.fusion import (
    ACTIVITY_WEIGHTS,
    ActivityTier,
    build_user_history,
    classify_activity_tier,
    get_channel_weights_for_user,
    infer_sequence_channel,
    load_channel_recall_csv,
)
from fashionrec.baseline.ranking.weighted_rrf import WeightedRRFRanker
from fashionrec.baseline.recall.category_popular import (
    CATEGORY_POPULAR_RECALL_TOP_K,
    SEED_ITEMS as CATEGORY_SEED_ITEMS,
)
from fashionrec.baseline.recall.channel_registry import (
    PrecomputedChannel,
    build_rule_channel_registry,
)
from fashionrec.baseline.recall.generator import generate_candidates, read_candidate_csv
from fashionrec.baseline.recall.item2item import (
    COOCCUR_WEEKS,
    ITEM2ITEM_RECALL_TOP_K,
    SEED_ITEMS,
    TOP_SIM_K,
)
from fashionrec.baseline.recall.popular import POPULAR_RECALL_TOP_K
from fashionrec.shared.domain.ids import canonical_item_id, canonical_user_id
from fashionrec.shared.metrics.ranking import hit_at_k, map_at_k, ndcg_at_k, recall_at_k


TRAIN_INTER = TRAIN_INTER_FILE
VALID_INTER = VALID_INTER_FILE
TEST_INTER = TEST_INTER_FILE
SASREC_RECALL_DIR = Path("outputs/recommendations")
FUSION_OUT_DIR = Path("outputs/recommendations")
EVAL_OUT_DIR = Path("outputs/evaluation")


def default_sasrec_recall_csv(eval_split: str, prefer_sasrecf: bool = True) -> Path:
    if prefer_sasrecf:
        path = SASREC_RECALL_DIR / f"sasrecf_{eval_split}.csv"
        if path.exists():
            return path
    return SASREC_RECALL_DIR / f"sasrec_{eval_split}.csv"


def _load_targets(path: Path) -> dict[str, set[str]]:
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["user_id:token", "item_id:token"],
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    frame["user_id:token"] = frame["user_id:token"].map(canonical_user_id)
    frame["item_id:token"] = frame["item_id:token"].map(canonical_item_id)
    return (
        frame.groupby("user_id:token")["item_id:token"]
        .apply(lambda values: {canonical_item_id(value) for value in values})
        .to_dict()
    )


def _recall_at_k(actual: set[str], pred: list[str], k: int) -> float:
    return recall_at_k(actual, pred, k)


def _hit_at_k(actual: set[str], pred: list[str], k: int) -> float:
    return hit_at_k(actual, pred, k)


def _ndcg_at_k(actual: set[str], pred: list[str], k: int) -> float:
    return ndcg_at_k(actual, pred, k)


def _map_at_k(actual: set[str], pred: list[str], k: int) -> float:
    return map_at_k(actual, pred, k)


@dataclass
class FusionEvalContext:
    targets: dict[str, set[str]]
    users: list[dict[str, Any]]
    sequence_channel: str
    final_top_k: int


def build_fusion_eval_context(
    eval_split: str = "valid",
    recall_top_k: int = 100,
    popular_recall_top_k: int = POPULAR_RECALL_TOP_K,
    category_popular_recall_top_k: int = CATEGORY_POPULAR_RECALL_TOP_K,
    item2item_recall_top_k: int = ITEM2ITEM_RECALL_TOP_K,
    item2item_cooccur_weeks: int = COOCCUR_WEEKS,
    item2item_top_sim_k: int = TOP_SIM_K,
    item2item_seed_items: int = SEED_ITEMS,
    category_popular_seed_items: int = CATEGORY_SEED_ITEMS,
    final_top_k: int = 12,
    sasrec_recall_csv: str | Path | None = None,
    sequence_channel: str | None = None,
    strict: bool = False,
    candidate_csv: str | Path | None = None,
    max_user_history: int = 100,
    data_dir: str | Path | None = None,
) -> FusionEvalContext:
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")

    recall_path = Path(sasrec_recall_csv) if sasrec_recall_csv else default_sasrec_recall_csv(eval_split)
    data_paths = ProcessedDataPaths.from_root(data_dir)
    eval_path = data_paths.valid_inter if eval_split == "valid" else data_paths.test_inter
    history_paths = history_paths_for_eval(eval_split, data_paths.train_inter, data_paths.valid_inter)
    assert_history_paths_allowed(
        eval_split,
        history_paths,
        data_paths.train_inter,
        data_paths.valid_inter,
        data_paths.test_inter,
    )
    user_history = build_user_history(*history_paths, max_user_history=max_user_history)
    targets = _load_targets(eval_path)

    if candidate_csv is not None:
        candidate_path = Path(candidate_csv)
        if strict and not candidate_path.exists():
            raise FileNotFoundError(f"Missing required candidate artifact: {candidate_path}")
        candidates = read_candidate_csv(candidate_path) if candidate_path.exists() else []
        bad_splits = sorted({row.split for row in candidates if row.split != eval_split})
        if bad_splits:
            raise ValueError(f"Candidate artifact split mismatch: expected={eval_split}, found={bad_splits}")
        sequence_names = sorted({row.channel for row in candidates if row.channel.startswith("sasrec")})
        resolved_sequence = sequence_channel or (sequence_names[0] if sequence_names else "sasrecf")
        if strict and not candidates:
            raise ValueError(f"Candidate artifact is empty: {candidate_path}")
        if strict and not sequence_names:
            raise ValueError(f"Candidate artifact has no SASRec/SASRecF channel: {candidate_path}")
    else:
        if strict and not recall_path.exists():
            raise FileNotFoundError(f"Missing required sequence recall: {recall_path}")
        sequence_map = load_channel_recall_csv(recall_path)
        resolved_sequence = sequence_channel or infer_sequence_channel(recall_path)
        registry = build_rule_channel_registry(
            history_paths,
            item2item_cooccur_weeks=item2item_cooccur_weeks,
            item2item_top_sim_k=item2item_top_sim_k,
            item2item_seed_items=item2item_seed_items,
            category_seed_items=category_popular_seed_items,
            item_file=data_paths.seq_item,
        )
        registry[resolved_sequence] = PrecomputedChannel(
            resolved_sequence,
            {user: [(item, score) for item, score, _rank in rows] for user, rows in sequence_map.items()},
        )
        candidates = generate_candidates(
            eval_users=targets,
            user_history=user_history,
            channels=registry,
            split=eval_split,
            top_k_by_channel={
                "popular": popular_recall_top_k,
                "category_popular": category_popular_recall_top_k,
                "item2item": item2item_recall_top_k,
                resolved_sequence: recall_top_k,
            },
        )

    by_user: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        by_user[candidate.user_id][candidate.channel].append((candidate.item_id, candidate.score))

    users = []
    for user_id, actual_items in targets.items():
        history = user_history.get(user_id, [])
        users.append(
            {
                "user_id": user_id,
                "actual_items": actual_items,
                "history": history,
                "history_set": set(history),
                "channel_candidates": dict(by_user.get(user_id, {})),
            }
        )
    return FusionEvalContext(targets, users, resolved_sequence, final_top_k)


def evaluate_fusion_map_at_k(
    context: FusionEvalContext,
    activity_weights: dict[ActivityTier, dict[str, float]],
    exclude_seen: bool = False,
) -> float:
    scores = []
    for row in context.users:
        weights = get_channel_weights_for_user(
            len(row["history"]),
            context.sequence_channel,
            activity_weights=activity_weights,
        )
        ranked = WeightedRRFRanker(weights, exclude_seen=exclude_seen).rank(
            user_id=row["user_id"],
            user_history=row["history_set"],
            channel_candidates=row["channel_candidates"],
            top_k=context.final_top_k,
        )
        scores.append(map_at_k(row["actual_items"], [item.item_id for item in ranked], context.final_top_k))
    return float(sum(scores) / len(scores)) if scores else 0.0


def evaluate_fusion(
    eval_split: str = "valid",
    recall_top_k: int = 100,
    popular_recall_top_k: int = POPULAR_RECALL_TOP_K,
    category_popular_recall_top_k: int = CATEGORY_POPULAR_RECALL_TOP_K,
    final_top_k: int = 12,
    popular_weight: float = 0.15,
    category_popular_weight: float = 0.15,
    item2item_weight: float = 0.25,
    sasrec_weight: float = 0.45,
    item2item_recall_top_k: int = ITEM2ITEM_RECALL_TOP_K,
    item2item_cooccur_weeks: int = COOCCUR_WEEKS,
    item2item_top_sim_k: int = TOP_SIM_K,
    item2item_seed_items: int = SEED_ITEMS,
    category_popular_seed_items: int = CATEGORY_SEED_ITEMS,
    sasrec_recall_csv: str | Path | None = None,
    adaptive_weights: bool = True,
    activity_weights: dict[ActivityTier, dict[str, float]] | None = None,
    exclude_seen: bool = False,
    sequence_channel: str | None = None,
    output_dir: str | Path = FUSION_OUT_DIR,
    evaluation_dir: str | Path = EVAL_OUT_DIR,
    strict: bool = False,
    candidate_csv: str | Path | None = None,
    max_user_history: int = 100,
    data_dir: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    context = build_fusion_eval_context(
        eval_split=eval_split,
        recall_top_k=recall_top_k,
        popular_recall_top_k=popular_recall_top_k,
        category_popular_recall_top_k=category_popular_recall_top_k,
        item2item_recall_top_k=item2item_recall_top_k,
        item2item_cooccur_weeks=item2item_cooccur_weeks,
        item2item_top_sim_k=item2item_top_sim_k,
        item2item_seed_items=item2item_seed_items,
        category_popular_seed_items=category_popular_seed_items,
        final_top_k=final_top_k,
        sasrec_recall_csv=sasrec_recall_csv,
        sequence_channel=sequence_channel,
        strict=strict,
        candidate_csv=candidate_csv,
        max_user_history=max_user_history,
        data_dir=data_dir,
    )
    weights_table = activity_weights or ACTIVITY_WEIGHTS
    fixed_weights = {
        "popular": popular_weight,
        "category_popular": category_popular_weight,
        "item2item": item2item_weight,
        context.sequence_channel: sasrec_weight,
    }
    metric_values = {"map": [], "recall": [], "ndcg": [], "hit": []}
    tier_counts: dict[str, int] = defaultdict(int)
    recommendation_rows = []

    for row in context.users:
        if adaptive_weights:
            tier = classify_activity_tier(len(row["history"]))
            tier_counts[tier] += 1
            weights = get_channel_weights_for_user(
                len(row["history"]),
                context.sequence_channel,
                activity_weights=weights_table,
            )
        else:
            weights = fixed_weights
        ranked = WeightedRRFRanker(weights, exclude_seen=exclude_seen).rank(
            user_id=row["user_id"],
            user_history=row["history_set"],
            channel_candidates=row["channel_candidates"],
            top_k=final_top_k,
        )
        predicted = [item.item_id for item in ranked]
        actual = row["actual_items"]
        metric_values["map"].append(map_at_k(actual, predicted, final_top_k))
        metric_values["recall"].append(recall_at_k(actual, predicted, final_top_k))
        metric_values["ndcg"].append(ndcg_at_k(actual, predicted, final_top_k))
        metric_values["hit"].append(hit_at_k(actual, predicted, final_top_k))
        recommendation_rows.extend(
            {
                "user_id": row["user_id"],
                "item_id": item.item_id,
                "score": item.score,
                "rank": item.rank,
                "split": eval_split,
                "channel": "fusion",
            }
            for item in ranked
        )

    resolved_output = Path(output_dir)
    resolved_evaluation = Path(evaluation_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    resolved_evaluation.mkdir(parents=True, exist_ok=True)
    rec_out = resolved_output / f"fusion_{eval_split}.csv"
    metric_out = resolved_evaluation / f"fusion_{eval_split}_metrics.json"
    with rec_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "item_id", "score", "rank", "split", "channel"],
        )
        writer.writeheader()
        writer.writerows(recommendation_rows)

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    metrics: dict[str, Any] = {
        f"MAP@{final_top_k}": mean(metric_values["map"]),
        f"Recall@{final_top_k}": mean(metric_values["recall"]),
        f"NDCG@{final_top_k}": mean(metric_values["ndcg"]),
        f"Hit@{final_top_k}": mean(metric_values["hit"]),
        "users_evaluated": len(context.targets),
        "adaptive_weights": adaptive_weights,
        "exclude_seen": exclude_seen,
        "sequence_channel": context.sequence_channel,
        "activity_weights": {tier: dict(values) for tier, values in weights_table.items()} if adaptive_weights else None,
        "popular_recall_top_k": popular_recall_top_k,
        "category_popular_recall_top_k": category_popular_recall_top_k,
        "recall_top_k": recall_top_k,
        "item2item_recall_top_k": item2item_recall_top_k,
        "item2item_cooccur_weeks": item2item_cooccur_weeks,
        "item2item_top_sim_k": item2item_top_sim_k,
        "item2item_seed_items": item2item_seed_items,
        "category_popular_seed_items": category_popular_seed_items,
        "weights": fixed_weights if not adaptive_weights else "per-user by activity tier",
        "activity_tier_counts": dict(tier_counts) if adaptive_weights else {},
        "eval_split": eval_split,
        "target_protocol": "deduplicated items from hm.<split>.inter",
    }
    metric_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved fusion recommendations: {rec_out}")
    print(f"Saved evaluation metrics: {metric_out}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return rec_out, metric_out, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fashionrec baseline evaluate",
        description="Evaluate the fixed four-channel Weighted RRF baseline.",
    )
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")
    parser.add_argument("--recall-top-k", type=int, default=100)
    parser.add_argument("--popular-recall-top-k", type=int, default=POPULAR_RECALL_TOP_K)
    parser.add_argument("--category-popular-recall-top-k", type=int, default=CATEGORY_POPULAR_RECALL_TOP_K)
    parser.add_argument("--final-top-k", type=int, default=12)
    parser.add_argument("--popular-weight", type=float, default=0.15)
    parser.add_argument("--category-popular-weight", type=float, default=0.15)
    parser.add_argument("--item2item-weight", type=float, default=0.25)
    parser.add_argument("--sasrec-weight", type=float, default=0.45)
    parser.add_argument("--item2item-recall-top-k", type=int, default=ITEM2ITEM_RECALL_TOP_K)
    parser.add_argument("--item2item-cooccur-weeks", type=int, default=COOCCUR_WEEKS)
    parser.add_argument("--item2item-top-sim-k", type=int, default=TOP_SIM_K)
    parser.add_argument("--item2item-seed-items", type=int, default=SEED_ITEMS)
    parser.add_argument("--category-popular-seed-items", type=int, default=CATEGORY_SEED_ITEMS)
    parser.add_argument("--sasrec-recall-csv", type=Path, default=None)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=FUSION_OUT_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=EVAL_OUT_DIR)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-user-history", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sequence-channel", type=str, default=None)
    parser.add_argument("--no-adaptive-weights", action="store_true")
    parser.add_argument("--exclude-seen", action="store_true")
    parser.add_argument("--weights-json", type=Path, default=None)
    args = parser.parse_args(argv)

    loaded_weights = None
    exclude_seen = args.exclude_seen
    if args.weights_json is not None:
        from fashionrec.baseline.evaluation.weight_search import load_best_weights

        payload = load_best_weights(args.weights_json)
        loaded_weights = payload["best_weights"]
        if "exclude_seen" in payload and not args.exclude_seen:
            exclude_seen = bool(payload["exclude_seen"])

    evaluate_fusion(
        eval_split=args.eval_split,
        recall_top_k=args.recall_top_k,
        popular_recall_top_k=args.popular_recall_top_k,
        category_popular_recall_top_k=args.category_popular_recall_top_k,
        final_top_k=args.final_top_k,
        popular_weight=args.popular_weight,
        category_popular_weight=args.category_popular_weight,
        item2item_weight=args.item2item_weight,
        sasrec_weight=args.sasrec_weight,
        item2item_recall_top_k=args.item2item_recall_top_k,
        item2item_cooccur_weeks=args.item2item_cooccur_weeks,
        item2item_top_sim_k=args.item2item_top_sim_k,
        item2item_seed_items=args.item2item_seed_items,
        category_popular_seed_items=args.category_popular_seed_items,
        sasrec_recall_csv=args.sasrec_recall_csv,
        adaptive_weights=not args.no_adaptive_weights,
        activity_weights=loaded_weights,
        exclude_seen=exclude_seen,
        sequence_channel=args.sequence_channel,
        candidate_csv=args.candidate_csv,
        output_dir=args.output_dir,
        evaluation_dir=args.evaluation_dir,
        strict=args.strict,
        max_user_history=args.max_user_history,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
