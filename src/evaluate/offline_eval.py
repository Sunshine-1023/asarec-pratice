"""Run multi-channel fusion and offline evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

from src.fusion.weighted_fusion import build_user_history, fuse_candidates, load_channel_recall_csv
from src.recall.itemcf import build_itemcf_index, recall_itemcf
from src.recall.popular import build_popular_index, recall_popular


TRAIN_INTER = Path("data/processed/hm/hm.train.inter")
VALID_INTER = Path("data/processed/hm/hm.valid.inter")
TEST_INTER = Path("data/processed/hm/hm.test.inter")

SASREC_RECALL_DIR = Path("outputs/recommendations")
FUSION_OUT_DIR = Path("outputs/recommendations")
EVAL_OUT_DIR = Path("outputs/evaluation")


def default_sasrec_recall_csv(eval_split: str) -> Path:
    return SASREC_RECALL_DIR / f"sasrec_{eval_split}.csv"


def _load_targets(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", usecols=["user_id:token", "item_id:token"])
    grouped = (
        df.groupby("user_id:token")["item_id:token"]
        .apply(lambda s: {str(x) for x in s.tolist()})
        .to_dict()
    )
    return grouped


def _recall_at_k(actual: set[str], pred: list[str], k: int) -> float:
    if not actual:
        return 0.0
    return len(set(pred[:k]) & actual) / len(actual)


def _hit_at_k(actual: set[str], pred: list[str], k: int) -> float:
    return 1.0 if set(pred[:k]) & actual else 0.0


def _ndcg_at_k(actual: set[str], pred: list[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(pred[:k]):
        if item in actual:
            dcg += 1.0 / (math.log2(i + 2))
    ideal_hits = min(len(actual), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / (math.log2(i + 2)) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _map_at_k(actual: set[str], pred: list[str], k: int) -> float:
    """
    Mean Average Precision at K for one user.

    AP@K = sum(P@i * rel_i) / min(|actual|, K)
    """
    if not actual:
        return 0.0

    hits = 0
    ap_sum = 0.0
    for i, item in enumerate(pred[:k], start=1):
        if item in actual:
            hits += 1
            ap_sum += hits / i

    denom = min(len(actual), k)
    return ap_sum / denom if denom > 0 else 0.0


def evaluate_fusion(
    eval_split: str = "valid",
    recall_top_k: int = 100,
    final_top_k: int = 12,
    popular_weight: float = 0.2,
    itemcf_weight: float = 0.3,
    sasrec_weight: float = 0.5,
    itemcf_min_cooccur: int = 2,
    sasrec_recall_csv: str | Path | None = None,
) -> tuple[Path, Path, dict[str, float]]:
    """
    Run multi-channel recall fusion and evaluate on valid/test split.

    Data flow:
      train or train+valid    -> build popular / itemcf indexes (depends on eval split)
      train(+valid).inter     -> user history for each eval user
      valid or test.inter     -> ground truth labels
      outputs/recommendations/sasrec_{eval_split}.csv (optional) -> SASRec channel candidates
    """
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")

    sasrec_recall_csv = (
        Path(sasrec_recall_csv)
        if sasrec_recall_csv is not None
        else default_sasrec_recall_csv(eval_split)
    )

    eval_path = VALID_INTER if eval_split == "valid" else TEST_INTER
    history_paths = [TRAIN_INTER] if eval_split == "valid" else [TRAIN_INTER, VALID_INTER]

    user_history_map = build_user_history(*history_paths)
    targets = _load_targets(eval_path)

    popular_index = build_popular_index(*history_paths)
    itemcf_index = build_itemcf_index(
        history_paths,
        min_cooccur=itemcf_min_cooccur,
    )
    if not sasrec_recall_csv.exists():
        print(
            f"Warning: SASRec recall file not found: {sasrec_recall_csv}. "
            "Fusion will run without SASRec channel."
        )
    sasrec_map = load_channel_recall_csv(sasrec_recall_csv)

    weights = {
        "popular": popular_weight,
        "itemcf": itemcf_weight,
        "sasrec": sasrec_weight,
    }

    FUSION_OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec_out = FUSION_OUT_DIR / f"fusion_{eval_split}.csv"
    metric_out = EVAL_OUT_DIR / f"fusion_{eval_split}_metrics.json"

    maps, recalls, ndcgs, hits = [], [], [], []
    rows = []

    for user_id, actual_items in targets.items():
        history = user_history_map.get(user_id, [])
        history_set = set(history)

        pop_cands = recall_popular(popular_index, user_history=history_set, top_k=recall_top_k)
        itemcf_cands = recall_itemcf(history, itemcf_index, top_k=recall_top_k)
        sasrec_cands = [(iid, score) for iid, score, _ in sasrec_map.get(user_id, [])[:recall_top_k]]

        fused = fuse_candidates(
            user_id=user_id,
            user_history=history_set,
            channel_candidates={
                "popular": pop_cands,
                "itemcf": itemcf_cands,
                "sasrec": sasrec_cands,
            },
            channel_weights=weights,
            top_k=final_top_k,
        )

        pred_items = [item_id for item_id, _ in fused]
        maps.append(_map_at_k(actual_items, pred_items, final_top_k))
        recalls.append(_recall_at_k(actual_items, pred_items, final_top_k))
        ndcgs.append(_ndcg_at_k(actual_items, pred_items, final_top_k))
        hits.append(_hit_at_k(actual_items, pred_items, final_top_k))

        for rank, (item_id, score) in enumerate(fused, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "score": score,
                    "rank": rank,
                    "split": eval_split,
                    "channel": "fusion",
                }
            )

    with rec_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "item_id", "score", "rank", "split", "channel"],
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        f"MAP@{final_top_k}": float(sum(maps) / len(maps)) if maps else 0.0,
        f"Recall@{final_top_k}": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        f"NDCG@{final_top_k}": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        f"Hit@{final_top_k}": float(sum(hits) / len(hits)) if hits else 0.0,
        "users_evaluated": len(targets),
        "weights": weights,
        "eval_split": eval_split,
    }
    metric_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved fusion recommendations: {rec_out}")
    print(f"Saved evaluation metrics: {metric_out}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return rec_out, metric_out, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-channel fusion offline evaluation")
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")
    parser.add_argument("--recall-top-k", type=int, default=100)
    parser.add_argument("--final-top-k", type=int, default=12)
    parser.add_argument("--popular-weight", type=float, default=0.2)
    parser.add_argument("--itemcf-weight", type=float, default=0.3)
    parser.add_argument("--sasrec-weight", type=float, default=0.5)
    parser.add_argument("--itemcf-min-cooccur", type=int, default=2)
    parser.add_argument("--sasrec-recall-csv", type=Path, default=None)
    args = parser.parse_args()

    evaluate_fusion(
        eval_split=args.eval_split,
        recall_top_k=args.recall_top_k,
        final_top_k=args.final_top_k,
        popular_weight=args.popular_weight,
        itemcf_weight=args.itemcf_weight,
        sasrec_weight=args.sasrec_weight,
        itemcf_min_cooccur=args.itemcf_min_cooccur,
        sasrec_recall_csv=args.sasrec_recall_csv,
    )


if __name__ == "__main__":
    main()

