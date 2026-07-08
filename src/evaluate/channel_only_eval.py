"""Evaluate single-channel recall quality (popular/itemcf) with MAP@K."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

from src.fusion.weighted_fusion import build_user_history
from src.recall.itemcf import build_itemcf_index, recall_itemcf
from src.recall.popular import build_popular_index, recall_popular

TRAIN_INTER = Path("data/processed/hm/hm.train.inter")
VALID_INTER = Path("data/processed/hm/hm.valid.inter")
TEST_INTER = Path("data/processed/hm/hm.test.inter")

REC_OUT_DIR = Path("outputs/recommendations")
EVAL_OUT_DIR = Path("outputs/evaluation")


def _load_targets(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", usecols=["user_id:token", "item_id:token"])
    return (
        df.groupby("user_id:token")["item_id:token"]
        .apply(lambda s: {str(x) for x in s.tolist()})
        .to_dict()
    )


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


def _run_channel(
    channel: str,
    eval_split: str,
    recall_top_k: int,
    final_top_k: int,
    itemcf_min_cooccur: int,
) -> tuple[Path, Path, dict[str, float]]:
    if channel not in {"popular", "itemcf"}:
        raise ValueError("channel must be 'popular' or 'itemcf'")
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")

    eval_path = VALID_INTER if eval_split == "valid" else TEST_INTER
    history_paths = [TRAIN_INTER] if eval_split == "valid" else [TRAIN_INTER, VALID_INTER]

    user_history_map = build_user_history(*history_paths)
    targets = _load_targets(eval_path)

    popular_index = build_popular_index(*history_paths) if channel == "popular" else None
    itemcf_index = (
        build_itemcf_index(history_paths, min_cooccur=itemcf_min_cooccur)
        if channel == "itemcf"
        else None
    )

    REC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec_out = REC_OUT_DIR / f"{channel}_{eval_split}.csv"
    metric_out = EVAL_OUT_DIR / f"{channel}_only_{eval_split}_metrics.json"

    maps, recalls, ndcgs, hits = [], [], [], []
    rows = []

    for user_id, actual_items in targets.items():
        history = user_history_map.get(user_id, [])
        history_set = set(history)

        if channel == "popular":
            cands = recall_popular(popular_index, user_history=history_set, top_k=recall_top_k)  # type: ignore[arg-type]
        else:
            cands = recall_itemcf(history, itemcf_index, top_k=recall_top_k)  # type: ignore[arg-type]

        pred_items = [item_id for item_id, _ in cands[:final_top_k]]
        maps.append(_map_at_k(actual_items, pred_items, final_top_k))
        recalls.append(_recall_at_k(actual_items, pred_items, final_top_k))
        ndcgs.append(_ndcg_at_k(actual_items, pred_items, final_top_k))
        hits.append(_hit_at_k(actual_items, pred_items, final_top_k))

        for rank, (item_id, score) in enumerate(cands[:final_top_k], start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "score": score,
                    "rank": rank,
                    "split": eval_split,
                    "channel": channel,
                }
            )

    with rec_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["user_id", "item_id", "score", "rank", "split", "channel"]
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        f"MAP@{final_top_k}": float(sum(maps) / len(maps)) if maps else 0.0,
        f"Recall@{final_top_k}": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        f"NDCG@{final_top_k}": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        f"Hit@{final_top_k}": float(sum(hits) / len(hits)) if hits else 0.0,
        "users_evaluated": len(targets),
        "channel": channel,
        "eval_split": eval_split,
    }
    metric_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved channel recommendations: {rec_out}")
    print(f"Saved channel metrics: {metric_out}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return rec_out, metric_out, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate popular/itemcf channels only")
    parser.add_argument("--channel", choices=["popular", "itemcf", "both"], default="both")
    parser.add_argument("--eval-split", choices=["valid", "test", "both"], default="both")
    parser.add_argument("--recall-top-k", type=int, default=100)
    parser.add_argument("--final-top-k", type=int, default=12)
    parser.add_argument("--itemcf-min-cooccur", type=int, default=2)
    args = parser.parse_args()

    channels = ["popular", "itemcf"] if args.channel == "both" else [args.channel]
    splits = ["valid", "test"] if args.eval_split == "both" else [args.eval_split]

    for channel in channels:
        for split in splits:
            _run_channel(
                channel=channel,
                eval_split=split,
                recall_top_k=args.recall_top_k,
                final_top_k=args.final_top_k,
                itemcf_min_cooccur=args.itemcf_min_cooccur,
            )


if __name__ == "__main__":
    main()
