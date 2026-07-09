"""Valid-set grid search for per-tier multi-channel fusion weights."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.fusion.weighted_fusion import ACTIVITY_WEIGHTS, ActivityTier
from src.evaluate.offline_eval import (
    EVAL_OUT_DIR,
    FusionEvalContext,
    build_fusion_eval_context,
    evaluate_fusion_map_at_k,
)

DEFAULT_OUTPUT_JSON = EVAL_OUT_DIR / "best_fusion_weights.json"

# 各活跃度分层、各通道权重搜索范围（sequence 即 sasrecf）
TIER_WEIGHT_RANGES: dict[ActivityTier, dict[str, tuple[float, float]]] = {
    "high": {
        "sequence": (0.45, 0.70),
        "popular": (0.05, 0.25),
        "category_popular": (0.05, 0.25),
        "item2item": (0.10, 0.30),
    },
    "medium": {
        "sequence": (0.30, 0.55),
        "popular": (0.15, 0.40),
        "category_popular": (0.10, 0.30),
        "item2item": (0.10, 0.30),
    },
    "low": {
        "sequence": (0.00, 0.25),
        "popular": (0.40, 0.75),
        "category_popular": (0.10, 0.35),
        "item2item": (0.00, 0.20),
    },
    "cold_start": {
        "sequence": (0.00, 0.00),
        "popular": (0.60, 0.90),
        "category_popular": (0.10, 0.40),
        "item2item": (0.00, 0.00),
    },
}

CHANNEL_KEYS = ("sequence", "popular", "category_popular", "item2item")


def _grid_values(low: float, high: float, step: float) -> list[float]:
    if low == high:
        return [round(low, 4)]
    values: list[float] = []
    current = low
    while current <= high + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def _normalize_tier_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw[k] for k in CHANNEL_KEYS)
    if total <= 0:
        raise ValueError(f"Invalid tier weights (sum <= 0): {raw}")
    return {k: raw[k] / total for k in CHANNEL_KEYS}


def _in_range(value: float, bounds: tuple[float, float], tol: float = 1e-6) -> bool:
    return bounds[0] - tol <= value <= bounds[1] + tol


def generate_weight_candidates(
    tier: ActivityTier,
    step: float = 0.05,
) -> list[dict[str, float]]:
    """Generate normalized weight tuples for one activity tier within search ranges."""
    ranges = TIER_WEIGHT_RANGES[tier]
    seq_vals = _grid_values(ranges["sequence"][0], ranges["sequence"][1], step)
    pop_vals = _grid_values(ranges["popular"][0], ranges["popular"][1], step)
    cat_vals = _grid_values(ranges["category_popular"][0], ranges["category_popular"][1], step)
    i2i_vals = _grid_values(ranges["item2item"][0], ranges["item2item"][1], step)

    # 冷启动：sasrecf 与 item2item 固定为 0，popular + category_popular = 1
    if tier == "cold_start":
        candidates: list[dict[str, float]] = []
        for pop in pop_vals:
            cat = round(1.0 - pop, 4)
            if not _in_range(cat, ranges["category_popular"]):
                continue
            candidates.append(
                {
                    "sequence": 0.0,
                    "popular": pop,
                    "category_popular": cat,
                    "item2item": 0.0,
                }
            )
        return candidates

    candidates = []
    for seq, pop, cat, i2i in product(seq_vals, pop_vals, cat_vals, i2i_vals):
        raw = {"sequence": seq, "popular": pop, "category_popular": cat, "item2item": i2i}
        total = sum(raw.values())
        if abs(total - 1.0) > 1e-4:
            continue
        if all(_in_range(raw[k], ranges[k]) for k in CHANNEL_KEYS):
            candidates.append(raw)

    if candidates:
        return candidates

    # 若严格 sum=1 无结果，退化为生成后归一化并校验范围
    for seq, pop, cat in product(seq_vals, pop_vals, cat_vals):
        i2i = round(1.0 - seq - pop - cat, 4)
        if i2i < -1e-6:
            continue
        raw = {"sequence": seq, "popular": pop, "category_popular": cat, "item2item": max(i2i, 0.0)}
        normalized = _normalize_tier_weights(raw)
        if all(_in_range(normalized[k], ranges[k]) for k in CHANNEL_KEYS):
            candidates.append(normalized)
    return candidates


def save_best_weights(payload: dict[str, Any], output_path: Path | None = None) -> Path:
    output_path = output_path or DEFAULT_OUTPUT_JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def load_best_weights(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data.get("best_weights", data)
    activity_weights: dict[ActivityTier, dict[str, float]] = {}
    for tier in ("high", "medium", "low", "cold_start"):
        if tier not in best:
            raise KeyError(f"Missing tier '{tier}' in weights file: {path}")
        activity_weights[tier] = {k: float(best[tier][k]) for k in CHANNEL_KEYS}
    data["best_weights"] = activity_weights
    return data


def search_best_weights(
    context: FusionEvalContext,
    step: float = 0.05,
    exclude_seen: bool = False,
    max_passes: int = 2,
    verbose: bool = True,
) -> tuple[dict[ActivityTier, dict[str, float]], float]:
    """
    Coordinate-descent grid search: optimize each tier's weights on valid MAP@12.

    Only fusion weights change; recall candidates are fixed in context.
    """
    best_weights = copy.deepcopy(ACTIVITY_WEIGHTS)
    best_map = evaluate_fusion_map_at_k(context, best_weights, exclude_seen=exclude_seen)
    if verbose:
        print(f"Baseline MAP@12={best_map:.6f} exclude_seen={exclude_seen}")
        print(f"Baseline weights: {json.dumps(best_weights, ensure_ascii=False)}")

    tier_order: list[ActivityTier] = ["high", "medium", "low", "cold_start"]

    for pass_idx in range(1, max_passes + 1):
        if verbose:
            print(f"\n--- Pass {pass_idx}/{max_passes} ---")
        improved_any = False
        for tier in tier_order:
            candidates = generate_weight_candidates(tier, step=step)
            if verbose:
                print(f"Tier {tier}: {len(candidates)} candidate weight sets")
            for candidate in candidates:
                trial_weights = copy.deepcopy(best_weights)
                trial_weights[tier] = candidate
                trial_map = evaluate_fusion_map_at_k(
                    context, trial_weights, exclude_seen=exclude_seen
                )
                if trial_map > best_map + 1e-9:
                    best_map = trial_map
                    best_weights[tier] = candidate
                    improved_any = True
                    if verbose:
                        print(
                            f">> New best MAP@12={best_map:.6f} "
                            f"tier={tier} "
                            f"trial_weights={json.dumps(candidate, ensure_ascii=False)} "
                            f"all_tiers={json.dumps(best_weights, ensure_ascii=False)}"
                        )
        if not improved_any:
            if verbose:
                print("No improvement in this pass; stopping early.")
            break

    if verbose:
        print(f"\nFinal best MAP@12={best_map:.6f} exclude_seen={exclude_seen}")
        print(f"Final best weights: {json.dumps(best_weights, ensure_ascii=False)}")
    return best_weights, best_map


def run_weight_search(
    eval_split: str = "valid",
    step: float = 0.05,
    max_passes: int = 2,
    output_json: Path | None = None,
    sasrec_recall_csv: str | Path | None = None,
    compare_exclude_seen: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    if eval_split != "valid":
        raise ValueError("Weight search is only allowed on eval_split='valid'")

    if verbose:
        print("Building fusion eval context (recall computed once)...")
    context = build_fusion_eval_context(
        eval_split="valid",
        sasrec_recall_csv=sasrec_recall_csv,
    )

    modes = [False, True] if compare_exclude_seen else [False]
    mode_results: dict[str, dict[str, Any]] = {}

    for exclude_seen in modes:
        label = "exclude_seen=true" if exclude_seen else "exclude_seen=false"
        if verbose:
            print(f"\n========== Search mode: {label} ==========")
        best_weights, best_map = search_best_weights(
            context,
            step=step,
            exclude_seen=exclude_seen,
            max_passes=max_passes,
            verbose=verbose,
        )
        mode_results[label] = {
            "exclude_seen": exclude_seen,
            "best_map@12": best_map,
            "best_weights": best_weights,
        }

    selected_label = max(mode_results, key=lambda k: mode_results[k]["best_map@12"])
    selected = mode_results[selected_label]

    payload: dict[str, Any] = {
        "protocol": "hm_fusion_weight_search",
        "eval_split": "valid",
        "sequence_channel": context.sequence_channel,
        "search_step": step,
        "max_passes": max_passes,
        "exclude_seen": selected["exclude_seen"],
        "best_map@12": selected["best_map@12"],
        "best_weights": {
            tier: {k: float(selected["best_weights"][tier][k]) for k in CHANNEL_KEYS}
            for tier in ("high", "medium", "low", "cold_start")
        },
        "compared_exclude_seen": {
            label: {
                "exclude_seen": res["exclude_seen"],
                "best_map@12": res["best_map@12"],
                "best_weights": {
                    tier: {k: float(res["best_weights"][tier][k]) for k in CHANNEL_KEYS}
                    for tier in ("high", "medium", "low", "cold_start")
                },
            }
            for label, res in mode_results.items()
        },
        "selected_mode": selected_label,
    }

    out_path = save_best_weights(payload, output_json)
    if verbose:
        print(f"\nSaved best weights: {out_path}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Search fusion weights on valid MAP@12")
    parser.add_argument("--eval-split", choices=["valid"], default="valid")
    parser.add_argument("--step", type=float, default=0.05, help="Grid step (default: 0.05)")
    parser.add_argument("--max-passes", type=int, default=2, help="Coordinate descent passes")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--sasrec-recall-csv", type=Path, default=None)
    parser.add_argument(
        "--exclude-seen-only",
        choices=["both", "false", "true"],
        default="both",
        help="Search with exclude_seen=false, true, or both (default: both)",
    )
    args = parser.parse_args()

    compare = args.exclude_seen_only == "both"
    if args.exclude_seen_only == "false":
        compare = False
        # force only false - handled in run_weight_search by passing compare_exclude_seen=False
    elif args.exclude_seen_only == "true":
        # need custom single mode true - extend run_weight_search
        pass

    if args.exclude_seen_only == "true":
        context = build_fusion_eval_context(eval_split="valid", sasrec_recall_csv=args.sasrec_recall_csv)
        best_weights, best_map = search_best_weights(
            context, step=args.step, exclude_seen=True, max_passes=args.max_passes
        )
        payload = {
            "protocol": "hm_fusion_weight_search",
            "eval_split": "valid",
            "sequence_channel": context.sequence_channel,
            "search_step": args.step,
            "exclude_seen": True,
            "best_map@12": best_map,
            "best_weights": {
                tier: {k: float(best_weights[tier][k]) for k in CHANNEL_KEYS}
                for tier in ("high", "medium", "low", "cold_start")
            },
            "selected_mode": "exclude_seen=true",
        }
        save_best_weights(payload, args.output_json)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.exclude_seen_only == "false":
        context = build_fusion_eval_context(eval_split="valid", sasrec_recall_csv=args.sasrec_recall_csv)
        best_weights, best_map = search_best_weights(
            context, step=args.step, exclude_seen=False, max_passes=args.max_passes
        )
        payload = {
            "protocol": "hm_fusion_weight_search",
            "eval_split": "valid",
            "sequence_channel": context.sequence_channel,
            "search_step": args.step,
            "exclude_seen": False,
            "best_map@12": best_map,
            "best_weights": {
                tier: {k: float(best_weights[tier][k]) for k in CHANNEL_KEYS}
                for tier in ("high", "medium", "low", "cold_start")
            },
            "selected_mode": "exclude_seen=false",
        }
        save_best_weights(payload, args.output_json)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    run_weight_search(
        eval_split=args.eval_split,
        step=args.step,
        max_passes=args.max_passes,
        output_json=args.output_json,
        sasrec_recall_csv=args.sasrec_recall_csv,
        compare_exclude_seen=True,
    )


if __name__ == "__main__":
    main()
