"""Registry-driven rule recall export and candidate materialization."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from fashionrec.industrial.recall.union import (
    DEFAULT_UNION_TOP_K,
    UNION_SCHEMA_VERSION,
    build_union_evidence,
    union_candidates,
    write_union_evidence_csv,
)
from fashionrec.industrial.data.paths import ProcessedDataPaths
from fashionrec.industrial.data.split import (
    TEST_INTER_FILE,
    TRAIN_INTER_FILE,
    VALID_INTER_FILE,
    assert_history_paths_allowed,
    history_paths_for_eval,
)
from fashionrec.industrial.ranking.fusion import build_user_history, infer_sequence_channel, load_channel_recall_csv
from fashionrec.industrial.recall.category_popular import CATEGORY_POPULAR_RECALL_TOP_K, SEED_ITEMS as CATEGORY_SEED_ITEMS
from fashionrec.industrial.recall.generator import generate_candidates, write_candidate_csv
from fashionrec.industrial.recall.item2item import (
    COOCCUR_WEEKS,
    ITEM2ITEM_RECALL_TOP_K,
    SEED_ITEMS as ITEM2ITEM_SEED_ITEMS,
    TOP_SIM_K,
)
from fashionrec.industrial.recall.content import CONTENT_RECALL_TOP_K
from fashionrec.industrial.recall.popular import POPULAR_RECALL_TOP_K
from fashionrec.industrial.recall.repurchase import REPURCHASE_RECALL_TOP_K
from fashionrec.industrial.recall.style import STYLE_RECALL_TOP_K
from fashionrec.industrial.recall.channel_registry import PrecomputedChannel, build_rule_channel_registry, select_channels


TRAIN_INTER = TRAIN_INTER_FILE  # 兼容旧模块常量，实际路径由数据切分层统一定义
VALID_INTER = VALID_INTER_FILE
TEST_INTER = TEST_INTER_FILE
OUTPUT_DIR = Path("outputs/recommendations")

ChannelName = Literal[
    "popular",
    "category_popular",
    "item2item",
    "repurchase",
    "style",
    "content",
]
ALL_CHANNELS: tuple[ChannelName, ...] = (
    "popular",
    "category_popular",
    "item2item",
    "repurchase",
    "style",
    "content",
)


def _load_eval_users(eval_split: str, data_paths: ProcessedDataPaths) -> list[str]:
    path = data_paths.valid_inter if eval_split == "valid" else data_paths.test_inter
    if not path.exists():
        raise FileNotFoundError(f"Missing eval split file: {path}")
    frame = pd.read_csv(path, sep="\t", usecols=["user_id:token"])
    return sorted(frame["user_id:token"].astype(str).unique().tolist())


def export_rule_recalls(
    eval_split: str = "valid",
    channels: tuple[ChannelName, ...] = ALL_CHANNELS,
    top_k: int | None = None,
    output_dir: Path = OUTPUT_DIR,
    union_top_k: int = DEFAULT_UNION_TOP_K,
    candidate_output_dir: Path | None = None,
    sequence_recall_csv: Path | None = None,
    sequence_top_k: int = 100,
    channel_top_k: dict[str, int] | None = None,
    max_user_history: int = 100,
    item_file: Path | None = None,
    articles_path: Path | None = None,
    customers_path: Path | None = None,
    union_feature_version: str = UNION_SCHEMA_VERSION,
    item2item_cooccur_weeks: int = COOCCUR_WEEKS,
    item2item_top_sim_k: int = TOP_SIM_K,
    item2item_seed_items: int = ITEM2ITEM_SEED_ITEMS,
    category_seed_items: int = CATEGORY_SEED_ITEMS,
    data_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Build each index once and materialize one shared Candidate contract."""
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")
    unknown = [channel for channel in channels if channel not in ALL_CHANNELS]
    if unknown:
        raise ValueError(f"Unknown channels: {unknown}; available={list(ALL_CHANNELS)}")

    data_paths = ProcessedDataPaths.from_root(data_dir)
    resolved_item_file = item_file or data_paths.seq_item
    history_paths = history_paths_for_eval(eval_split, data_paths.train_inter, data_paths.valid_inter)
    assert_history_paths_allowed(eval_split, history_paths, data_paths.train_inter, data_paths.valid_inter, data_paths.test_inter)
    user_history = build_user_history(*history_paths, max_user_history=max_user_history)
    registry = build_rule_channel_registry(
        history_paths,
        item2item_cooccur_weeks=item2item_cooccur_weeks,
        item2item_top_sim_k=item2item_top_sim_k,
        item2item_seed_items=item2item_seed_items,
        category_seed_items=category_seed_items,
        item_file=resolved_item_file,
        customers_path=customers_path,
        articles_path=articles_path,
        channel_names=channels,
    )
    available = tuple(registry.keys())
    missing_registry = [channel for channel in channels if channel not in registry]
    if missing_registry:
        raise ValueError(f"Channels not in registry (missing articles_path?): {missing_registry}; available={available}")
    registry = select_channels(registry, list(channels))
    defaults = {
        "popular": POPULAR_RECALL_TOP_K,
        "category_popular": CATEGORY_POPULAR_RECALL_TOP_K,
        "item2item": ITEM2ITEM_RECALL_TOP_K,
        "repurchase": REPURCHASE_RECALL_TOP_K,
        "style": STYLE_RECALL_TOP_K,
        "content": CONTENT_RECALL_TOP_K,
    }
    top_k_by_channel = {
        channel: int(top_k if top_k is not None else (channel_top_k or {}).get(channel, defaults[channel]))
        for channel in channels
    }
    if sequence_recall_csv is not None:
        if not sequence_recall_csv.exists():
            raise FileNotFoundError(f"Missing sequence recall file: {sequence_recall_csv}")
        sequence_map = load_channel_recall_csv(sequence_recall_csv)
        sequence_name = infer_sequence_channel(sequence_recall_csv)
        registry[sequence_name] = PrecomputedChannel(
            sequence_name,
            {user: [(item, score) for item, score, _rank in rows] for user, rows in sequence_map.items()},
        )
        top_k_by_channel[sequence_name] = sequence_top_k

    candidates = generate_candidates(
        eval_users=_load_eval_users(eval_split, data_paths),
        user_history=user_history,
        channels=registry,
        split=eval_split,
        top_k_by_channel=top_k_by_channel,
    )
    outputs: dict[str, Path] = {}
    for channel in channels:
        path = output_dir / f"{channel}_{eval_split}.csv"
        outputs[channel] = write_candidate_csv(
            (candidate for candidate in candidates if candidate.channel == channel),
            path,
        )
        print(f"Saved {channel} recall ({eval_split}): {path}")

    union_dir = candidate_output_dir or output_dir
    union_path = union_dir / f"{eval_split}.csv"
    source_timestamp = datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    union_rows = union_candidates(
        candidates,
        top_k_items_per_user=union_top_k,
        source_timestamp=source_timestamp,
        feature_version=union_feature_version,
    )
    outputs["candidate_union"] = write_candidate_csv(union_rows, union_path)
    channel_order = tuple(top_k_by_channel.keys())
    evidence_path = union_dir / f"{eval_split}_evidence.csv"
    outputs["candidate_union_evidence"] = write_union_evidence_csv(
        build_union_evidence(
            candidates,
            top_k_items_per_user=union_top_k,
            channels=channel_order,
            source_timestamp=source_timestamp,
            feature_version=union_feature_version,
        ),
        evidence_path,
        channels=channel_order,
    )
    print(f"Saved candidate union ({eval_split}): {union_path}")
    print(f"Saved candidate union evidence ({eval_split}): {evidence_path}")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fashionrec candidates", description="Export registry-driven rule recall candidates")
    parser.add_argument("--eval-split", choices=["valid", "test", "both"], default="both")
    parser.add_argument("--channels", type=str, default="all")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--candidate-output-dir", type=Path, default=None)
    parser.add_argument("--sequence-recall-csv", type=Path, default=None)
    parser.add_argument("--sequence-top-k", type=int, default=100)
    parser.add_argument("--union-top-k", type=int, default=DEFAULT_UNION_TOP_K)
    parser.add_argument("--repurchase-top-k", type=int, default=REPURCHASE_RECALL_TOP_K)
    parser.add_argument("--style-top-k", type=int, default=STYLE_RECALL_TOP_K)
    parser.add_argument("--content-top-k", type=int, default=CONTENT_RECALL_TOP_K)
    parser.add_argument("--item-file", type=Path, default=None, help="Item features; defaults to --data-dir/hm_seq/hm_seq.item.")
    parser.add_argument("--articles-path", type=Path, default=None)
    parser.add_argument("--customers-path", type=Path, default=None)
    parser.add_argument("--popular-top-k", type=int, default=POPULAR_RECALL_TOP_K)
    parser.add_argument("--category-popular-top-k", type=int, default=CATEGORY_POPULAR_RECALL_TOP_K)
    parser.add_argument("--item2item-top-k", type=int, default=ITEM2ITEM_RECALL_TOP_K)
    parser.add_argument("--max-user-history", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed dataset root; defaults to data/processed.")
    args = parser.parse_args(argv)

    channels: tuple[ChannelName, ...]
    if args.channels.strip().lower() == "all":
        channels = ALL_CHANNELS
    else:
        channels = tuple(channel.strip() for channel in args.channels.split(",") if channel.strip())  # type: ignore[assignment]
    splits = ["valid", "test"] if args.eval_split == "both" else [args.eval_split]
    started = time.perf_counter()
    for split in splits:
        export_rule_recalls(
            eval_split=split,
            channels=channels,
            top_k=args.top_k,
            output_dir=args.output_dir,
            candidate_output_dir=args.candidate_output_dir,
            sequence_recall_csv=args.sequence_recall_csv,
            sequence_top_k=args.sequence_top_k,
            union_top_k=args.union_top_k,
            channel_top_k={
                "popular": args.popular_top_k,
                "category_popular": args.category_popular_top_k,
                "item2item": args.item2item_top_k,
                "repurchase": args.repurchase_top_k,
                "style": args.style_top_k,
                "content": args.content_top_k,
            },
            max_user_history=args.max_user_history,
            item_file=args.item_file,
            articles_path=args.articles_path,
            customers_path=args.customers_path,
            data_dir=args.data_dir,
        )
    print(f"All rule-based recalls finished in {time.perf_counter() - started:.1f}s")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
