"""Prepare the data artifacts consumed by the stable Baseline application."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fashionrec.baseline.data.backtest import (
    BACKTEST_SCHEMA_VERSION,
    DEFAULT_N_WINDOWS,
    build_backtest_windows,
    required_preprocess_weeks,
)
from fashionrec.baseline.data.build_item_features import build_item_features
from fashionrec.baseline.data.build_sequences import (
    prepare_recbole_benchmark_files,
    read_max_item_list_length,
)
from fashionrec.baseline.data.filter import run_filter
from fashionrec.baseline.data.manifest import (
    SCHEMA_VERSION,
    build_processed_hm_manifest,
    write_manifest,
)
from fashionrec.baseline.data.preprocess import (
    MAX_USER_HISTORY,
    MIN_USER_PURCHASES,
    RAW_PATH,
    WEEKS,
    build_inter_file,
)
from fashionrec.baseline.data.split import (
    TEST_WEEKS,
    TRAIN_WEEKS,
    VALID_WEEKS,
    build_model_train_split,
    split_bounds_dict,
    split_by_time,
)
from fashionrec.shared.experiment.config import load_experiment_config


DEFAULT_CONFIG = Path("configs/baseline/models/sasrecf.yaml")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_ARTICLES = Path("data/raw/articles.csv")


def processed_layout(processed_dir: Path) -> dict[str, Path]:
    """Return only the artifact layout owned by the Baseline protocol."""

    root = Path(processed_dir)
    hm = root / "hm"
    seq = root / "hm_seq"
    return {
        "root": root,
        "hm": hm,
        "seq": seq,
        "filtered": root / "filtered",
        "inter": hm / "hm.inter",
        "train": hm / "hm.train.inter",
        "model_train": hm / "hm.model_train.inter",
        "valid": hm / "hm.valid.inter",
        "test": hm / "hm.test.inter",
        "seq_item": seq / "hm_seq.item",
        "backtest": root / "backtest",
        "manifest": root / "manifest.json",
    }


def select_transactions_input(*, with_filter: bool, filtered_path: Path | None = None) -> Path:
    """Select raw input explicitly; never reuse a stale global filtered file."""

    if not with_filter:
        return RAW_PATH
    if filtered_path is None:
        raise ValueError(
            "with_filter=True requires filtered_path produced in this run; "
            "refusing to reuse data/raw/filtered/"
        )
    path = Path(filtered_path)
    if not path.is_file():
        raise FileNotFoundError(f"filtered_path does not exist: {path}")
    return path


def _run_step(name: str, fn) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{name}]")
    print("=" * 60)
    started = time.perf_counter()
    fn()
    print(f"Done in {time.perf_counter() - started:.1f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fashionrec baseline data",
        description="Prepare Baseline hm.inter, time splits, SASRecF sequences, and hm_seq.item.",
    )
    parser.add_argument(
        "--with-filter",
        action="store_true",
        help="Create and use a fresh train-fitted filtered dataset inside this run.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-item-features",
        action="store_true",
        help="Skip hm_seq conversion and hm_seq.item (offline rule evaluation only).",
    )
    parser.add_argument("--experiment-config", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument(
        "--build-backtest",
        action="store_true",
        help="Write rolling interaction splits only; Baseline does not build PIT labels/features.",
    )
    parser.add_argument(
        "--articles",
        type=Path,
        default=DEFAULT_ARTICLES,
        help="articles.csv used to build the RecBole hm_seq.item file.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    experiment = load_experiment_config(args.experiment_config) if args.experiment_config else None

    processed_dir = Path(args.processed_dir) if args.processed_dir else DEFAULT_PROCESSED_DIR
    layout = processed_layout(processed_dir)
    layout["root"].mkdir(parents=True, exist_ok=True)

    protocol_weeks = experiment.data.total_weeks if experiment else WEEKS
    history_weeks = experiment.data.history_weeks if experiment else TRAIN_WEEKS
    valid_weeks = experiment.data.valid_weeks if experiment else VALID_WEEKS
    test_weeks = experiment.data.test_weeks if experiment else TEST_WEEKS
    n_windows = experiment.data.backtest_windows if experiment else DEFAULT_N_WINDOWS
    weeks = (
        required_preprocess_weeks(
            train_weeks=history_weeks,
            valid_weeks=valid_weeks,
            test_weeks=test_weeks,
            n_windows=n_windows,
        )
        if args.build_backtest
        else protocol_weeks
    )
    min_user_purchases = experiment.data.min_user_purchases if experiment else MIN_USER_PURCHASES
    max_user_history = experiment.data.max_user_history if experiment else MAX_USER_HISTORY

    total_steps = 6 if args.with_filter else 5
    step = 1
    transactions_input = RAW_PATH
    if args.with_filter:
        filtered_dir = layout["filtered"]
        _run_step(
            f"{step}/{total_steps} causal item sampling",
            lambda: run_filter(
                min_user_purchases=min_user_purchases,
                max_user_behaviors=max_user_history,
                weeks=weeks,
                valid_weeks=valid_weeks,
                test_weeks=test_weeks,
                output_dir=filtered_dir,
            ),
        )
        transactions_input = select_transactions_input(
            with_filter=True,
            filtered_path=filtered_dir / "transactions_train.csv",
        )
        step += 1
    else:
        transactions_input = select_transactions_input(with_filter=False)

    _run_step(
        f"{step}/{total_steps} preprocess",
        lambda: build_inter_file(
            transactions_path=transactions_input,
            output_path=layout["inter"],
            weeks=weeks,
            min_user_purchases=min_user_purchases,
            max_user_history=max_user_history,
        ),
    )
    step += 1

    split_result = None

    def _split() -> None:
        nonlocal split_result
        split_result = split_by_time(
            inter_path=layout["inter"],
            train_inter_path=layout["train"],
            valid_inter_path=layout["valid"],
            test_inter_path=layout["test"],
            total_weeks=protocol_weeks,
            train_weeks=history_weeks,
            valid_weeks=valid_weeks,
            test_weeks=test_weeks,
        )

    _run_step(f"{step}/{total_steps} split", _split)
    step += 1
    _run_step(
        f"{step}/{total_steps} model_train",
        lambda: build_model_train_split(
            train_path=layout["train"],
            output_path=layout["model_train"],
            min_user_purchases=min_user_purchases,
        ),
    )
    step += 1

    if args.build_backtest:
        if split_result is None:
            raise RuntimeError("split must finish before --build-backtest")
        _run_step(
            "build backtest interaction windows",
            lambda: build_backtest_windows(
                inter_path=layout["inter"],
                output_dir=layout["backtest"],
                train_weeks=history_weeks,
                valid_weeks=valid_weeks,
                test_weeks=test_weeks,
                n_windows=n_windows,
                max_date=split_result.max_date,
            ),
        )

    def _write_manifest() -> None:
        preprocess = {
            "application": "baseline",
            "schema_version": SCHEMA_VERSION,
            "weeks": weeks,
            "protocol_weeks": protocol_weeks,
            "min_user_purchases": min_user_purchases,
            "max_user_history": max_user_history,
            "with_filter": bool(args.with_filter),
            "transactions_input": str(transactions_input),
            "processed_dir": str(processed_dir),
            "skip_item_features": bool(args.skip_item_features),
            "sasrec_config": str(config_path),
            "experiment_config": str(args.experiment_config) if args.experiment_config else None,
            "experiment_name": experiment.experiment.name if experiment else None,
            "seed": experiment.experiment.seed if experiment else None,
            "deduplicate_user_day_item": experiment.data.deduplicate_user_day_item if experiment else True,
            "ranking_enabled": False,
            "build_backtest": bool(args.build_backtest),
            "backtest_schema_version": BACKTEST_SCHEMA_VERSION if args.build_backtest else None,
            "backtest_dir": str(layout["backtest"]) if args.build_backtest else None,
            "backtest_windows": n_windows if args.build_backtest else None,
            "seq_item_path": None if args.skip_item_features else str(layout["seq_item"]),
            "ranking_feature_parquets": False,
        }
        payload = build_processed_hm_manifest(
            processed_dir=processed_dir,
            raw_transactions=transactions_input,
            true_raw_transactions=RAW_PATH,
            preprocess=preprocess,
            split_bounds=split_bounds_dict(split_result) if split_result else {},
            repo_root=Path.cwd(),
        )
        out = write_manifest(payload, layout["manifest"])
        print(f"Wrote data manifest: {out}")

    if args.skip_item_features:
        print("\nSkipped hm_seq + hm_seq.item (--skip-item-features).")
        _write_manifest()
        return

    max_item_list_length = read_max_item_list_length(config_path)
    _run_step(
        f"{step}/{total_steps} hm_seq",
        lambda: prepare_recbole_benchmark_files(
            max_item_list_length,
            train_split_file=layout["model_train"],
            valid_split_file=layout["valid"],
            test_split_file=layout["test"],
            target_dir=layout["seq"],
            train_history_file=layout["train"],
            max_shopping_days=max_user_history if experiment else None,
        ),
    )
    step += 1
    _run_step(
        f"{step}/{total_steps} hm_seq.item",
        lambda: build_item_features(
            articles_path=args.articles,
            output_path=layout["seq_item"],
            inter_paths=(
                layout["seq"] / "hm_seq.train.inter",
                layout["seq"] / "hm_seq.valid.inter",
                layout["seq"] / "hm_seq.test.inter",
            ),
        ),
    )

    _write_manifest()
    print("\nBaseline data preparation finished.")
    print("Next: make train RUN_ID=<id>")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
