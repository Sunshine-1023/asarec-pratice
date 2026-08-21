"""Execute one application-owned pipeline DAG in a run-scoped context."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from fashionrec.shared.experiment.context import RunContext, create_run_context
from fashionrec.shared.runtime.contracts import PipelineOptions, PipelineStep


PipelineBuilder = Callable[..., list[PipelineStep]]


def _run_step(step_no: int, total: int, step: PipelineStep, *, cwd: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{step_no}/{total}] {step.name}")
    print(f"命令: {' '.join(step.command)}")
    print("=" * 60)
    started = time.perf_counter()
    result = subprocess.run(step.command, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(
            f"Step {step_no} failed (exit {result.returncode}): {' '.join(step.command)}"
        )
    print(f"完成，耗时 {time.perf_counter() - started:.1f}s")


def _parser(*, application: str, default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m fashionrec.{application} pipeline",
        description=f"Run the isolated FashionRec {application} pipeline.",
    )
    parser.add_argument("--with-filter", action="store_true")
    parser.add_argument("--build-backtest", action="store_true")
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-checkpoint-selection", action="store_true")
    parser.add_argument("--skip-recall", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    parser.add_argument("--skip-weight-search", action="store_true")
    parser.add_argument("--skip-ranker", action="store_true")
    parser.add_argument("--skip-valid-eval", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--weights-json", type=Path, default=None)
    parser.add_argument("--experiment-config", type=Path, default=default_config)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--no-strict", action="store_true")
    return parser


def run_application_pipeline(
    argv: list[str] | None,
    *,
    application: str,
    default_config: Path,
    build_steps: PipelineBuilder,
    cwd: Path | None = None,
) -> None:
    args = _parser(application=application, default_config=default_config).parse_args(argv)
    context: RunContext = create_run_context(
        config_path=args.experiment_config,
        output_root=args.output_root,
        run_id=args.run_id,
        profile=application,
        strict=not args.no_strict,
        initialize=True,
    )
    options = PipelineOptions(
        with_filter=args.with_filter,
        skip_data_prep=args.skip_data_prep,
        skip_train=args.skip_train,
        skip_checkpoint_selection=args.skip_checkpoint_selection,
        skip_recall=args.skip_recall,
        skip_candidates=args.skip_candidates,
        skip_weight_search=args.skip_weight_search,
        skip_ranker=args.skip_ranker,
        skip_valid_eval=args.skip_valid_eval,
        skip_test_eval=args.skip_test_eval,
        weights_json=str(args.weights_json) if args.weights_json is not None else None,
        build_backtest=args.build_backtest,
    )
    steps = build_steps(context, python_executable=sys.executable, options=options)
    if not steps:
        print("No steps to run (all skipped).")
        context.write_manifest(status="no_steps", completed_steps=[])
        return

    root = cwd or Path.cwd()
    print(f"{application.title()} pipeline run_id={context.run_id}: {len(steps)} step(s)")
    completed: list[str] = []
    started = time.perf_counter()
    try:
        for index, step in enumerate(steps, start=1):
            _run_step(index, len(steps), step, cwd=root)
            completed.append(step.name)
    except BaseException:
        context.write_manifest(status="failed", completed_steps=completed)
        raise
    context.write_manifest(status="complete", completed_steps=completed)
    print(f"\nPipeline finished in {time.perf_counter() - started:.1f}s")
    print(f"Run artifacts: {context.artifacts.root}")
