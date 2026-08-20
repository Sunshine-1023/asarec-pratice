"""Tests for config-driven, run-scoped pipeline planning."""

from __future__ import annotations

from pathlib import Path

from fashionrec.experiment.context import create_run_context
from fashionrec.pipeline.orchestrator import PipelineOptions, build_pipeline_steps


def test_pipeline_steps_share_one_run_artifact_tree(tmp_path: Path) -> None:
    context = create_run_context(
        "configs/experiment.yaml",
        output_root=tmp_path,
        run_id="run-1",
        strict=True,
    )
    steps = build_pipeline_steps(context, python_executable="python", options=PipelineOptions(skip_train=True))
    commands = [" ".join(step.command) for step in steps]
    assert any(str(tmp_path / "run-1" / "recall" / "sasrecf_valid.csv") in command for command in commands)
    assert any(str(tmp_path / "run-1" / "candidates" / "valid.csv") in command for command in commands)
    assert any("--final-top-k 12" in command for command in commands)
    configured_commands = [command for command in commands if " fashionrec candidates " in command or " fashionrec weights " in command or " fashionrec evaluate " in command]
    assert configured_commands
    assert all("--max-user-history 200" in command for command in configured_commands)
    data_command = next(command for command in commands if " fashionrec data " in command)
    assert f"--processed-dir {tmp_path / 'run-1' / 'data'}" in data_command
    assert "--with-filter" not in data_command
    assert "--build-events" not in data_command
    assert "--build-baskets" not in data_command
    assert "--build-labels" not in data_command
    assert "--build-backtest" not in data_command
    selection_index = next(index for index, step in enumerate(steps) if "MAP" in step.name)
    valid_recall_index = next(index for index, step in enumerate(steps) if step.name == "SASRecF 召回 valid")
    assert selection_index < valid_recall_index
    selected_model = str(tmp_path / "run-1" / "checkpoints" / "sasrecf_selected.pth")
    recall_commands = [" ".join(step.command) for step in steps if step.name.startswith("SASRecF 召回")]
    assert all(f"--model-file {selected_model}" in command for command in recall_commands)
    weights_path = str(tmp_path / "run-1" / "ranking" / "best_fusion_weights.json")
    eval_commands = [" ".join(step.command) for step in steps if step.name.startswith("离线排序评估")]
    assert len(eval_commands) == 2
    assert all(f"--weights-json {weights_path}" in command for command in eval_commands)
    assert all("outputs/recommendations" not in command for command in commands)
    run_data = tmp_path / "run-1" / "data"
    data_consumers = [
        command
        for command in commands
        if any(f" fashionrec {name} " in command for name in ("train", "select-checkpoint", "recall", "candidates", "weights", "evaluate"))
    ]
    assert data_consumers
    assert all(command.count("--data-dir") == 1 for command in data_consumers)
    assert all(f"--data-dir {run_data}" in command for command in data_consumers)
    candidate_commands = [command for command in commands if " fashionrec candidates " in command]
    assert all("--articles-path data/raw/articles.csv" in command for command in candidate_commands)
    assert all("--customers-path data/raw/customers.csv" in command for command in candidate_commands)


def test_pipeline_backtest_flag_does_not_train_three_models(tmp_path: Path) -> None:
    context = create_run_context(
        "configs/experiment.yaml",
        output_root=tmp_path,
        run_id="run-bt",
        strict=True,
    )
    steps = build_pipeline_steps(
        context,
        python_executable="python",
        options=PipelineOptions(build_backtest=True),
    )
    train_steps = [step for step in steps if step.name == "训练 SASRecF"]
    assert len(train_steps) == 1
    data_command = next(" ".join(step.command) for step in steps if " fashionrec data " in " ".join(step.command))
    assert "--build-backtest" in data_command
    assert steps[-1].name == "离线排序评估 test"


def test_pipeline_with_filter_still_writes_into_run_data_dir(tmp_path: Path) -> None:
    context = create_run_context(
        "configs/experiment.yaml",
        output_root=tmp_path,
        run_id="run-filter",
        strict=True,
    )
    steps = build_pipeline_steps(
        context,
        python_executable="python",
        options=PipelineOptions(with_filter=True, skip_train=True, skip_checkpoint_selection=True, skip_recall=True, skip_candidates=True, skip_weight_search=True, skip_valid_eval=True, skip_test_eval=True),
    )
    command = " ".join(steps[0].command)
    assert steps[0].name == "数据准备"
    assert "--with-filter" in command
    assert f"--processed-dir {tmp_path / 'run-filter' / 'data'}" in command


def test_pipeline_can_plan_read_only_smoke_with_all_mutating_stages_skipped(tmp_path: Path) -> None:
    context = create_run_context("configs/experiment.yaml", output_root=tmp_path, run_id="run-2")
    options = PipelineOptions(
        skip_data_prep=True,
        skip_train=True,
        skip_checkpoint_selection=True,
        skip_recall=True,
        skip_candidates=True,
        skip_weight_search=True,
        skip_valid_eval=True,
        skip_test_eval=True,
    )
    assert build_pipeline_steps(context, python_executable="python", options=options) == []


def test_test_evaluation_is_the_final_pipeline_step(tmp_path: Path) -> None:
    context = create_run_context("configs/experiment.yaml", output_root=tmp_path, run_id="run-3")
    steps = build_pipeline_steps(context, python_executable="python", options=PipelineOptions())
    train_command = next(" ".join(step.command) for step in steps if step.name == "训练 SASRecF")
    assert "--skip-preprocess" not in train_command
    assert "-m fashionrec train --config configs/sasrecf.yaml" in train_command
    assert steps[-1].name == "离线排序评估 test"
    earlier_commands = [" ".join(step.command) for step in steps[:-1]]
    assert all("fashionrec evaluate --eval-split test" not in command for command in earlier_commands)


def test_pipeline_uses_only_the_unified_cli(tmp_path: Path) -> None:
    context = create_run_context("configs/experiment.yaml", output_root=tmp_path, run_id="run-4")
    commands = [" ".join(step.command) for step in build_pipeline_steps(context, python_executable="python", options=PipelineOptions())]
    assert commands
    assert all(command.startswith("python -m fashionrec ") for command in commands)
