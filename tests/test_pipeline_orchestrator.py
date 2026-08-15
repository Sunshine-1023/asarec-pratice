"""Tests for config-driven, run-scoped pipeline planning."""

from __future__ import annotations

from pathlib import Path

from src.experiment.context import create_run_context
from src.pipeline.orchestrator import PipelineOptions, build_pipeline_steps


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
    selection_index = next(index for index, step in enumerate(steps) if "MAP" in step.name)
    valid_recall_index = next(index for index, step in enumerate(steps) if step.name == "SASRecF 召回 valid")
    assert selection_index < valid_recall_index
    selected_model = str(tmp_path / "run-1" / "checkpoints" / "sasrecf_selected.pth")
    recall_commands = [" ".join(step.command) for step in steps if step.name.startswith("SASRecF 召回")]
    assert all(f"--model-file {selected_model}" in command for command in recall_commands)
    assert all("outputs/recommendations" not in command for command in commands)


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
    assert steps[-1].name == "离线排序评估 test"
    earlier_commands = [" ".join(step.command) for step in steps[:-1]]
    assert all("run_offline_eval.py --eval-split test" not in command for command in earlier_commands)
