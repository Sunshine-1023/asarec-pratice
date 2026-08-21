"""Tests for config-driven, run-scoped pipeline planning."""

from __future__ import annotations

from pathlib import Path

from fashionrec.baseline.recall.registry import BASELINE_RULE_CHANNELS
from fashionrec.experiment.context import create_run_context
from fashionrec.pipeline.orchestrator import PipelineOptions, build_pipeline_steps


def test_pipeline_steps_share_one_run_artifact_tree(tmp_path: Path) -> None:
    context = create_run_context(
        "configs/baseline/experiment.yaml",
        output_root=tmp_path,
        run_id="run-1",
        strict=True,
    )
    steps = build_pipeline_steps(context, python_executable="python", options=PipelineOptions(skip_train=True))
    commands = [" ".join(step.command) for step in steps]
    run_root = tmp_path / "baseline" / "run-1"
    assert any(str(run_root / "recall" / "sasrecf_valid.csv") in command for command in commands)
    assert any(str(run_root / "candidates" / "valid.csv") in command for command in commands)
    assert any("--final-top-k 12" in command for command in commands)
    configured_commands = [command for command in commands if any(f" fashionrec.baseline {stage} " in command for stage in ("candidates", "weights", "evaluate"))]
    assert configured_commands
    assert all("--max-user-history 200" in command for command in configured_commands)
    data_command = next(command for command in commands if " fashionrec.baseline data " in command)
    assert f"--processed-dir {run_root / 'data'}" in data_command
    assert "--with-filter" not in data_command
    assert "--build-events" not in data_command
    assert "--build-baskets" not in data_command
    assert "--build-labels" not in data_command
    assert "--build-backtest" not in data_command
    selection_index = next(index for index, step in enumerate(steps) if "MAP" in step.name)
    valid_recall_index = next(index for index, step in enumerate(steps) if step.name == "SASRecF 召回 valid")
    assert selection_index < valid_recall_index
    selected_model = str(run_root / "checkpoints" / "sasrecf_selected.pth")
    recall_commands = [" ".join(step.command) for step in steps if step.name.startswith("SASRecF 召回")]
    assert all(f"--model-file {selected_model}" in command for command in recall_commands)
    weights_path = str(run_root / "ranking" / "best_fusion_weights.json")
    eval_commands = [" ".join(step.command) for step in steps if step.name.startswith("离线排序评估")]
    assert len(eval_commands) == 2
    assert all(f"--weights-json {weights_path}" in command for command in eval_commands)
    assert all("outputs/recommendations" not in command for command in commands)
    run_data = run_root / "data"
    data_consumers = [
        command
        for command in commands
        if any(f" fashionrec.baseline {name} " in command for name in ("train", "select-checkpoint", "recall", "candidates", "weights", "evaluate"))
    ]
    assert data_consumers
    assert all(command.count("--data-dir") == 1 for command in data_consumers)
    assert all(f"--data-dir {run_data}" in command for command in data_consumers)
    candidate_commands = [command for command in commands if " fashionrec.baseline candidates " in command]
    assert BASELINE_RULE_CHANNELS == ("popular", "category_popular", "item2item")
    assert all("--articles-path data/raw/articles.csv" in command for command in candidate_commands)
    assert all("--customers-path data/raw/customers.csv" in command for command in candidate_commands)


def test_pipeline_backtest_flag_does_not_train_three_models(tmp_path: Path) -> None:
    context = create_run_context(
        "configs/baseline/experiment.yaml",
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
    data_command = next(" ".join(step.command) for step in steps if " fashionrec.baseline data " in " ".join(step.command))
    assert "--build-backtest" in data_command
    assert steps[-1].name == "离线排序评估 test"


def test_pipeline_with_filter_still_writes_into_run_data_dir(tmp_path: Path) -> None:
    context = create_run_context(
        "configs/baseline/experiment.yaml",
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
    assert f"--processed-dir {tmp_path / 'baseline' / 'run-filter' / 'data'}" in command


def test_pipeline_can_plan_read_only_smoke_with_all_mutating_stages_skipped(tmp_path: Path) -> None:
    context = create_run_context("configs/baseline/experiment.yaml", output_root=tmp_path, run_id="run-2")
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
    context = create_run_context("configs/baseline/experiment.yaml", output_root=tmp_path, run_id="run-3")
    steps = build_pipeline_steps(context, python_executable="python", options=PipelineOptions())
    train_command = next(" ".join(step.command) for step in steps if step.name == "训练 SASRecF")
    assert "--skip-preprocess" not in train_command
    assert "-m fashionrec.baseline train" in train_command
    assert steps[-1].name == "离线排序评估 test"
    earlier_commands = [" ".join(step.command) for step in steps[:-1]]
    assert all("fashionrec.baseline evaluate --eval-split test" not in command for command in earlier_commands)


def test_pipeline_adds_ranker_steps_only_when_enabled(tmp_path: Path) -> None:
    default_steps = build_pipeline_steps(
        create_run_context("configs/baseline/experiment.yaml", output_root=tmp_path, run_id="same"),
        python_executable="python",
        options=PipelineOptions(),
    )
    assert all("LambdaRank" not in step.name for step in default_steps)

    steps = build_pipeline_steps(
        create_run_context("configs/industrial/experiment.yaml", output_root=tmp_path, run_id="same"),
        python_executable="python",
        options=PipelineOptions(skip_data_prep=True, skip_train=True, skip_checkpoint_selection=True, skip_recall=True, skip_candidates=True, skip_weight_search=True),
    )
    names = [step.name for step in steps]
    assert names[0] == "复用唯一 SASRecF 生成 LambdaRank 序列证据"
    assert names[1] == "构建 LambdaRank 训练表（SASRecF 简单复用）"
    assert names[2] == "训练 LightGBM LambdaRank"
    assert "LambdaRank 打分 valid" in names
    assert names[-1] == "离线排序评估 test"
    sequence_command = " ".join(steps[0].command)
    dataset_command = " ".join(steps[1].command)
    train_command = " ".join(steps[2].command)
    industrial_root = tmp_path / "industrial" / "same"
    assert f"--output-dir {industrial_root / 'ranking'}" in dataset_command
    assert "-m fashionrec.industrial ranker-sequence" in sequence_command
    assert f"--model-file {industrial_root / 'checkpoints' / 'sasrecf_selected.pth'}" in sequence_command
    assert "--checkpoint-dir" not in sequence_command
    assert f"--sequence-feature-dir {industrial_root / 'ranking' / 'sasrecf_model_reuse'}" in dataset_command
    assert f"--train-parquet {industrial_root / 'ranking' / 'train.parquet'}" in train_command
    assert "-m fashionrec.industrial ranker-train" in train_command
    eval_commands = [" ".join(step.command) for step in steps if step.name.startswith("离线排序评估")]
    assert eval_commands
    assert all(f"--ranker-scored-csv {industrial_root / 'ranking' / 'valid_scored.csv'}" in command or f"--ranker-scored-csv {industrial_root / 'ranking' / 'test_scored.csv'}" in command for command in eval_commands)
    assert all(f"--labels-dir {industrial_root / 'data' / 'labels'}" in command for command in eval_commands)
    default_eval = [" ".join(step.command) for step in default_steps if step.name.startswith("离线排序评估")]
    assert default_eval
    assert all("--ranker-scored-csv" not in command for command in default_eval)
    assert all("--labels-dir" not in command for command in default_eval)


def test_industrial_skip_ranker_keeps_industrial_data_and_eval_protocol(tmp_path: Path) -> None:
    context = create_run_context(
        "configs/industrial/experiment.yaml",
        output_root=tmp_path,
        run_id="staged",
        profile="industrial",
    )
    steps = build_pipeline_steps(
        context,
        python_executable="python",
        options=PipelineOptions(
            skip_train=True,
            skip_checkpoint_selection=True,
            skip_recall=True,
            skip_candidates=True,
            skip_weight_search=True,
            skip_ranker=True,
        ),
    )
    commands = [" ".join(step.command) for step in steps]
    data_command = next(command for command in commands if " fashionrec.industrial data " in command)
    assert "--processed-dir" in data_command
    assert all("ranker-train" not in command and "ranker-dataset" not in command for command in commands)
    eval_commands = [command for command in commands if " fashionrec.industrial evaluate " in command]
    labels_dir = tmp_path / "industrial" / "staged" / "data" / "labels"
    assert eval_commands and all(f"--labels-dir {labels_dir}" in command for command in eval_commands)


def test_pipeline_uses_only_the_baseline_application_cli(tmp_path: Path) -> None:
    context = create_run_context("configs/baseline/experiment.yaml", output_root=tmp_path, run_id="run-4")
    commands = [" ".join(step.command) for step in build_pipeline_steps(context, python_executable="python", options=PipelineOptions())]
    assert commands
    assert all(command.startswith("python -m fashionrec.baseline ") for command in commands)


def test_industrial_pipeline_uses_only_the_industrial_application_cli(tmp_path: Path) -> None:
    context = create_run_context("configs/industrial/experiment.yaml", output_root=tmp_path, run_id="run-5")
    commands = [" ".join(step.command) for step in build_pipeline_steps(context, python_executable="python", options=PipelineOptions())]
    assert commands
    assert all(command.startswith("python -m fashionrec.industrial ") for command in commands)
