"""Pure pipeline-plan construction from one resolved run context."""  # 编排层只组装阶段与产物依赖

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.experiment.context import RunContext


@dataclass(frozen=True, slots=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    with_filter: bool = False
    skip_data_prep: bool = False
    skip_train: bool = False
    skip_checkpoint_selection: bool = False
    skip_recall: bool = False
    skip_candidates: bool = False
    skip_weight_search: bool = False
    skip_valid_eval: bool = False
    skip_test_eval: bool = False
    weights_json: str | None = None  # 跳过搜索时可复用显式指定权重


def build_pipeline_steps(
    context: RunContext,
    *,
    python_executable: str,
    options: PipelineOptions,
) -> list[PipelineStep]:
    """Resolve all commands against one config and run-scoped artifact tree."""
    cfg = context.config
    artifacts = context.artifacts
    config_path = str(cfg.source_path)
    steps: list[PipelineStep] = []
    strict_args = ("--strict",) if context.strict else ()  # 兼容模式可显式关闭严格检查

    if not options.skip_data_prep:
        command = [python_executable, "run_data_prep.py", "--experiment-config", config_path]
        if options.with_filter:
            command.append("--with-filter")
        steps.append(PipelineStep("数据准备", tuple(command)))

    if not options.skip_train:
        steps.append(
            PipelineStep(
                "训练 SASRecF",
                (
                    python_executable,
                    "run_sasrecf.py",
                    "--skip-preprocess",
                    "--seed",
                    str(cfg.experiment.seed),
                    "--checkpoint-dir",
                    str(artifacts.checkpoints / "sasrecf"),
                    "--checkpoint-shortlist-size",
                    str(cfg.model_selection.checkpoint_shortlist_size),
                    "--report-path",
                    str(artifacts.evaluation / "sasrecf_train_results.json"),
                ),
            )
        )

    selected_checkpoint = artifacts.selected_checkpoint_file("sasrecf")
    if not options.skip_checkpoint_selection:
        steps.append(
            PipelineStep(
                "valid 用户周 MAP 选择 SASRecF checkpoint",
                (
                    python_executable,
                    "run_select_checkpoint.py",
                    "--checkpoint-dir",
                    str(artifacts.checkpoints / "sasrecf"),
                    "--recall-dir",
                    str(artifacts.recall / "checkpoint_selection"),
                    "--output-json",
                    str(artifacts.checkpoint_selection_file("sasrecf")),
                    "--selected-model-path",
                    str(selected_checkpoint),
                    "--top-k",
                    str(cfg.candidate.final_top_k),
                ),
            )
        )

    for split in ("valid", "test"):
        sequence_path = artifacts.recall_file("sasrecf", split)
        candidate_path = artifacts.candidate_file(split)
        if not options.skip_recall:
            steps.append(
                PipelineStep(
                    f"SASRecF 召回 {split}",
                    (
                        python_executable,
                        "run_sasrecf_recall.py",
                        "--eval-split",
                        split,
                        "--top-k",
                        str(cfg.candidate.sequence_top_k),
                        "--model-file",
                        str(selected_checkpoint),
                        "--output-path",
                        str(sequence_path),
                    ),
                )
            )
        if not options.skip_candidates:
            steps.append(
                PipelineStep(
                    f"四路候选物化 {split}",
                    (
                        python_executable,
                        "run_rule_recall.py",
                        "--eval-split",
                        split,
                        "--output-dir",
                        str(artifacts.recall),
                        "--candidate-output-dir",
                        str(artifacts.candidates),
                        "--sequence-recall-csv",
                        str(sequence_path),
                        "--sequence-top-k",
                        str(cfg.candidate.sequence_top_k),
                        "--popular-top-k",
                        str(cfg.candidate.popular_top_k),
                        "--category-popular-top-k",
                        str(cfg.candidate.category_popular_top_k),
                        "--item2item-top-k",
                        str(cfg.candidate.item2item_top_k),
                        "--union-top-k",
                        str(cfg.candidate.union_top_k),
                    ),
                )
            )

    weights_path = Path(options.weights_json) if options.weights_json else artifacts.ranking / "best_fusion_weights.json"
    if not options.skip_weight_search:
        steps.append(
            PipelineStep(
                "融合权重搜索 valid",
                (
                    python_executable,
                    "run_fusion_weight_search.py",
                    "--candidate-csv",
                    str(artifacts.candidate_file("valid")),
                    "--output-json",
                    str(weights_path),
                    "--final-top-k",
                    str(cfg.candidate.final_top_k),
                    *strict_args,
                ),
            )
        )

    for split, skip in (("valid", options.skip_valid_eval), ("test", options.skip_test_eval)):
        if skip:
            continue
        command = [
            python_executable,
            "run_offline_eval.py",
            "--eval-split",
            split,
            "--candidate-csv",
            str(artifacts.candidate_file(split)),
            "--final-top-k",
            str(cfg.candidate.final_top_k),
            "--output-dir",
            str(artifacts.ranking),
            "--evaluation-dir",
            str(artifacts.evaluation),
            *strict_args,
        ]
        if split == "test":
            command.extend(["--weights-json", str(weights_path)])
        steps.append(PipelineStep(f"离线排序评估 {split}", tuple(command)))
    return steps
