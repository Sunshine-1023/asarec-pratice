"""Pure pipeline-plan construction from one resolved run context."""  # 编排层只组装阶段与产物依赖

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fashionrec.experiment.context import RunContext


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
    skip_ranker: bool = False
    skip_valid_eval: bool = False
    skip_test_eval: bool = False
    weights_json: str | None = None  # 跳过搜索时可复用显式指定权重
    build_backtest: bool = False  # 透传 data --build-backtest；默认关，避免训三遍


def _cli_command(python_executable: str, command: str, *args: str) -> tuple[str, ...]:  # 组装统一 CLI 命令
    return (python_executable, "-m", "fashionrec", command, *args)  # 所有阶段只依赖公开命令协议


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
        command = list(
            _cli_command(
                python_executable,
                "data",
                "--experiment-config",
                config_path,
                "--processed-dir",
                str(artifacts.data),
            )
        )
        if options.with_filter:
            command.append("--with-filter")
        if options.build_backtest:
            command.append("--build-backtest")
        steps.append(PipelineStep("数据准备", tuple(command)))

    if not options.skip_train:
        steps.append(
            PipelineStep(
                "训练 SASRecF",
                _cli_command(
                    python_executable,
                    "train",
                    "--config",
                    "configs/sasrecf.yaml",
                    "--data-dir",
                    str(artifacts.data),
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
                _cli_command(
                    python_executable,
                    "select-checkpoint",
                    "--checkpoint-dir",
                    str(artifacts.checkpoints / "sasrecf"),
                    "--data-dir",
                    str(artifacts.data),
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
        if not options.skip_recall:
            steps.append(
                PipelineStep(
                    f"SASRecF 召回 {split}",
                    _cli_command(
                        python_executable,
                        "recall",
                        "--config",
                        "configs/sasrecf.yaml",
                        "--data-dir",
                        str(artifacts.data),
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
                    _cli_command(
                        python_executable,
                        "candidates",
                        "--eval-split",
                        split,
                        "--data-dir",
                        str(artifacts.data),
                        "--articles-path",
                        "data/raw/articles.csv",
                        "--customers-path",
                        "data/raw/customers.csv",
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
                        "--max-user-history",
                        str(cfg.data.max_user_history),
                    ),
                )
            )

    weights_path = Path(options.weights_json) if options.weights_json else artifacts.ranking / "best_fusion_weights.json"
    if not options.skip_weight_search:
        steps.append(
            PipelineStep(
                "融合权重搜索 valid",
                _cli_command(
                    python_executable,
                    "weights",
                    "--candidate-csv",
                    str(artifacts.candidate_file("valid")),
                    "--data-dir",
                    str(artifacts.data),
                    "--output-json",
                    str(weights_path),
                    "--final-top-k",
                    str(cfg.candidate.final_top_k),
                    "--max-user-history",
                    str(cfg.data.max_user_history),
                    *strict_args,
                ),
            )
        )

    if cfg.ranking.enabled and not options.skip_ranker:
        ranker_dir = artifacts.ranker_dir()
        steps.append(
            PipelineStep(
                "训练 LightGBM LambdaRank",
                _cli_command(
                    python_executable,
                    "ranker-train",
                    "--train-parquet",
                    str(artifacts.ranking_table_file("train")),
                    "--valid-parquet",
                    str(artifacts.ranking_table_file("valid")),
                    "--output-dir",
                    str(ranker_dir),
                    "--n-estimators",
                    "200",
                    "--seed",
                    str(cfg.experiment.seed),
                ),
            )
        )
        for split, skip in (("valid", options.skip_valid_eval), ("test", options.skip_test_eval)):
            if skip:
                continue
            steps.append(
                PipelineStep(
                    f"LambdaRank 打分 {split}",
                    _cli_command(
                        python_executable,
                        "ranker-predict",
                        "--model-dir",
                        str(ranker_dir),
                        "--input-parquet",
                        str(artifacts.ranking_table_file(split)),
                        "--output-csv",
                        str(artifacts.ranker_scored_file(split)),
                        "--top-k",
                        str(cfg.candidate.final_top_k),
                    ),
                )
            )

    for split, skip in (("valid", options.skip_valid_eval), ("test", options.skip_test_eval)):
        if skip:
            continue
        command = list(_cli_command(
            python_executable,
            "evaluate",
            "--eval-split",
            split,
            "--data-dir",
            str(artifacts.data),
            "--candidate-csv",
            str(artifacts.candidate_file(split)),
            "--final-top-k",
            str(cfg.candidate.final_top_k),
            "--output-dir",
            str(artifacts.ranking),
            "--evaluation-dir",
            str(artifacts.evaluation),
            "--max-user-history",
            str(cfg.data.max_user_history),
            *strict_args,
        ))
        command.extend(["--weights-json", str(weights_path)])
        if cfg.ranking.enabled and not options.skip_ranker:
            command.extend(["--ranker-scored-csv", str(artifacts.ranker_scored_file(split))])
        steps.append(PipelineStep(f"离线排序评估 {split}", tuple(command)))
    return steps
