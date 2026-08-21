"""Command builders owned exclusively by the industrial application."""

from __future__ import annotations

from pathlib import Path

from fashionrec.shared.experiment.context import RunContext
from fashionrec.shared.runtime.contracts import PipelineOptions, PipelineStep


def command(python_executable: str, stage: str, *args: str) -> tuple[str, ...]:
    return (python_executable, "-m", "fashionrec.industrial", stage, *args)


def append(steps: list[PipelineStep], step: PipelineStep | None) -> None:
    if step is not None:
        steps.append(step)


def data_step(context: RunContext, python: str, options: PipelineOptions) -> PipelineStep | None:
    if options.skip_data_prep:
        return None
    args = ["--processed-dir", str(context.artifacts.data)]
    if options.with_filter:
        args.append("--with-filter")
    if options.build_backtest:
        args.append("--build-backtest")
    return PipelineStep("数据准备", command(python, "data", *args))


def train_step(context: RunContext, python: str, options: PipelineOptions) -> PipelineStep | None:
    if options.skip_train:
        return None
    cfg = context.config
    artifacts = context.artifacts
    return PipelineStep(
        "训练 SASRecF",
        command(
            python,
            "train",
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


def checkpoint_step(context: RunContext, python: str, options: PipelineOptions) -> PipelineStep | None:
    if options.skip_checkpoint_selection:
        return None
    cfg = context.config
    artifacts = context.artifacts
    return PipelineStep(
        "valid 用户周 MAP 选择 SASRecF checkpoint",
        command(
            python,
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
            str(artifacts.selected_checkpoint_file("sasrecf")),
            "--top-k",
            str(cfg.candidate.final_top_k),
        ),
    )


def recall_step(
    context: RunContext,
    python: str,
    options: PipelineOptions,
    split: str,
) -> PipelineStep | None:
    if options.skip_recall:
        return None
    cfg = context.config
    artifacts = context.artifacts
    return PipelineStep(
        f"SASRecF 召回 {split}",
        command(
            python,
            "recall",
            "--data-dir",
            str(artifacts.data),
            "--eval-split",
            split,
            "--top-k",
            str(cfg.candidate.sequence_top_k),
            "--model-file",
            str(artifacts.selected_checkpoint_file("sasrecf")),
            "--output-path",
            str(artifacts.recall_file("sasrecf", split)),
        ),
    )


def candidate_step(
    context: RunContext,
    python: str,
    options: PipelineOptions,
    split: str,
) -> PipelineStep | None:
    if options.skip_candidates:
        return None
    cfg = context.config
    artifacts = context.artifacts
    return PipelineStep(
        f"扩展多路候选物化 {split}",
        command(
            python,
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
            str(artifacts.recall_file("sasrecf", split)),
            "--sequence-top-k",
            str(cfg.candidate.sequence_top_k),
            "--popular-top-k",
            str(cfg.candidate.popular_top_k),
            "--category-popular-top-k",
            str(cfg.candidate.category_popular_top_k),
            "--item2item-top-k",
            str(cfg.candidate.item2item_top_k),
            "--repurchase-top-k",
            str(cfg.candidate.repurchase_top_k),
            "--style-top-k",
            str(cfg.candidate.style_top_k),
            "--content-top-k",
            str(cfg.candidate.content_top_k),
            "--union-top-k",
            str(cfg.candidate.union_top_k),
            "--max-user-history",
            str(cfg.data.max_user_history),
        ),
    )


def ranker_dataset_step(context: RunContext, python: str, options: PipelineOptions) -> PipelineStep | None:
    if options.skip_ranker:
        return None
    artifacts = context.artifacts
    return PipelineStep(
        "构建 LambdaRank 训练表（SASRecF 简单复用）",
        command(
            python,
            "ranker-dataset",
            "--experiment-config",
            str(context.config.source_path),
            "--data-dir",
            str(artifacts.data),
            "--candidate-dir",
            str(artifacts.candidates),
            "--output-dir",
            str(artifacts.ranking),
            "--diagnostics-dir",
            str(artifacts.evaluation),
            "--sequence-feature-dir",
            str(artifacts.ranking / "sasrecf_model_reuse"),
            "--articles-path",
            "data/raw/articles.csv",
            "--customers-path",
            "data/raw/customers.csv",
        ),
    )


def ranker_sequence_step(context: RunContext, python: str, options: PipelineOptions) -> PipelineStep | None:
    if options.skip_ranker or not context.config.ranking.use_sequence_features:
        return None
    artifacts = context.artifacts
    return PipelineStep(
        "复用唯一 SASRecF 生成 LambdaRank 序列证据",
        command(
            python,
            "ranker-sequence",
            "--experiment-config",
            str(context.config.source_path),
            "--data-dir",
            str(artifacts.data),
            "--model-file",
            str(artifacts.selected_checkpoint_file("sasrecf")),
            "--output-dir",
            str(artifacts.ranking / "sasrecf_model_reuse"),
        ),
    )


def weights_path(context: RunContext, options: PipelineOptions) -> Path:
    return Path(options.weights_json) if options.weights_json else context.artifacts.ranking / "best_fusion_weights.json"


def weight_step(context: RunContext, python: str, options: PipelineOptions) -> PipelineStep | None:
    if options.skip_weight_search:
        return None
    cfg = context.config
    artifacts = context.artifacts
    strict = ("--strict",) if context.strict else ()
    return PipelineStep(
        "融合权重搜索 valid",
        command(
            python,
            "weights",
            "--candidate-csv",
            str(artifacts.candidate_file("valid")),
            "--data-dir",
            str(artifacts.data),
            "--labels-dir",
            str(artifacts.data / "labels"),
            "--output-json",
            str(weights_path(context, options)),
            "--final-top-k",
            str(cfg.candidate.final_top_k),
            "--max-user-history",
            str(cfg.data.max_user_history),
            *strict,
        ),
    )


def ranker_train_step(context: RunContext, python: str, options: PipelineOptions) -> PipelineStep | None:
    if options.skip_ranker:
        return None
    cfg = context.config
    artifacts = context.artifacts
    return PipelineStep(
        "训练 LightGBM LambdaRank",
        command(
            python,
            "ranker-train",
            "--train-parquet",
            str(artifacts.ranking_table_file("train")),
            "--valid-parquet",
            str(artifacts.ranking_table_file("valid")),
            "--output-dir",
            str(artifacts.ranker_dir()),
            "--n-estimators",
            "200",
            "--seed",
            str(cfg.experiment.seed),
        ),
    )


def ranker_predict_step(
    context: RunContext,
    python: str,
    options: PipelineOptions,
    split: str,
) -> PipelineStep | None:
    if options.skip_ranker or (split == "valid" and options.skip_valid_eval) or (split == "test" and options.skip_test_eval):
        return None
    cfg = context.config
    artifacts = context.artifacts
    return PipelineStep(
        f"LambdaRank 打分 {split}",
        command(
            python,
            "ranker-predict",
            "--model-dir",
            str(artifacts.ranker_dir()),
            "--input-parquet",
            str(artifacts.ranking_table_file(split)),
            "--output-csv",
            str(artifacts.ranker_scored_file(split)),
            "--top-k",
            str(cfg.candidate.final_top_k),
        ),
    )


def evaluation_step(
    context: RunContext,
    python: str,
    options: PipelineOptions,
    split: str,
) -> PipelineStep | None:
    if (split == "valid" and options.skip_valid_eval) or (split == "test" and options.skip_test_eval):
        return None
    cfg = context.config
    artifacts = context.artifacts
    strict = ("--strict",) if context.strict else ()
    return PipelineStep(
        f"离线排序评估 {split}",
        command(
            python,
            "evaluate",
            "--eval-split",
            split,
            "--data-dir",
            str(artifacts.data),
            "--labels-dir",
            str(artifacts.data / "labels"),
            "--candidate-csv",
            str(artifacts.candidate_file(split)),
            "--ranker-scored-csv",
            str(artifacts.ranker_scored_file(split)),
            "--final-top-k",
            str(cfg.candidate.final_top_k),
            "--output-dir",
            str(artifacts.ranking),
            "--evaluation-dir",
            str(artifacts.evaluation),
            "--max-user-history",
            str(cfg.data.max_user_history),
            *strict,
            "--weights-json",
            str(weights_path(context, options)),
        ),
    )
