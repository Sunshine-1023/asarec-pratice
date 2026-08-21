"""Complete next-basket/PIT/multi-recall/LambdaRank industrial DAG."""

from __future__ import annotations

from fashionrec.experiment.context import RunContext
from fashionrec.industrial.data.protocol import validate_context
from fashionrec.industrial.pipeline.stages import (
    append,
    candidate_step,
    checkpoint_step,
    data_step,
    evaluation_step,
    ranker_dataset_step,
    ranker_predict_step,
    ranker_sequence_step,
    ranker_train_step,
    recall_step,
    train_step,
    weight_step,
)
from fashionrec.shared.runtime.contracts import PipelineOptions, PipelineStep


def build_pipeline_steps(
    context: RunContext,
    *,
    python_executable: str,
    options: PipelineOptions,
) -> list[PipelineStep]:
    validate_context(context)
    steps: list[PipelineStep] = []
    append(steps, data_step(context, python_executable, options))
    append(steps, train_step(context, python_executable, options))
    append(steps, checkpoint_step(context, python_executable, options))
    for split in ("valid", "test"):
        append(steps, recall_step(context, python_executable, options, split))
        append(steps, candidate_step(context, python_executable, options, split))
    append(steps, ranker_sequence_step(context, python_executable, options))
    append(steps, ranker_dataset_step(context, python_executable, options))
    append(steps, weight_step(context, python_executable, options))
    append(steps, ranker_train_step(context, python_executable, options))
    for split in ("valid", "test"):
        append(steps, ranker_predict_step(context, python_executable, options, split))
    for split in ("valid", "test"):
        append(steps, evaluation_step(context, python_executable, options, split))
    return steps
