"""Complete baseline DAG owned by the baseline application."""

from __future__ import annotations

from fashionrec.baseline.pipeline.stages import (
    append,
    candidate_step,
    checkpoint_step,
    data_step,
    evaluation_step,
    recall_step,
    train_step,
    weight_step,
)
from fashionrec.experiment.context import RunContext
from fashionrec.shared.runtime.contracts import PipelineOptions, PipelineStep


def build_pipeline_steps(
    context: RunContext,
    *,
    python_executable: str,
    options: PipelineOptions,
) -> list[PipelineStep]:
    if context.artifacts.profile != "baseline":
        raise ValueError("baseline application requires a baseline run context")
    steps: list[PipelineStep] = []
    append(steps, data_step(context, python_executable, options))
    append(steps, train_step(context, python_executable, options))
    append(steps, checkpoint_step(context, python_executable, options))
    for split in ("valid", "test"):
        append(steps, recall_step(context, python_executable, options, split))
        append(steps, candidate_step(context, python_executable, options, split))
    append(steps, weight_step(context, python_executable, options))
    for split in ("valid", "test"):
        append(steps, evaluation_step(context, python_executable, options, split))
    return steps
