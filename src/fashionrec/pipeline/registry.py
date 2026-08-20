"""Route a resolved run context to exactly one workflow implementation."""

from __future__ import annotations

from collections.abc import Callable

from fashionrec.experiment.context import RunContext
from fashionrec.baseline.pipeline.orchestrator import build_pipeline_steps as build_baseline_steps
from fashionrec.shared.runtime.contracts import PipelineOptions, PipelineStep
from fashionrec.industrial.pipeline.orchestrator import build_pipeline_steps as build_industrial_steps


PipelineBuilder = Callable[..., list[PipelineStep]]

PIPELINE_BUILDERS: dict[str, PipelineBuilder] = {
    "baseline": build_baseline_steps,
    "industrial": build_industrial_steps,
}


def build_pipeline_steps(
    context: RunContext,
    *,
    python_executable: str,
    options: PipelineOptions,
) -> list[PipelineStep]:
    """Dispatch only from the immutable profile namespace on the run context."""

    profile = context.artifacts.profile
    try:
        builder = PIPELINE_BUILDERS[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported pipeline profile: {profile}") from exc
    return builder(context, python_executable=python_executable, options=options)
