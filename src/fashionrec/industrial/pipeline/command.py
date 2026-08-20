from __future__ import annotations

from fashionrec.industrial.pipeline.orchestrator import build_pipeline_steps
from fashionrec.industrial.settings import EXPERIMENT_CONFIG
from fashionrec.shared.runtime.pipeline_runner import run_application_pipeline


def main(argv: list[str] | None = None) -> None:
    run_application_pipeline(
        argv,
        application="industrial",
        default_config=EXPERIMENT_CONFIG,
        build_steps=build_pipeline_steps,
    )
