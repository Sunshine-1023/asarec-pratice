"""Compatibility facade for the profile-specific pipeline orchestrators."""

from fashionrec.shared.runtime.contracts import PipelineOptions, PipelineStep
from fashionrec.pipeline.registry import build_pipeline_steps

__all__ = ["PipelineOptions", "PipelineStep", "build_pipeline_steps"]
