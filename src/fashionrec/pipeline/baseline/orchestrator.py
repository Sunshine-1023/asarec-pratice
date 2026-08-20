"""Compatibility facade for the top-level baseline application DAG."""

from fashionrec.baseline.pipeline.orchestrator import build_pipeline_steps

__all__ = ["build_pipeline_steps"]
