"""Compatibility facade for the top-level industrial application DAG."""

from fashionrec.industrial.pipeline.orchestrator import build_pipeline_steps

__all__ = ["build_pipeline_steps"]
