"""Compatibility facade for pipeline contracts moved to shared runtime."""

from fashionrec.shared.runtime.contracts import PipelineOptions, PipelineStep

__all__ = ["PipelineOptions", "PipelineStep"]
