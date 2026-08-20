"""Compatibility facade for industrial protocol validation."""

from fashionrec.industrial.data.command import REQUIRED_FLAGS as INDUSTRIAL_DATA_FLAGS
from fashionrec.industrial.data.protocol import validate_context as validate_industrial_context

__all__ = ["INDUSTRIAL_DATA_FLAGS", "validate_industrial_context"]
