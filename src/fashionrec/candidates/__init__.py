"""Compatibility alias for candidate union now owned by Industrial."""

from importlib import import_module
import sys


union = import_module("fashionrec.industrial.recall.union")
sys.modules[f"{__name__}.union"] = union

from fashionrec.industrial.recall.union import DEFAULT_UNION_TOP_K, UNION_SCHEMA_VERSION, UnionEvidenceRow, build_union_evidence, select_union_items, union_candidates, write_union_evidence_csv

__all__ = [
    "DEFAULT_UNION_TOP_K",
    "UNION_SCHEMA_VERSION",
    "UnionEvidenceRow",
    "build_union_evidence",
    "select_union_items",
    "union_candidates",
    "write_union_evidence_csv",
]
