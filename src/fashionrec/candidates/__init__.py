"""Candidate materialization and union utilities."""

from fashionrec.candidates.union import (
    DEFAULT_UNION_TOP_K,
    UNION_SCHEMA_VERSION,
    UnionEvidenceRow,
    build_union_evidence,
    select_union_items,
    union_candidates,
    write_union_evidence_csv,
)

__all__ = [
    "DEFAULT_UNION_TOP_K",
    "UNION_SCHEMA_VERSION",
    "UnionEvidenceRow",
    "build_union_evidence",
    "select_union_items",
    "union_candidates",
    "write_union_evidence_csv",
]
