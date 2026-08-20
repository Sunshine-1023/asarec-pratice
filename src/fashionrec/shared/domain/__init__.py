"""Canonical identifiers and candidate records."""

from fashionrec.shared.domain.candidates import Candidate
from fashionrec.shared.domain.ids import (
    ARTICLE_ID_WIDTH,
    canonical_item_id,
    canonical_item_ids,
    canonical_user_id,
    submission_item_id,
)

__all__ = [
    "ARTICLE_ID_WIDTH",
    "Candidate",
    "canonical_item_id",
    "canonical_item_ids",
    "canonical_user_id",
    "submission_item_id",
]
