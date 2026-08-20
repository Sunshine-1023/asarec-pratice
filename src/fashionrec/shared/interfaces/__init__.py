"""Stable application-neutral recommendation interfaces."""

from fashionrec.shared.interfaces.ranking import RankedItem, Ranker
from fashionrec.shared.interfaces.recall import RecallChannel, RecallResult

__all__ = ["RankedItem", "Ranker", "RecallChannel", "RecallResult"]
