"""Application-neutral recall boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


RecallResult = list[tuple[str, float]]


@runtime_checkable
class RecallChannel(Protocol):
    """Minimum contract implemented by every recall channel."""

    name: str

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        """Return at most ``top_k`` ranked items for one user."""
