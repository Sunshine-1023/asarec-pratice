"""Recall-channel boundary shared by rule and model-based retrievers."""  # 召回通道统一接口

from __future__ import annotations

from typing import Protocol, runtime_checkable


RecallResult = list[tuple[str, float]]  # 通道内已排序的 (item_id, score)


@runtime_checkable
class RecallChannel(Protocol):  # 所有召回通道的最小契约
    name: str  # 稳定通道名

    def recall(self, user_id: str, history: list[str], top_k: int) -> RecallResult:
        """Return at most top_k ranked candidates for one user."""

