"""Shared domain contracts used across data, recall, ranking, and evaluation."""  # 跨层共享的领域契约

from .candidates import Candidate  # 统一候选记录
from .ids import canonical_item_id, canonical_user_id, submission_item_id  # 统一 ID 规范

__all__ = ["Candidate", "canonical_item_id", "canonical_user_id", "submission_item_id"]  # 公开接口

