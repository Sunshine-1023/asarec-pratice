"""Offline evaluation utilities."""  # 离线评估工具模块

from .metrics import hit_at_k, map_at_k, ndcg_at_k, recall_at_k  # 统一指标
from .offline_eval import evaluate_fusion  # 导入融合评估函数

__all__ = ["evaluate_fusion", "hit_at_k", "map_at_k", "ndcg_at_k", "recall_at_k"]  # 定义模块公开接口
