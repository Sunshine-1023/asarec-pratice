"""Offline evaluation utilities."""  # 离线评估工具模块

from .metrics import hit_at_k, map_at_k, ndcg_at_k, recall_at_k  # 统一指标

__all__ = ["evaluate_fusion", "hit_at_k", "map_at_k", "ndcg_at_k", "recall_at_k"]  # 定义模块公开接口


def __getattr__(name: str):  # 延迟导入可执行评估模块，避免 python -m 重复加载
    if name == "evaluate_fusion":  # 若请求融合评估函数
        from .offline_eval import evaluate_fusion  # 仅在实际使用时导入

        return evaluate_fusion  # 返回融合评估函数
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # 未知属性直接报错
