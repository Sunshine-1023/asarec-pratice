"""Fusion helpers for multi-channel recall."""  # 多通道召回融合辅助模块

from .weighted_fusion import (  # 从加权融合模块导入
    build_user_history,  # 构建用户历史
    fuse_candidates,  # 融合候选集
    load_channel_recall_csv,  # 加载通道召回 CSV
)  # 导入结束

__all__ = [  # 定义模块公开接口
    "build_user_history",  # 构建用户历史
    "fuse_candidates",  # 融合候选集
    "load_channel_recall_csv",  # 加载通道召回 CSV
]  # 公开接口列表结束
