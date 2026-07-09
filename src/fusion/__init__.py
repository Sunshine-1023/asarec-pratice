"""Fusion helpers for multi-channel recall."""  # 多通道召回融合辅助模块

from .weighted_fusion import (  # 从加权融合模块导入
    ACTIVITY_WEIGHTS,  # 活跃度权重模板
    build_user_history,  # 构建用户历史
    classify_activity_tier,  # 划分活跃度
    fuse_candidates,  # 融合候选集
    get_channel_weights_for_user,  # 按用户取通道权重
    infer_sequence_channel,  # 推断序列通道名
    load_channel_recall_csv,  # 加载通道召回 CSV
)  # 导入结束

__all__ = [  # 定义模块公开接口
    "ACTIVITY_WEIGHTS",  # 活跃度权重模板
    "build_user_history",  # 构建用户历史
    "classify_activity_tier",  # 划分活跃度
    "fuse_candidates",  # 融合候选集
    "get_channel_weights_for_user",  # 按用户取通道权重
    "infer_sequence_channel",  # 推断序列通道名
    "load_channel_recall_csv",  # 加载通道召回 CSV
]  # 公开接口列表结束
