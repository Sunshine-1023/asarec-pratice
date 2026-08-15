"""Candidate records shared by every recall and ranking implementation."""  # 统一候选记录

from __future__ import annotations  # 延迟注解

from dataclasses import dataclass  # 不可变候选结构

from src.domain.ids import canonical_item_id, canonical_user_id  # ID 契约


@dataclass(frozen=True, slots=True)  # 候选数量大，slots 减少内存
class Candidate:  # 单个用户-商品候选
    user_id: str  # 用户 ID
    item_id: str  # 十位商品 ID
    channel: str  # 召回通道
    score: float  # 通道原始分数
    rank: int  # 通道内排名，从 1 开始
    split: str  # valid 或 test

    def __post_init__(self) -> None:  # 构造时统一校验
        object.__setattr__(self, "user_id", canonical_user_id(self.user_id))  # 用户 ID
        object.__setattr__(self, "item_id", canonical_item_id(self.item_id))  # 商品 ID
        object.__setattr__(self, "channel", str(self.channel).strip().lower())  # 通道名
        object.__setattr__(self, "split", str(self.split).strip().lower())  # 划分名
        object.__setattr__(self, "score", float(self.score))  # 分数类型
        object.__setattr__(self, "rank", int(self.rank))  # 排名类型
        if not self.channel:  # 通道不能为空
            raise ValueError("candidate channel must not be empty")  # 抛错
        if self.rank < 1:  # 排名从 1 开始
            raise ValueError(f"candidate rank must be >= 1, got {self.rank}")  # 抛错
        if self.split not in {"valid", "test"}:  # 当前正式候选只允许两类
            raise ValueError(f"candidate split must be valid or test, got {self.split!r}")  # 抛错

    def as_dict(self) -> dict[str, object]:  # 稳定的表格/JSON schema
        return {  # 返回字段
            "user_id": self.user_id,  # 用户
            "item_id": self.item_id,  # 商品
            "channel": self.channel,  # 通道
            "score": self.score,  # 分数
            "rank": self.rank,  # 排名
            "split": self.split,  # 划分
        }  # 字典结束

