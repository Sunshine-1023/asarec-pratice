"""Tests for canonical IDs and candidate records."""  # 领域契约测试

from __future__ import annotations  # 延迟注解

import pytest  # 异常与近似断言

from fashionrec.domain.candidates import Candidate  # 候选结构
from fashionrec.domain.ids import canonical_item_id, canonical_user_id, submission_item_id  # ID 规范


@pytest.mark.parametrize(  # 多种输入应映射为同一商品
    ("raw", "expected"),  # 参数名
    [  # 样例
        ("0706016001", "0706016001"),  # 已补齐
        (706016001, "0706016001"),  # 整数
        ("706016001", "0706016001"),  # 未补齐字符串
        ("706016001.0", "0706016001"),  # 浮点字符串
    ],  # 样例结束
)
def test_canonical_item_id_uses_one_ten_digit_representation(raw: object, expected: str) -> None:  # 十位 ID
    assert canonical_item_id(raw) == expected  # 统一表示
    assert submission_item_id(raw) == expected  # 提交边界相同


def test_canonical_ids_reject_empty_values() -> None:  # 空 ID 尽早失败
    with pytest.raises(ValueError):  # 商品
        canonical_item_id("   ")  # 空白
    with pytest.raises(ValueError):  # 用户
        canonical_user_id("")  # 空字符串


def test_candidate_normalizes_ids_and_validates_rank() -> None:  # 候选构造即规范化
    candidate = Candidate(" user ", "706016001", "SASRecF", 1, 1, "VALID")  # 原始输入
    assert candidate.as_dict() == {  # 标准 schema
        "user_id": "user",  # 用户
        "item_id": "0706016001",  # 商品
        "channel": "sasrecf",  # 通道
        "score": 1.0,  # 分数
        "rank": 1,  # 排名
        "split": "valid",  # 划分
    }  # 字典结束
    with pytest.raises(ValueError, match="rank"):  # 非法排名
        Candidate("u", "1", "popular", 1.0, 0, "valid")  # rank=0
    train = Candidate("u", "1", "popular", 1.0, 1, "train")  # 排序训练快照
    assert train.split == "train"

