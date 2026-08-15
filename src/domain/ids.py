"""Canonical user and item identifier helpers."""  # 用户与商品 ID 的唯一规范

from __future__ import annotations  # 延迟注解

import re  # 处理被解析为浮点字符串的数字 ID
from collections.abc import Iterable  # 批量规范化类型


ARTICLE_ID_WIDTH = 10  # H&M article_id 的正式宽度
_FLOAT_SUFFIX = re.compile(r"\.0+$")  # CSV 数字列可能产生的 .0 后缀


def canonical_item_id(value: object) -> str:  # 内部统一使用十位商品 ID
    """Return one stable internal H&M article ID representation."""  # 返回稳定的内部商品 ID
    text = str(value).strip()  # 转字符串并去空白
    if not text:  # 空 ID 非法
        raise ValueError("item_id must not be empty")  # 尽早暴露坏数据
    numeric = _FLOAT_SUFFIX.sub("", text)  # 兼容 123.0
    if numeric.isdigit():  # H&M 商品 ID 为纯数字
        if len(numeric) > ARTICLE_ID_WIDTH:  # 超宽数字不能静默截断
            raise ValueError(f"numeric item_id exceeds {ARTICLE_ID_WIDTH} digits: {text!r}")  # 抛错
        return numeric.zfill(ARTICLE_ID_WIDTH)  # 统一补齐十位
    return text  # 非数字测试/扩展 ID 保持原样


def submission_item_id(value: object) -> str:  # Kaggle 输出格式
    return canonical_item_id(value)  # 内部格式已经是十位正式格式


def canonical_user_id(value: object) -> str:  # 用户 ID 统一字符串
    text = str(value).strip()  # 去除 CSV 空白
    if not text:  # 空用户非法
        raise ValueError("user_id must not be empty")  # 尽早暴露坏数据
    return text  # H&M customer_id 为哈希字符串，不做数值转换


def canonical_item_ids(values: Iterable[object]) -> list[str]:  # 批量商品 ID
    return [canonical_item_id(value) for value in values]  # 保序规范化

