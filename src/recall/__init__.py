"""Recall modules for multi-channel recommendation."""  # 多通道推荐的召回模块

from .category_popular import (  # 类别热门召回
    CATEGORY_POPULAR_RECALL_TOP_K,
    build_category_popular_index,
    recall_category_popular,
)
from .item2item import ITEM2ITEM_RECALL_TOP_K, build_item2item_index, recall_item2item  # item2item 共现召回
from .itemcf import build_itemcf_index, recall_itemcf  # 导入 ItemCF 索引构建与召回函数
from .popular import POPULAR_RECALL_TOP_K, build_popular_index, recall_popular  # 导入热门召回索引构建与召回函数

__all__ = [  # 定义模块公开接口
    "POPULAR_RECALL_TOP_K",  # 热门召回 Top-K 默认值
    "CATEGORY_POPULAR_RECALL_TOP_K",  # 类别热门召回 Top-K 默认值
    "ITEM2ITEM_RECALL_TOP_K",  # item2item 召回 Top-K 默认值
    "build_popular_index",  # 热门召回索引构建
    "recall_popular",  # 热门召回
    "build_category_popular_index",  # 类别热门索引构建
    "recall_category_popular",  # 类别热门召回
    "build_item2item_index",  # item2item 索引构建
    "recall_item2item",  # item2item 召回
    "build_itemcf_index",  # ItemCF 索引构建
    "recall_itemcf",  # ItemCF 召回
    "export_sasrec_recall",  # SASRec 召回导出
]  # 公开接口列表结束


def __getattr__(name: str):  # 延迟导入依赖 RecBole 的模块
    if name == "export_sasrec_recall":
        from .sasrec_recall import export_sasrec_recall

        return export_sasrec_recall
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
