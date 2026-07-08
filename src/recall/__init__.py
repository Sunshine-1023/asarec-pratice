"""Recall modules for multi-channel recommendation."""  # 多通道推荐的召回模块

from .itemcf import build_itemcf_index, recall_itemcf  # 导入 ItemCF 索引构建与召回函数
from .popular import build_popular_index, recall_popular  # 导入热门召回索引构建与召回函数
from .sasrec_recall import export_sasrec_recall  # 导入 SASRec 召回导出函数

__all__ = [  # 定义模块公开接口
    "build_popular_index",  # 热门召回索引构建
    "recall_popular",  # 热门召回
    "build_itemcf_index",  # ItemCF 索引构建
    "recall_itemcf",  # ItemCF 召回
    "export_sasrec_recall",  # SASRec 召回导出
]  # 公开接口列表结束
