"""Recall modules for multi-channel recommendation."""  # 多通道推荐的召回模块

from .category_popular import (  # 从类别热门召回模块导入
    CATEGORY_POPULAR_RECALL_TOP_K,  # 类别热门召回 Top-K 默认值
    build_category_popular_index,  # 构建类别热门索引
    recall_category_popular,  # 类别热门召回函数
)  # 类别热门召回导入结束
from .item2item import (  # item2item 共现召回
    DEFAULT_SIMILARITY_MODE,
    ITEM2ITEM_RECALL_TOP_K,
    ITEM2ITEM_SCHEMA_VERSION,
    SIMILARITY_MODES,
    build_item2item_index,
    recall_item2item,
)  # 导入结束
from .content import CONTENT_RECALL_TOP_K, ContentIndex, build_content_index, recall_content  # 内容召回
from .repurchase import REPURCHASE_RECALL_TOP_K, RepurchaseIndex, build_repurchase_index, recall_repurchase  # 复购
from .style import STYLE_RECALL_TOP_K, StyleIndex, build_style_index, recall_style  # 款式
from .popular import (  # 热门召回
    POPULAR_RECALL_TOP_K,
    PopularIndex,
    build_popular_index,
    build_user_cohort_lookup,
    recall_popular,
)  # 导入结束

__all__ = [  # 定义模块公开接口
    "POPULAR_RECALL_TOP_K",  # 热门召回 Top-K 默认值
    "CATEGORY_POPULAR_RECALL_TOP_K",  # 类别热门召回 Top-K 默认值
    "ITEM2ITEM_RECALL_TOP_K",  # item2item 召回 Top-K 默认值
    "ITEM2ITEM_SCHEMA_VERSION",  # 索引语义
    "DEFAULT_SIMILARITY_MODE",  # 默认相似度变体
    "SIMILARITY_MODES",  # 全部变体
    "build_popular_index",  # 热门召回索引构建
    "PopularIndex",  # 热门索引结构
    "build_user_cohort_lookup",  # 冷启动 cohort 查表
    "recall_popular",  # 热门召回
    "build_category_popular_index",  # 类别热门索引构建
    "recall_category_popular",  # 类别热门召回
    "build_item2item_index",  # item2item 索引构建
    "recall_item2item",  # item2item 召回
    "REPURCHASE_RECALL_TOP_K",  # 复购 Top-K
    "RepurchaseIndex",
    "build_repurchase_index",
    "recall_repurchase",
    "STYLE_RECALL_TOP_K",  # 款式 Top-K
    "StyleIndex",
    "build_style_index",
    "recall_style",
    "CONTENT_RECALL_TOP_K",  # 内容 Top-K
    "ContentIndex",
    "build_content_index",
    "recall_content",
    "export_sasrec_recall",  # SASRec 召回导出
]  # 公开接口列表结束


def __getattr__(name: str):  # 延迟导入依赖 RecBole 的模块
    if name == "export_sasrec_recall":  # 若请求 SASRec 召回导出函数
        from .sasrec_recall import export_sasrec_recall  # 延迟导入避免启动时加载 RecBole

        return export_sasrec_recall  # 返回导出函数
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # 未知属性则抛出异常
