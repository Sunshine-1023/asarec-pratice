"""Compatibility aliases for recall implementations now owned by Industrial."""

from importlib import import_module
import sys


_ALIASES = {
    "base": "fashionrec.industrial.recall.base",
    "category_popular": "fashionrec.industrial.recall.category_popular",
    "content": "fashionrec.industrial.recall.content",
    "generator": "fashionrec.industrial.recall.generator",
    "item2item": "fashionrec.industrial.recall.item2item",
    "popular": "fashionrec.industrial.recall.popular",
    "registry": "fashionrec.industrial.recall.channel_registry",
    "repurchase": "fashionrec.industrial.recall.repurchase",
    "rule_recall_export": "fashionrec.industrial.recall.service",
    "style": "fashionrec.industrial.recall.style",
    "window_scores": "fashionrec.industrial.recall.window_scores",
}

for _name, _target in _ALIASES.items():
    _module = import_module(_target)
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

from fashionrec.industrial.recall.category_popular import CATEGORY_POPULAR_RECALL_TOP_K, build_category_popular_index, recall_category_popular
from fashionrec.industrial.recall.content import CONTENT_RECALL_TOP_K, ContentIndex, build_content_index, recall_content
from fashionrec.industrial.recall.item2item import DEFAULT_SIMILARITY_MODE, ITEM2ITEM_RECALL_TOP_K, ITEM2ITEM_SCHEMA_VERSION, SIMILARITY_MODES, build_item2item_index, recall_item2item
from fashionrec.industrial.recall.popular import POPULAR_RECALL_TOP_K, PopularIndex, build_popular_index, build_user_cohort_lookup, recall_popular
from fashionrec.industrial.recall.repurchase import REPURCHASE_RECALL_TOP_K, RepurchaseIndex, build_repurchase_index, recall_repurchase
from fashionrec.industrial.recall.style import STYLE_RECALL_TOP_K, StyleIndex, build_style_index, recall_style
__all__ = [name for name in globals() if not name.startswith("_") and name not in {"import_module", "sys"}]


def __getattr__(name: str):
    if name == "export_sasrec_recall":
        from fashionrec.industrial.models.sasrecf.recall_service import export_sasrec_recall

        return export_sasrec_recall
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
