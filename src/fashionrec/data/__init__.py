"""Compatibility aliases for data implementations now owned by Industrial."""

from importlib import import_module
import sys


_ALIASES = {
    "backtest": "fashionrec.industrial.data.backtest",
    "build_baskets": "fashionrec.industrial.data.build_baskets",
    "build_events": "fashionrec.industrial.data.build_events",
    "build_item_features": "fashionrec.industrial.data.build_item_features",
    "build_sequences": "fashionrec.industrial.data.build_sequences",
    "command": "fashionrec.industrial.data.service",
    "cross_features": "fashionrec.industrial.data.cross_features",
    "customer_features": "fashionrec.industrial.data.customer_features",
    "filter": "fashionrec.industrial.data.filter",
    "item_features": "fashionrec.industrial.data.item_features",
    "labels": "fashionrec.industrial.data.labels",
    "manifest": "fashionrec.industrial.data.manifest",
    "paths": "fashionrec.industrial.data.paths",
    "preprocess": "fashionrec.industrial.data.preprocess",
    "profile": "fashionrec.industrial.data.profile",
    "snapshots": "fashionrec.industrial.data.snapshots",
    "split": "fashionrec.industrial.data.split",
    "time": "fashionrec.industrial.data.time",
    "user_features": "fashionrec.industrial.data.user_features",
}

for _name, _target in _ALIASES.items():
    _module = import_module(_target)
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

__all__ = sorted(_ALIASES)
