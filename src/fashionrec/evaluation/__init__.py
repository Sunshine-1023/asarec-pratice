"""Compatibility aliases for evaluation implementations now owned by Industrial."""

from importlib import import_module
import sys


_ALIASES = {
    "baseline_command": "fashionrec.industrial.evaluation.baseline_command",
    "candidate_diagnostics": "fashionrec.industrial.evaluation.candidate_diagnostics",
    "coverage_metrics": "fashionrec.industrial.evaluation.coverage_metrics",
    "experiment_report": "fashionrec.industrial.evaluation.experiment_report",
    "offline_eval": "fashionrec.industrial.evaluation.offline_eval",
    "weight_search": "fashionrec.industrial.evaluation.weight_search",
}

for _name, _target in _ALIASES.items():
    _module = import_module(_target)
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

from fashionrec.industrial.evaluation.offline_eval import evaluate_fusion
from fashionrec.shared.metrics.ranking import hit_at_k, map_at_k, ndcg_at_k, recall_at_k

__all__ = ["evaluate_fusion", "hit_at_k", "map_at_k", "ndcg_at_k", "recall_at_k"]
