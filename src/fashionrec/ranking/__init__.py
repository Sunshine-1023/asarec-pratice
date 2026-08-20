"""Compatibility aliases for ranking implementations now owned by Industrial."""

from importlib import import_module
import sys


_ALIASES = {
    "base": "fashionrec.industrial.ranking.base",
    "dataset": "fashionrec.industrial.ranking.dataset",
    "features": "fashionrec.industrial.ranking.features",
    "fusion": "fashionrec.industrial.ranking.fusion",
    "predict": "fashionrec.industrial.ranking.predict",
    "train": "fashionrec.industrial.ranking.train",
    "weighted_rrf": "fashionrec.industrial.ranking.weighted_rrf",
}

for _name, _target in _ALIASES.items():
    _module = import_module(_target)
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

from fashionrec.industrial.models.lambdarank.predict import LightGBMRanker, load_ranker, rank_feature_frame
from fashionrec.industrial.models.lambdarank.train import RankerArtifact, train_lambdarank
from fashionrec.industrial.ranking.base import RankedItem, Ranker
from fashionrec.industrial.ranking.dataset import RankingDataset, build_ranking_dataset, write_ranking_dataset
from fashionrec.industrial.ranking.features import build_ranking_features, lambda_rank_group_sizes
from fashionrec.industrial.ranking.weighted_rrf import WeightedRRFRanker

__all__ = [name for name in globals() if not name.startswith("_") and name not in {"import_module", "sys"}]
