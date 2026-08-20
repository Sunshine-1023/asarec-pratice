"""Stable contracts shared by pipeline profiles and the command runner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipelineStep:
    """One executable stage in a resolved pipeline plan."""

    name: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Stage-selection options that apply consistently to every profile."""

    with_filter: bool = False
    skip_data_prep: bool = False
    skip_train: bool = False
    skip_checkpoint_selection: bool = False
    skip_recall: bool = False
    skip_candidates: bool = False
    skip_weight_search: bool = False
    skip_ranker: bool = False
    skip_valid_eval: bool = False
    skip_test_eval: bool = False
    weights_json: str | None = None
    build_backtest: bool = False
