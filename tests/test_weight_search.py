"""Regression tests for the unified weight-search execution path."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from fashionrec.evaluation import weight_search
from fashionrec.evaluation.offline_eval import FusionEvalContext
from fashionrec.ranking.fusion import ACTIVITY_WEIGHTS


@pytest.mark.parametrize(
    ("mode", "expected_modes"),
    [("both", [False, True]), ("false", [False]), ("true", [True])],
)
def test_run_weight_search_uses_one_path_for_all_exclude_seen_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    expected_modes: list[bool],
) -> None:
    context_kwargs: dict[str, object] = {}
    searched_modes: list[bool] = []

    def fake_context(**kwargs: object) -> FusionEvalContext:
        context_kwargs.update(kwargs)
        return FusionEvalContext(targets={}, users=[], sequence_channel="sasrecf", final_top_k=12)

    def fake_search(
        context: FusionEvalContext,
        *,
        step: float,
        exclude_seen: bool,
        max_passes: int,
        verbose: bool,
    ) -> tuple[dict, float]:
        searched_modes.append(exclude_seen)
        return copy.deepcopy(ACTIVITY_WEIGHTS), float(exclude_seen)

    monkeypatch.setattr(weight_search, "build_fusion_eval_context", fake_context)
    monkeypatch.setattr(weight_search, "search_best_weights", fake_search)

    payload = weight_search.run_weight_search(
        output_json=tmp_path / f"{mode}.json",
        exclude_seen_mode=mode,  # type: ignore[arg-type]
        max_user_history=37,
        verbose=False,
    )

    assert context_kwargs["max_user_history"] == 37
    assert searched_modes == expected_modes
    assert payload["selected_mode"] == f"exclude_seen={str(expected_modes[-1]).lower()}"
    assert list(payload["compared_exclude_seen"]) == [
        f"exclude_seen={str(value).lower()}" for value in expected_modes
    ]


def test_run_weight_search_rejects_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        weight_search,
        "build_fusion_eval_context",
        lambda **_: FusionEvalContext(targets={}, users=[], sequence_channel="sasrecf", final_top_k=12),
    )
    with pytest.raises(ValueError, match="Unknown exclude_seen_mode"):
        weight_search.run_weight_search(exclude_seen_mode="invalid", verbose=False)  # type: ignore[arg-type]
