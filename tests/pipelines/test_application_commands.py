"""Application wrappers must fix protocol choices before calling shared services."""

from __future__ import annotations

import pytest

from fashionrec.baseline.data import command as baseline_data
from fashionrec.baseline.evaluation import command as baseline_evaluation
from fashionrec.baseline.ranking import weights as baseline_weights
from fashionrec.baseline.recall import command as baseline_candidates
from fashionrec.industrial.data import command as industrial_data
from fashionrec.industrial.evaluation import command as industrial_evaluation
from fashionrec.industrial.ranking import weights as industrial_weights
from fashionrec.industrial.recall import command as industrial_candidates


def test_baseline_data_forces_baseline_configs_and_forbids_industrial_flags(monkeypatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(baseline_data, "run_shared_data_service", lambda argv: captured.extend(argv))
    baseline_data.main(["--processed-dir", "run/data"])
    assert captured[-4:] == [
        "--config",
        "configs/baseline/models/sasrecf.yaml",
        "--experiment-config",
        "configs/baseline/experiment.yaml",
    ]
    assert "--build-labels" not in captured
    with pytest.raises(ValueError, match="does not allow"):
        baseline_data.main(["--build-labels"])


def test_industrial_data_always_builds_protocol_artifacts(monkeypatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(industrial_data, "run_shared_data_service", lambda argv: captured.extend(argv))
    industrial_data.main(["--processed-dir", "run/data"])
    for flag in ("--build-events", "--build-baskets", "--build-labels", "--build-user-features"):
        assert flag in captured
    assert "configs/industrial/models/sasrecf.yaml" in captured
    assert "configs/industrial/experiment.yaml" in captured


def test_application_candidate_registries_are_fixed(monkeypatch) -> None:
    baseline_args: list[str] = []
    industrial_args: list[str] = []
    monkeypatch.setattr(baseline_candidates, "run_candidate_service", lambda argv: baseline_args.extend(argv))
    monkeypatch.setattr(industrial_candidates, "run_candidate_service", lambda argv: industrial_args.extend(argv))
    baseline_candidates.main([])
    industrial_candidates.main([])
    assert baseline_args[-2:] == ["--channels", "popular,category_popular,item2item"]
    assert industrial_args[-2:] == [
        "--channels",
        "popular,category_popular,item2item,repurchase,style,content",
    ]


def test_ranking_and_evaluation_protocols_reject_cross_application_use(monkeypatch) -> None:
    monkeypatch.setattr(baseline_weights, "run_weight_search", lambda _argv: None)
    monkeypatch.setattr(industrial_weights, "run_weight_search", lambda _argv: None)
    monkeypatch.setattr(baseline_evaluation, "run_evaluation", lambda _argv: None)
    monkeypatch.setattr(industrial_evaluation, "run_evaluation", lambda _argv: None)
    with pytest.raises(ValueError, match="does not allow"):
        baseline_weights.main(["--labels-dir", "labels"])
    with pytest.raises(ValueError, match="requires --labels-dir"):
        industrial_weights.main([])
    with pytest.raises(ValueError, match="does not allow"):
        baseline_evaluation.main(["--ranker-scored-csv", "scores.csv"])
    with pytest.raises(ValueError, match="requires --labels-dir"):
        industrial_evaluation.main(["--ranker-scored-csv", "scores.csv"])
    industrial_evaluation.main(["--labels-dir", "labels", "--ranker-scored-csv", "scores.csv"])
