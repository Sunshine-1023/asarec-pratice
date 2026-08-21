"""Regression tests for the two isolated workflow ownership boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from fashionrec.baseline.pipeline.orchestrator import build_pipeline_steps as build_baseline_steps
from fashionrec.industrial.pipeline.orchestrator import build_pipeline_steps as build_industrial_steps
from fashionrec.shared.experiment.context import create_run_context
from fashionrec.shared.runtime.contracts import PipelineOptions


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "fashionrec"
RETIRED_NAMESPACES = {
    "candidates",
    "data",
    "domain",
    "evaluation",
    "experiment",
    "pipeline",
    "ranking",
    "recall",
    "training",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_baseline_application_never_imports_industrial_application() -> None:
    baseline_files = (PACKAGE_ROOT / "baseline").rglob("*.py")
    imports = {module for path in baseline_files for module in _imports(path)}
    assert all(not module.startswith("fashionrec.industrial") for module in imports)
    assert all(not module.startswith("fashionrec.pipeline") for module in imports)
    assert all(
        not module.startswith(
            (
                "fashionrec.data",
                "fashionrec.recall",
                "fashionrec.ranking",
                "fashionrec.training",
                "fashionrec.evaluation",
                "fashionrec.candidates",
            )
        )
        for module in imports
    )


def test_industrial_application_never_imports_baseline_application() -> None:
    imports = {
        module
        for path in (PACKAGE_ROOT / "industrial").rglob("*.py")
        for module in _imports(path)
    }
    assert all(not module.startswith("fashionrec.baseline") for module in imports)
    assert all(not module.startswith("fashionrec.pipeline") for module in imports)
    assert all(
        not module.startswith(
            (
                "fashionrec.data",
                "fashionrec.recall",
                "fashionrec.ranking",
                "fashionrec.training",
                "fashionrec.evaluation",
                "fashionrec.candidates",
            )
        )
        for module in imports
    )


def test_legacy_namespaces_are_removed() -> None:
    for component in RETIRED_NAMESPACES:
        assert not (PACKAGE_ROOT / component).exists(), component
    assert not (PACKAGE_ROOT / "pytorch_compat.py").exists()


def test_shared_kernel_never_imports_applications() -> None:
    imports = {
        module for path in (PACKAGE_ROOT / "shared").rglob("*.py") for module in _imports(path)
    }
    assert all(
        not module.startswith(("fashionrec.baseline", "fashionrec.industrial"))
        for module in imports
    )


def test_applications_own_distinct_model_configs() -> None:
    baseline = ROOT / "configs" / "baseline" / "models" / "sasrecf.yaml"
    industrial = ROOT / "configs" / "industrial" / "models" / "sasrecf.yaml"
    assert baseline.is_file() and industrial.is_file()
    assert baseline != industrial


def test_target_physical_packages_exist() -> None:
    for relative in (
        "shared/domain",
        "shared/interfaces",
        "shared/io",
        "shared/metrics",
        "shared/experiment",
        "shared/runtime",
        "baseline/data",
        "baseline/models/sasrecf",
        "baseline/recall",
        "baseline/ranking",
        "baseline/evaluation",
        "baseline/pipeline",
        "industrial/data",
        "industrial/models/sasrecf",
        "industrial/models/lambdarank",
        "industrial/recall",
        "industrial/ranking",
        "industrial/evaluation",
        "industrial/pipeline",
    ):
        assert (PACKAGE_ROOT / relative).is_dir(), relative


def test_formal_implementations_live_inside_applications() -> None:
    assert (PACKAGE_ROOT / "baseline" / "ranking" / "weighted_rrf.py").is_file()
    assert (PACKAGE_ROOT / "industrial" / "data" / "events.py").is_file()
    assert (PACKAGE_ROOT / "industrial" / "data" / "baskets.py").is_file()
    assert (PACKAGE_ROOT / "industrial" / "data" / "labels.py").is_file()
    assert (PACKAGE_ROOT / "industrial" / "data" / "features.py").is_file()
    assert (PACKAGE_ROOT / "industrial" / "models" / "lambdarank" / "train.py").is_file()
    assert (PACKAGE_ROOT / "industrial" / "models" / "lambdarank" / "predict.py").is_file()
    assert (PACKAGE_ROOT / "shared" / "experiment" / "context.py").is_file()
    assert (PACKAGE_ROOT / "shared" / "runtime" / "pytorch_compat.py").is_file()


def test_formal_sources_do_not_import_support_facades() -> None:
    for component in ("baseline", "industrial", "shared"):
        imports = {
            module
            for path in (PACKAGE_ROOT / component).rglob("*.py")
            for module in _imports(path)
        }
        assert all(not module.startswith("fashionrec.experiment") for module in imports)
        assert "fashionrec.pytorch_compat" not in imports


def test_baseline_does_not_carry_industrial_only_implementations() -> None:
    for relative in (
        "data/build_events.py",
        "data/build_baskets.py",
        "data/labels.py",
        "data/snapshots.py",
        "data/user_features.py",
        "data/cross_features.py",
        "data/customer_features.py",
        "data/item_features.py",
        "recall/repurchase.py",
        "recall/style.py",
        "recall/content.py",
        "evaluation/experiment_report.py",
        "evaluation/candidate_diagnostics.py",
        "evaluation/coverage_metrics.py",
    ):
        assert not (PACKAGE_ROOT / "baseline" / relative).exists(), relative


def test_industrial_data_uses_canonical_event_and_basket_modules() -> None:
    industrial_data = PACKAGE_ROOT / "industrial" / "data"
    assert (industrial_data / "events.py").is_file()
    assert (industrial_data / "baskets.py").is_file()
    assert not (industrial_data / "build_events.py").exists()
    assert not (industrial_data / "build_baskets.py").exists()


def test_profile_dags_have_independent_ranker_commands(tmp_path: Path) -> None:
    baseline = create_run_context(
        "configs/baseline/experiment.yaml",
        output_root=tmp_path,
        run_id="same-id",
    )
    industrial = create_run_context(
        "configs/industrial/experiment.yaml",
        output_root=tmp_path,
        run_id="same-id",
    )
    baseline_commands = [
        " ".join(step.command)
        for step in build_baseline_steps(
            baseline,
            python_executable="python",
            options=PipelineOptions(),
        )
    ]
    industrial_commands = [
        " ".join(step.command)
        for step in build_industrial_steps(
            industrial,
            python_executable="python",
            options=PipelineOptions(),
        )
    ]
    assert all("ranker-" not in command for command in baseline_commands)
    assert any("ranker-sequence" in command for command in industrial_commands)
    assert any("ranker-dataset" in command for command in industrial_commands)
    assert any("ranker-train" in command for command in industrial_commands)
    assert baseline.artifacts.root != industrial.artifacts.root
