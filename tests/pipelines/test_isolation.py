"""Regression tests for the two isolated workflow ownership boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from fashionrec.experiment.context import create_run_context
from fashionrec.pipeline.contracts import PipelineOptions
from fashionrec.pipeline.registry import build_pipeline_steps


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "fashionrec"
LEGACY_FACADE_FILES = {
    "data": {"__init__.py"},
    "recall": {"__init__.py", "rule_recall_export.py", "sasrec_recall.py"},
    "ranking": {"__init__.py"},
    "training": {"__init__.py"},
    "evaluation": {"__init__.py", "metrics.py"},
    "candidates": {"__init__.py"},
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


def test_legacy_algorithm_namespaces_are_facades_only() -> None:
    for component, expected_files in LEGACY_FACADE_FILES.items():
        root = PACKAGE_ROOT / component
        actual_files = {path.name for path in root.glob("*.py")}
        assert actual_files == expected_files


def test_alias_backed_legacy_modules_have_no_shadow_files() -> None:
    for relative in (
        "data/command.py",
        "recall/registry.py",
        "training/command.py",
        "training/checkpoint_command.py",
    ):
        assert not (PACKAGE_ROOT / relative).exists(), relative


def test_shared_kernel_never_imports_applications() -> None:
    imports = {
        module for path in (PACKAGE_ROOT / "shared").rglob("*.py") for module in _imports(path)
    }
    assert all(
        not module.startswith(("fashionrec.baseline", "fashionrec.industrial"))
        for module in imports
    )


def test_former_common_workflow_builder_is_removed() -> None:
    assert not (PACKAGE_ROOT / "pipeline" / "common" / "stages.py").exists()


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
        for step in build_pipeline_steps(
            baseline,
            python_executable="python",
            options=PipelineOptions(),
        )
    ]
    industrial_commands = [
        " ".join(step.command)
        for step in build_pipeline_steps(
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
