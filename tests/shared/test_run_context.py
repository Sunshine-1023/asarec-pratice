"""Tests for run-scoped artifact isolation."""  # 运行上下文测试

from __future__ import annotations  # 延迟注解

import json  # 检查冻结配置
from pathlib import Path  # 临时路径

import pytest

from fashionrec.shared.experiment.artifacts import RunArtifacts  # 产物路径
from fashionrec.shared.experiment.context import RunConfigurationConflictError, create_run_context  # 上下文工厂


def test_run_artifacts_are_isolated_by_run_id(tmp_path: Path) -> None:  # 不同实验不覆盖
    first = RunArtifacts.from_root(tmp_path, "run-a")  # 第一次
    second = RunArtifacts.from_root(tmp_path, "run-b")  # 第二次
    assert first.recall_file("sasrecf", "valid") != second.recall_file("sasrecf", "valid")  # 隔离
    assert first.metrics_file("test") == tmp_path / "baseline" / "run-a" / "evaluation" / "test_metrics.json"  # 契约


def test_profiles_are_isolated_even_with_the_same_run_id(tmp_path: Path) -> None:
    baseline = RunArtifacts.from_root(tmp_path, "same", profile="baseline")
    industrial = RunArtifacts.from_root(tmp_path, "same", profile="industrial")
    assert baseline.root == tmp_path / "baseline" / "same"
    assert industrial.root == tmp_path / "industrial" / "same"
    assert baseline.root != industrial.root


@pytest.mark.parametrize("run_id", ["../other", "nested/run", "nested\\run", ".", ".."])
def test_run_id_cannot_escape_profile_namespace(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="path-safe"):
        RunArtifacts.from_root(tmp_path, run_id, profile="baseline")


def test_create_run_context_freezes_resolved_config(tmp_path: Path) -> None:  # 初始化上下文
    context = create_run_context(  # 创建
        config_path="configs/baseline/experiment.yaml",
        output_root=tmp_path,  # 临时输出
        run_id="fixed-run",  # 固定 ID
        strict=True,  # 正式模式
        initialize=True,  # 创建目录
    )  # 创建结束
    assert context.artifacts.resolved_config.exists()  # 配置已冻结
    payload = json.loads(context.artifacts.resolved_config.read_text(encoding="utf-8"))  # 读取
    assert payload["run_id"] == "fixed-run"  # ID
    assert payload["profile"] == "baseline"
    assert payload["strict"] is True  # 模式
    assert len(payload["config_sha256"]) == 64  # SHA256
    assert context.artifacts.candidates.is_dir()  # 目录齐全
    assert context.artifacts.data.is_dir()  # 处理后数据隔离到本次 run
    assert payload["data"]["history_weeks"] == 26  # 冻结 26 周协议
    assert payload["label"]["target_mode"] == "next_basket"  # 标签协议写入快照
    assert payload["ranking"]["enabled"] is False  # 学习排序默认关闭


def test_profile_must_match_ranking_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="profile/config mismatch"):
        create_run_context(
            "configs/baseline/experiment.yaml",
            output_root=tmp_path,
            run_id="wrong-profile",
            profile="industrial",
        )


def test_existing_run_rejects_changed_config(tmp_path: Path) -> None:
    original = tmp_path / "baseline.yaml"
    original.write_text(Path("configs/baseline/experiment.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    create_run_context(original, output_root=tmp_path, run_id="fixed", profile="baseline", initialize=True)
    original.write_text(original.read_text(encoding="utf-8").replace("seed: 2026", "seed: 2027"), encoding="utf-8")
    with pytest.raises(RunConfigurationConflictError, match="different profile or experiment config"):
        create_run_context(original, output_root=tmp_path, run_id="fixed", profile="baseline", initialize=True)
