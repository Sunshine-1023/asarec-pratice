"""验证顶层只暴露明确的双应用路由。"""

from __future__ import annotations  # 启用延迟注解

import tomllib  # 解析项目包元数据
from pathlib import Path  # 处理仓库路径
from types import SimpleNamespace

from fashionrec import cli  # 统一 CLI
from fashionrec.baseline import cli as baseline_cli
from fashionrec.industrial import cli as industrial_cli


ROOT = Path(__file__).resolve().parents[2]


def test_root_has_no_python_startup_scripts() -> None:  # 根目录不再暴露启动脚本
    assert list(ROOT.glob("run_*.py")) == []  # 所有 Python 命令必须位于正式包内


def test_cli_exposes_only_explicit_application_routes() -> None:
    assert set(cli.COMMANDS) == {"baseline", "industrial", "profile-data"}
    assert cli.COMMANDS["baseline"].module == "fashionrec.baseline.cli"
    assert cli.COMMANDS["industrial"].module == "fashionrec.industrial.cli"
    assert cli.COMMANDS["profile-data"].module == "fashionrec.industrial.data.profile"


def test_two_applications_expose_independent_command_surfaces() -> None:
    assert "ranker-train" not in baseline_cli.COMMANDS
    assert "ranker-dataset" not in baseline_cli.COMMANDS
    assert industrial_cli.COMMANDS["ranker-dataset"].module.startswith("fashionrec.industrial")
    assert industrial_cli.COMMANDS["ranker-sequence"].module == (
        "fashionrec.industrial.models.sasrecf.ranking_features"
    )
    assert baseline_cli.COMMANDS["pipeline"].module.startswith("fashionrec.baseline")
    assert industrial_cli.COMMANDS["pipeline"].module.startswith("fashionrec.industrial")


def test_pyproject_registers_one_console_script() -> None:  # 校验安装后的公开命令
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))  # 读取元数据
    assert payload["project"]["scripts"] == {"fashionrec": "fashionrec.cli:main"}  # 仅暴露一个脚本


def test_cli_dispatches_application_arguments_without_global_mutation(monkeypatch) -> None:
    received: list[str] = []  # 记录领域命令收到的参数

    def command_main(argv: list[str] | None = None) -> None:  # 假训练命令
        received.extend(argv or [])  # 保存收到的参数

    fake_module = SimpleNamespace(main=command_main)  # 构造符合命令契约的模块
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: fake_module)  # 替换延迟导入
    assert cli.main(["baseline", "train", "--config", "custom.yaml"]) == 0
    assert received == ["train", "--config", "custom.yaml"]
