"""验证项目只暴露统一 FashionRec 包和命令入口。"""  # 公开结构回归测试

from __future__ import annotations  # 启用延迟注解

import tomllib  # 解析项目包元数据
from pathlib import Path  # 处理仓库路径
from types import SimpleNamespace  # 构造假的命令模块

from fashionrec import cli  # 统一 CLI
from fashionrec.baseline import cli as baseline_cli
from fashionrec.industrial import cli as industrial_cli


ROOT = Path(__file__).resolve().parents[1]  # 项目根目录


def test_root_has_no_python_startup_scripts() -> None:  # 根目录不再暴露启动脚本
    assert list(ROOT.glob("run_*.py")) == []  # 所有 Python 命令必须位于正式包内


def test_cli_exposes_profile_data_command() -> None:  # raw 体检必须走统一 CLI
    assert "profile-data" in cli.COMMANDS  # 命令表含 profile-data
    assert cli.COMMANDS["profile-data"].module == "fashionrec.data.profile"  # 指向流式 profile 模块


def test_cli_exposes_ranker_commands() -> None:
    assert cli.COMMANDS["ranker-train"].module == "fashionrec.ranking.train"
    assert cli.COMMANDS["ranker-predict"].module == "fashionrec.ranking.predict"
    assert cli.COMMANDS["ranker-dataset"].module == "fashionrec.industrial.ranking.dataset_materialization"


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


def test_cli_dispatches_remaining_arguments_without_global_mutation(monkeypatch) -> None:  # 校验参数分发
    received: list[str] = []  # 记录领域命令收到的参数

    def command_main(argv: list[str] | None = None) -> None:  # 假训练命令
        received.extend(argv or [])  # 保存收到的参数

    fake_module = SimpleNamespace(main=command_main)  # 构造符合命令契约的模块
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: fake_module)  # 替换延迟导入
    assert cli.main(["train", "--config", "custom.yaml"]) == 0  # 执行统一分发
    assert received == ["--config", "configs/sasrecf.yaml", "--config", "custom.yaml"]  # 用户参数最后覆盖
