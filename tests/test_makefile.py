"""验证 Makefile 操作入口不会绕过正式 Python 编排器。"""

from __future__ import annotations  # 启用延迟注解评估

import subprocess  # 启动 Make 子进程进行无训练验证
from pathlib import Path  # 处理项目根目录路径


ROOT = Path(__file__).resolve().parents[1]  # 项目根目录


def _run_make(*args: str) -> subprocess.CompletedProcess[str]:  # 执行 Make 命令并捕获结果
    return subprocess.run(  # 返回完整子进程状态供断言
        ["make", *args],  # 组装 Make 参数
        cwd=ROOT,  # 固定从项目根目录执行
        check=False,  # 由测试显式检查退出码
        capture_output=True,  # 捕获标准输出与错误输出
        text=True,  # 以文本形式返回输出
    )  # 完成 Make 子进程调用


def test_makefile_pipeline_delegates_to_python_orchestrator() -> None:  # 验证完整流程委托给统一编排器
    result = _run_make("-n", "pipeline", "RUN_ID=make-smoke", "WITH_FILTER=1")  # 只展开命令
    assert result.returncode == 0  # Make 解析成功
    assert "python -m fashionrec pipeline" in result.stdout  # 使用唯一公开 CLI
    assert "--run-id \"make-smoke\"" in result.stdout  # 透传运行 ID
    assert "--with-filter" in result.stdout  # 透传过滤开关


def test_staged_make_target_requires_explicit_run_id() -> None:  # 验证分阶段命令拒绝匿名运行
    result = _run_make("train")  # 不提供运行 ID
    assert result.returncode != 0  # 应在训练启动前失败
    assert "RUN_ID is required for staged commands" in result.stdout  # 给出可操作提示


def test_train_target_runs_only_training_stage() -> None:  # 验证训练目标不会运行其他阶段
    result = _run_make("-n", "train", "RUN_ID=make-smoke")  # 只展开训练命令
    assert result.returncode == 0  # Make 解析成功
    assert "--skip-data-prep" in result.stdout  # 跳过数据准备
    assert "--skip-checkpoint-selection" in result.stdout  # 跳过 checkpoint 选择
    assert "--skip-test-eval" in result.stdout  # 跳过最终测试评估
    assert "--skip-train" not in result.stdout  # 保留模型训练阶段
