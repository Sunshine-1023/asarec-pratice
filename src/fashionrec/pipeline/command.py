"""Run the full SASRecF + four-channel offline fusion pipeline through shared scripts and module CLIs."""  # 按顺序运行完整 SASRecF 四路融合离线流水线

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import subprocess  # 导入子进程模块以执行流水线各步骤
import sys  # 导入系统模块以获取 Python 解释器路径
import time  # 导入时间模块以统计步骤耗时
from pathlib import Path  # 导入路径处理类

from fashionrec.experiment.context import create_run_context  # 单次运行上下文
from fashionrec.pipeline.orchestrator import PipelineOptions, build_pipeline_steps  # 纯编排计划

ROOT = Path.cwd()  # 命令从当前项目工作目录解析配置和数据


def _run_step(step_no: int, total: int, title: str, cmd: tuple[str, ...]) -> None:  # 执行单个流水线步骤并打印进度
    print(f"\n{'=' * 60}")  # 打印步骤分隔线
    print(f"[{step_no}/{total}] {title}")  # 打印当前步骤编号与标题
    print(f"命令: {' '.join(cmd)}")  # 打印即将执行的命令
    print("=" * 60)  # 打印分隔线结束
    started = time.perf_counter()  # 记录步骤开始时间
    result = subprocess.run(cmd, cwd=ROOT)  # 在项目根目录执行子进程命令
    elapsed = time.perf_counter() - started  # 计算步骤耗时
    if result.returncode != 0:  # 若子进程非零退出
        raise SystemExit(f"Step {step_no} failed (exit {result.returncode}): {' '.join(cmd)}")  # 以失败码退出并提示命令
    print(f"完成，耗时 {elapsed:.1f}s")  # 打印步骤完成与耗时


def main(argv: list[str] | None = None) -> None:  # 命令行入口：组装并顺序执行流水线步骤
    parser = argparse.ArgumentParser(prog="fashionrec pipeline", description="Run full experiment pipeline in order.")  # 创建参数解析器
    parser.add_argument(  # 定义 --with-filter 参数
        "--with-filter",  # 参数名
        action="store_true",  # 布尔开关
        help="Apply train-fitted item filtering during data preparation",  # 帮助文本
    )  # --with-filter 参数结束
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip step 1")  # 跳过步骤 1 数据准备
    parser.add_argument("--skip-train", action="store_true", help="Skip step 2 (SASRecF training)")  # 跳过步骤 2 SASRecF 训练
    parser.add_argument(
        "--skip-checkpoint-selection",
        action="store_true",
        help="Reuse the run's existing sasrecf_selected.pth instead of re-scoring valid",
    )
    parser.add_argument("--skip-recall", action="store_true", help="Skip step 3 (SASRecF recall export)")  # 跳过步骤 3 召回导出
    parser.add_argument("--skip-candidates", action="store_true", help="Skip four-channel candidate materialization")
    parser.add_argument("--skip-weight-search", action="store_true", help="Skip step 5")  # 跳过步骤 5 权重搜索
    parser.add_argument(  # 定义 --skip-valid-eval 参数
        "--skip-valid-eval",  # 参数名
        action="store_true",  # 布尔开关
        help="Skip valid offline_eval (step 6a)",  # 帮助文本
    )  # --skip-valid-eval 参数结束
    parser.add_argument(  # 定义 --skip-test-eval 参数
        "--skip-test-eval",  # 参数名
        action="store_true",  # 布尔开关
        help="Skip test offline_eval (step 6b)",  # 帮助文本
    )  # --skip-test-eval 参数结束
    parser.add_argument(  # 定义 --weights-json 参数
        "--weights-json",  # 参数名
        type=Path,  # 路径类型
        default=None,  # 默认使用当前 run 内搜索结果
        help="Optional existing weights file; default uses current run ranking artifact",  # 帮助文本
    )  # --weights-json 参数结束
    parser.add_argument(  # 统一实验协议配置
        "--experiment-config",  # 参数名
        type=Path,  # 路径类型
        default=Path("configs/experiment.yaml"),  # 默认实验 YAML
        help="Unified experiment protocol YAML (passed to data prep; default: configs/experiment.yaml)",  # 帮助文本
    )  # --experiment-config 结束
    parser.add_argument("--run-id", type=str, default=None, help="Reuse a specific run-scoped artifact directory")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"), help="Run artifact root")
    parser.add_argument("--no-strict", action="store_true", help="Compatibility mode; formal runs are strict by default")
    args = parser.parse_args(argv)  # 解析显式命令参数

    context = create_run_context(  # 一次解析配置并冻结到独立运行目录
        config_path=args.experiment_config,
        output_root=args.output_root,
        run_id=args.run_id,
        strict=not args.no_strict,
        initialize=True,
    )
    options = PipelineOptions(  # 将 CLI 开关转为纯编排选项
        with_filter=args.with_filter,
        skip_data_prep=args.skip_data_prep,
        skip_train=args.skip_train,
        skip_checkpoint_selection=args.skip_checkpoint_selection,
        skip_recall=args.skip_recall,
        skip_candidates=args.skip_candidates,
        skip_weight_search=args.skip_weight_search,
        skip_valid_eval=args.skip_valid_eval,
        skip_test_eval=args.skip_test_eval,
        weights_json=str(args.weights_json) if args.weights_json is not None else None,
    )
    steps = build_pipeline_steps(context, python_executable=sys.executable, options=options)

    if not steps:  # 若所有步骤均被跳过
        print("No steps to run (all skipped).")  # 提示无步骤可执行
        context.write_manifest(status="no_steps", completed_steps=[])
        return  # 直接返回

    total = len(steps)  # 总步骤数
    pipeline_started = time.perf_counter()  # 记录流水线开始时间
    print(f"Pipeline run_id={context.run_id}: {total} step(s)")  # 打印运行 ID 与步骤总数

    completed: list[str] = []
    try:
        for i, step in enumerate(steps, start=1):  # 遍历每个步骤
            _run_step(i, total, step.name, step.command)  # 执行当前步骤
            completed.append(step.name)
    except BaseException:
        context.write_manifest(status="failed", completed_steps=completed)
        raise

    context.write_manifest(status="complete", completed_steps=completed)
    print(f"\nPipeline finished in {time.perf_counter() - pipeline_started:.1f}s")  # 打印流水线总耗时
    print(f"Run artifacts: {context.artifacts.root}")


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用主函数
