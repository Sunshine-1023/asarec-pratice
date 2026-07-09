"""Run the full SASRecF + four-channel offline fusion pipeline in order. Steps: run_data_prep, run_sasrecf, run_sasrecf_recall (valid+test), optional run_rule_recall, run_fusion_weight_search, run_offline_eval (valid+test)."""  # 按顺序运行完整 SASRecF 四路融合离线流水线

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import subprocess  # 导入子进程模块以执行流水线各步骤
import sys  # 导入系统模块以获取 Python 解释器路径
import time  # 导入时间模块以统计步骤耗时
from pathlib import Path  # 导入路径处理类

ROOT = Path(__file__).resolve().parent  # 项目根目录（本脚本所在目录）


def _run_step(step_no: int, total: int, title: str, cmd: list[str]) -> None:  # 执行单个流水线步骤并打印进度
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


def main() -> None:  # 命令行入口：组装并顺序执行流水线步骤
    parser = argparse.ArgumentParser(description="Run full v2 experiment pipeline in order.")  # 创建参数解析器
    parser.add_argument(  # 定义 --with-filter 参数
        "--with-filter",  # 参数名
        action="store_true",  # 布尔开关
        help="Pass --with-filter to run_data_prep.py (step 1)",  # 帮助文本
    )  # --with-filter 参数结束
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip step 1")  # 跳过步骤 1 数据准备
    parser.add_argument("--skip-train", action="store_true", help="Skip step 2 (SASRecF training)")  # 跳过步骤 2 SASRecF 训练
    parser.add_argument("--skip-recall", action="store_true", help="Skip step 3 (SASRecF recall export)")  # 跳过步骤 3 召回导出
    parser.add_argument(  # 定义 --export-rule-recall 参数
        "--export-rule-recall",  # 参数名
        action="store_true",  # 布尔开关
        help="Run step 4: export Popular / Category Popular / Item2Item CSV (optional debug)",  # 帮助文本
    )  # --export-rule-recall 参数结束
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
        default=Path("outputs/evaluation/best_fusion_weights.json"),  # 默认融合权重文件
        help="Weights file for test offline_eval (default: best_fusion_weights.json)",  # 帮助文本
    )  # --weights-json 参数结束
    args = parser.parse_args()  # 解析命令行参数

    py = sys.executable  # 当前 Python 解释器路径
    steps: list[tuple[str, list[str]]] = []  # 待执行步骤列表（标题, 命令）

    if not args.skip_data_prep:  # 未跳过数据准备时
        cmd = [py, "run_data_prep.py"]  # 组装数据准备命令
        if args.with_filter:  # 若指定 --with-filter
            cmd.append("--with-filter")  # 追加过滤参数
        steps.append(("数据准备", cmd))  # 注册步骤 1

    if not args.skip_train:  # 未跳过训练时
        steps.append(("训练 SASRecF", [py, "run_sasrecf.py", "--skip-preprocess"]))  # 注册步骤 2

    if not args.skip_recall:  # 未跳过召回导出时
        steps.append(("SASRecF 召回 valid", [py, "run_sasrecf_recall.py", "--eval-split", "valid"]))  # 注册 valid 召回
        steps.append(("SASRecF 召回 test", [py, "run_sasrecf_recall.py", "--eval-split", "test"]))  # 注册 test 召回

    if args.export_rule_recall:  # 若指定导出规则召回
        steps.append(("规则三路召回导出", [py, "run_rule_recall.py", "--eval-split", "both"]))  # 注册可选步骤 4

    if not args.skip_weight_search:  # 未跳过权重搜索时
        steps.append(("融合权重搜索 (valid)", [py, "run_fusion_weight_search.py"]))  # 注册步骤 5

    if not args.skip_valid_eval:  # 未跳过 valid 评估时
        steps.append(("离线融合评估 valid", [py, "run_offline_eval.py", "--eval-split", "valid"]))  # 注册步骤 6a

    if not args.skip_test_eval:  # 未跳过 test 评估时
        test_cmd = [py, "run_offline_eval.py", "--eval-split", "test"]  # 组装 test 评估命令
        if args.weights_json.exists():  # 若权重文件存在
            test_cmd.extend(["--weights-json", str(args.weights_json)])  # 追加权重文件参数
        else:  # 权重文件不存在
            print(  # 打印警告
                f"\nWarning: {args.weights_json} not found; "  # 提示文件未找到
                "test eval will use default activity weights."  # 将使用默认活动权重
            )  # 警告打印结束
        steps.append(("离线融合评估 test", test_cmd))  # 注册步骤 6b

    if not steps:  # 若所有步骤均被跳过
        print("No steps to run (all skipped).")  # 提示无步骤可执行
        return  # 直接返回

    total = len(steps)  # 总步骤数
    pipeline_started = time.perf_counter()  # 记录流水线开始时间
    print(f"Pipeline: {total} step(s)")  # 打印待执行步骤总数

    for i, (title, cmd) in enumerate(steps, start=1):  # 遍历每个步骤
        _run_step(i, total, title, cmd)  # 执行当前步骤

    print(f"\nPipeline finished in {time.perf_counter() - pipeline_started:.1f}s")  # 打印流水线总耗时


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用主函数
