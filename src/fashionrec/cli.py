"""Single public command dispatcher for all FashionRec workflows."""  # FashionRec 唯一公开命令分发器

from __future__ import annotations  # 启用延迟注解

import argparse  # 解析顶层命令
import importlib  # 延迟加载命令模块
import sys  # 读取未显式传入的命令参数
from dataclasses import dataclass  # 命令元数据
from typing import Callable  # 命令函数类型


@dataclass(frozen=True, slots=True)  # 不可变命令描述
class CommandSpec:  # 单个子命令定义
    module: str  # 命令实现模块
    description: str  # 帮助说明
    default_args: tuple[str, ...] = ()  # 主线默认参数


COMMANDS: dict[str, CommandSpec] = {  # 统一公开命令表
    "pipeline": CommandSpec("fashionrec.pipeline.command", "Legacy profile pipeline compatibility entry"),
    "data": CommandSpec("fashionrec.data.command", "Prepare causal train/valid/test data"),  # 数据准备
    "profile-data": CommandSpec("fashionrec.data.profile", "Profile raw transactions/customers/articles"),  # raw 数据体检
    "train": CommandSpec(  # SASRecF 训练
        "fashionrec.training.command",  # 训练命令模块
        "Train the sequence model",  # 帮助说明
        ("--config", "configs/sasrecf.yaml"),  # 默认使用主线 SASRecF 配置
    ),  # 训练命令结束
    "select-checkpoint": CommandSpec(  # checkpoint 选择
        "fashionrec.training.checkpoint_command",  # 选择命令模块
        "Select a checkpoint on valid user-week MAP@K",  # 帮助说明
    ),  # checkpoint 命令结束
    "recall": CommandSpec(  # 序列召回
        "fashionrec.recall.sasrec_recall",  # 召回命令模块
        "Export sequence-model recall",  # 帮助说明
        ("--config", "configs/sasrecf.yaml"),  # 默认使用主线 SASRecF 配置
    ),  # 召回命令结束
    "candidates": CommandSpec(  # 候选物化
        "fashionrec.recall.rule_recall_export",  # 候选命令模块
        "Materialize rule and sequence candidates",  # 帮助说明
    ),  # 候选命令结束
    "weights": CommandSpec("fashionrec.evaluation.weight_search", "Search fusion weights on valid"),  # 搜权
    "ranker-dataset": CommandSpec(
        "fashionrec.industrial.ranking.dataset_materialization",
        "Build causal LambdaRank parquet tables",
    ),
    "ranker-train": CommandSpec("fashionrec.ranking.train", "Train LightGBM LambdaRank"),  # 学习排序训练
    "ranker-predict": CommandSpec("fashionrec.ranking.predict", "Score candidates with LambdaRank"),  # 学习排序推理
    "evaluate": CommandSpec("fashionrec.evaluation.offline_eval", "Evaluate fixed candidates and weights"),  # 评估
    "baseline": CommandSpec("fashionrec.evaluation.baseline_command", "Evaluate the current baseline"),  # 基线
}  # 公开命令表结束


def _command_help() -> str:  # 生成子命令帮助文本
    width = max(len(name) for name in COMMANDS)  # 对齐命令名宽度
    return "\n".join(  # 每个命令占一行
        f"  {name:<{width}}  {spec.description}"  # 命令名与说明
        for name, spec in COMMANDS.items()  # 保持定义顺序
    )  # 返回帮助文本


def _load_main(spec: CommandSpec) -> Callable[[list[str] | None], object]:  # 延迟加载命令 main
    module = importlib.import_module(spec.module)  # 只加载被请求的领域模块
    command_main = getattr(module, "main", None)  # 获取标准 main 函数
    if not callable(command_main):  # 命令模块契约不完整
        raise RuntimeError(f"Command module has no callable main(): {spec.module}")  # 快速失败
    return command_main  # 返回命令入口


def main(argv: list[str] | None = None) -> int:  # 统一 CLI 入口
    parser = argparse.ArgumentParser(  # 顶层解析器只负责选择领域命令
        prog="fashionrec",  # 安装后和模块调用使用同一名称
        description="FashionRec training and offline recommendation pipeline",  # 项目说明
        epilog=f"commands:\n{_command_help()}",  # 展示命令清单
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 保留帮助换行
    )  # 解析器创建结束
    parser.add_argument("command", nargs="?", help="Command to run")  # 子命令名称
    raw_args = list(sys.argv[1:] if argv is None else argv)  # 复制参数，避免修改全局状态
    if not raw_args or raw_args[0] in {"-h", "--help"}:  # 未指定命令或请求总帮助
        parser.print_help()  # 展示统一帮助
        return 0  # 正常退出
    command, command_args = raw_args[0], raw_args[1:]  # 仅消费第一个命令 token
    spec = COMMANDS.get(command)  # 查找命令定义
    if spec is None:  # 未知命令
        parser.error(f"unknown command: {command}")  # 使用 argparse 标准错误
    command_main = _load_main(spec)  # 延迟导入真实命令
    command_main([*spec.default_args, *command_args])  # 默认参数在前，用户参数可覆盖
    return 0  # 命令正常完成
