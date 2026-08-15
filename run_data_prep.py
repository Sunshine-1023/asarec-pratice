"""Step 1/6 — Data preparation: preprocess → split → hm_seq → hm_seq.item. Examples: python run_data_prep.py; python run_data_prep.py --with-filter"""  # 步骤 1/6：数据准备（预处理、划分、序列化、商品特征）

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import sys  # 导入系统模块以处理中断退出
import time  # 导入时间模块以统计各步骤耗时
from pathlib import Path  # 导入路径处理类

from src.data.build_item_features import build_item_features  # 导入商品特征构建函数
from src.data.build_sequences import prepare_recbole_benchmark_files, read_max_item_list_length  # 数据层序列构建
from src.data.filter import run_filter  # 导入原始数据过滤函数
from src.data.manifest import build_processed_hm_manifest, write_manifest  # 数据快照清单
from src.data.preprocess import MIN_USER_PURCHASES, MAX_USER_HISTORY, WEEKS, build_inter_file, _default_input_path  # 导入交互文件构建函数
from src.data.split import split_bounds_dict, split_by_time  # 导入按时间划分数据集函数
from src.experiment.config import load_experiment_config  # 可选统一实验协议

DEFAULT_CONFIG = Path("configs/sasrecf.yaml")  # SASRecF 默认配置文件路径


def _run_step(name: str, fn) -> None:  # 执行单个数据准备子步骤并打印耗时
    print(f"\n{'=' * 60}")  # 打印步骤分隔线
    print(f"[{name}]")  # 打印步骤名称
    print("=" * 60)  # 打印分隔线结束
    started = time.perf_counter()  # 记录步骤开始时间
    fn()  # 调用子步骤函数
    elapsed = time.perf_counter() - started  # 计算步骤耗时
    print(f"Done in {elapsed:.1f}s")  # 打印步骤完成与耗时


def main() -> None:  # 命令行入口：按顺序执行数据准备流程
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description="Run data preparation steps in order for SASRecF / offline eval.",  # 程序描述
    )  # 参数解析器创建结束
    parser.add_argument(  # 定义 --with-filter 参数
        "--with-filter",  # 参数名
        action="store_true",  # 布尔开关
        help="Run src.data.filter first (writes data/raw/filtered/). "  # 帮助文本（前半）
        "If omitted, preprocess uses filtered data when it already exists.",  # 帮助文本（后半）
    )  # --with-filter 参数结束
    parser.add_argument(  # 定义 --config 参数
        "--config",  # 参数名
        type=Path,  # 路径类型
        default=DEFAULT_CONFIG,  # 默认配置文件
        help=f"Config for MAX_ITEM_LIST_LENGTH when building hm_seq (default: {DEFAULT_CONFIG})",  # 帮助文本
    )  # --config 参数结束
    parser.add_argument(  # 定义 --skip-item-features 参数
        "--skip-item-features",  # 参数名
        action="store_true",  # 布尔开关
        help="Skip hm_seq conversion and hm_seq.item (offline eval only needs hm.*).",  # 帮助文本
    )  # --skip-item-features 参数结束
    parser.add_argument(  # 统一实验协议，只影响数据窗口/阈值并写入 manifest
        "--experiment-config",  # 参数名
        type=Path,  # 路径类型
        default=None,  # 默认不强制
        help="Optional unified experiment YAML; records protocol into data/processed/manifest.json",  # 帮助文本
    )  # --experiment-config 结束
    args = parser.parse_args()  # 解析命令行参数

    config_path = args.config  # 获取配置文件路径
    if not config_path.exists():  # 配置文件不存在
        raise FileNotFoundError(f"Config not found: {config_path}")  # 抛出文件未找到错误

    experiment = None  # 可选实验协议
    if args.experiment_config is not None:  # 若提供了统一配置
        experiment = load_experiment_config(args.experiment_config)  # 加载并校验

    weeks = experiment.data.total_weeks if experiment is not None else WEEKS  # 总周数
    min_user_purchases = experiment.data.min_user_purchases if experiment is not None else MIN_USER_PURCHASES  # 最少购买
    max_user_history = experiment.data.max_user_history if experiment is not None else MAX_USER_HISTORY  # 历史上限
    split_result = None  # 保存切分边界供 manifest 使用

    if args.with_filter:  # 若指定先运行过滤
        _run_step(  # 执行过滤步骤
            "1/5 filter",  # 步骤名
            lambda: run_filter(  # 按协议过滤
                min_user_purchases=min_user_purchases,  # 最少购买
                max_user_behaviors=max_user_history,  # 每用户最大行为
                weeks=weeks,  # 周数
            ),  # 过滤调用结束
        )  # 过滤步骤结束
        step = 2  # 下一步从 2 开始编号
    else:  # 未指定过滤
        step = 1  # 从步骤 1 开始编号

    _run_step(  # 执行预处理
        f"{step}/5 preprocess",  # 步骤名
        lambda: build_inter_file(  # 构建交互文件
            weeks=weeks,  # 周数
            min_user_purchases=min_user_purchases,  # 最少购买
            max_user_history=max_user_history,  # 历史上限
        ),  # 预处理结束
    )  # 预处理步骤结束
    step += 1  # 步骤编号加一

    def _split() -> None:  # 闭包：按协议切分并记住边界
        nonlocal split_result  # 写外层变量
        split_kwargs = {}  # 切分参数
        if experiment is not None:  # 有统一协议时
            split_kwargs = {  # 使用协议周数
                "total_weeks": experiment.data.total_weeks,  # 总周数
                "train_weeks": experiment.data.history_weeks,  # 训练周
                "valid_weeks": experiment.data.valid_weeks,  # 验证周
                "test_weeks": experiment.data.test_weeks,  # 测试周
            }  # 参数结束
        split_result = split_by_time(**split_kwargs)  # 执行切分

    _run_step(f"{step}/5 split", _split)  # 执行按时间划分
    step += 1  # 步骤编号加一

    def _write_manifest() -> None:  # 数据准备成功后写快照
        preprocess = {  # 记录实际使用的预处理参数
            "weeks": weeks,  # 总周数
            "min_user_purchases": min_user_purchases,  # 最少购买
            "max_user_history": max_user_history,  # 历史上限
            "with_filter": bool(args.with_filter),  # 是否先过滤
            "skip_item_features": bool(args.skip_item_features),  # 是否跳过序列特征
            "sasrec_config": str(config_path),  # 序列配置
            "experiment_config": str(args.experiment_config) if args.experiment_config else None,  # 实验协议
            "experiment_name": experiment.experiment.name if experiment is not None else None,  # 实验名
            "seed": experiment.experiment.seed if experiment is not None else None,  # 种子
        }  # 预处理记录结束
        payload = build_processed_hm_manifest(  # 流式统计处理后文件
            raw_transactions=_default_input_path(),  # 实际输入交易
            preprocess=preprocess,  # 参数
            split_bounds=split_bounds_dict(split_result) if split_result is not None else {},  # 时间边界
            repo_root=Path(__file__).resolve().parent,  # 仓库根
        )  # 清单结束
        out = write_manifest(payload, Path("data/processed/manifest.json"))  # 写出
        print(f"Wrote data manifest: {out}")  # 提示路径

    if args.skip_item_features:  # 若跳过序列化与商品特征
        print("\nSkipped hm_seq + build_item_features (--skip-item-features).")  # 提示已跳过
        _write_manifest()  # 仍写出数据清单
        return  # 提前结束

    max_item_list_length = read_max_item_list_length(config_path)  # 从配置读取最大序列长度

    def _prepare_seq() -> None:  # 闭包：准备 hm_seq 序列文件
        prepare_recbole_benchmark_files(max_item_list_length)  # 调用 RecBole 基准文件准备

    _run_step(f"{step}/5 hm_seq", _prepare_seq)  # 执行 hm_seq 转换
    step += 1  # 步骤编号加一
    _run_step(f"{step}/5 build_item_features", build_item_features)  # 执行商品特征构建

    print("\nData preparation finished.")  # 提示数据准备完成
    _write_manifest()  # 写出数据快照清单
    print("Next: python run_sasrecf.py --skip-preprocess")  # 提示下一步训练命令


if __name__ == "__main__":  # 脚本直接运行时
    try:  # 捕获键盘中断
        main()  # 调用主函数
    except KeyboardInterrupt:  # 用户 Ctrl+C 中断
        print("\nInterrupted.", file=sys.stderr)  # 向 stderr 打印中断提示
        sys.exit(130)  # 以标准中断退出码退出
