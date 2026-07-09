"""Step 1/6 — Data preparation: preprocess → split → hm_seq → hm_seq.item. Examples: python run_data_prep.py; python run_data_prep.py --with-filter"""  # 步骤 1/6：数据准备（预处理、划分、序列化、商品特征）

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import sys  # 导入系统模块以处理中断退出
import time  # 导入时间模块以统计各步骤耗时
from pathlib import Path  # 导入路径处理类

from run_sasrec import _read_max_item_list_length, prepare_recbole_benchmark_files  # 导入序列长度读取与 RecBole 基准文件准备
from src.data.build_item_features import build_item_features  # 导入商品特征构建函数
from src.data.filter import run_filter  # 导入原始数据过滤函数
from src.data.preprocess import build_inter_file  # 导入交互文件构建函数
from src.data.split import split_by_time  # 导入按时间划分数据集函数

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
    args = parser.parse_args()  # 解析命令行参数

    config_path = args.config  # 获取配置文件路径
    if not config_path.exists():  # 配置文件不存在
        raise FileNotFoundError(f"Config not found: {config_path}")  # 抛出文件未找到错误

    if args.with_filter:  # 若指定先运行过滤
        _run_step("1/5 filter", run_filter)  # 执行过滤步骤
        step = 2  # 下一步从 2 开始编号
    else:  # 未指定过滤
        step = 1  # 从步骤 1 开始编号

    _run_step(f"{step}/5 preprocess", build_inter_file)  # 执行预处理构建交互文件
    step += 1  # 步骤编号加一
    _run_step(f"{step}/5 split", split_by_time)  # 执行按时间划分
    step += 1  # 步骤编号加一

    if args.skip_item_features:  # 若跳过序列化与商品特征
        print("\nSkipped hm_seq + build_item_features (--skip-item-features).")  # 提示已跳过
        return  # 提前结束

    max_item_list_length = _read_max_item_list_length(config_path)  # 从配置读取最大序列长度

    def _prepare_seq() -> None:  # 闭包：准备 hm_seq 序列文件
        prepare_recbole_benchmark_files(max_item_list_length)  # 调用 RecBole 基准文件准备

    _run_step(f"{step}/5 hm_seq", _prepare_seq)  # 执行 hm_seq 转换
    step += 1  # 步骤编号加一
    _run_step(f"{step}/5 build_item_features", build_item_features)  # 执行商品特征构建

    print("\nData preparation finished.")  # 提示数据准备完成
    print("Next: python run_sasrecf.py --skip-preprocess")  # 提示下一步训练命令


if __name__ == "__main__":  # 脚本直接运行时
    try:  # 捕获键盘中断
        main()  # 调用主函数
    except KeyboardInterrupt:  # 用户 Ctrl+C 中断
        print("\nInterrupted.", file=sys.stderr)  # 向 stderr 打印中断提示
        sys.exit(130)  # 以标准中断退出码退出
