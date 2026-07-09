"""Export rule-based recall channels (Popular / Category Popular / Item2Item) to CSV."""  # 导出规则召回通道（热门/类别热门/商品共现）到 CSV

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析
import csv  # 导入 CSV 读写
import sys  # 导入系统模块用于路径注入
import time  # 导入计时工具
from pathlib import Path  # 导入路径处理类
from typing import Literal  # 导入字面量类型

import pandas as pd  # 导入 pandas 数据分析库

if __package__ is None or __package__ == "":  # 若以脚本方式直接运行
    project_root = Path(__file__).resolve().parents[2]  # 定位项目根目录
    if str(project_root) not in sys.path:  # 若根目录不在搜索路径中
        sys.path.insert(0, str(project_root))  # 注入项目根目录到 sys.path

from src.fusion.weighted_fusion import build_user_history  # 导入用户历史构建函数
from src.recall.category_popular import (  # 导入类别热门召回相关符号
    CATEGORY_POPULAR_RECALL_TOP_K,  # 类别热门召回 Top-K 默认值
    SEED_ITEMS as CATEGORY_SEED_ITEMS,  # 类别热门种子商品数
    build_category_popular_index,  # 构建类别热门索引
    recall_category_popular,  # 类别热门召回函数
)  # 类别热门召回导入结束
from src.recall.item2item import (  # 导入商品共现召回相关符号
    COOCCUR_WEEKS,  # 共现统计窗口周数
    ITEM2ITEM_RECALL_TOP_K,  # 商品共现召回 Top-K 默认值
    SEED_ITEMS as ITEM2ITEM_SEED_ITEMS,  # 商品共现种子商品数
    TOP_SIM_K,  # 每个商品保留的相似邻居数
    build_item2item_index,  # 构建商品共现索引
    recall_item2item,  # 商品共现召回函数
)  # 商品共现召回导入结束
from src.recall.popular import POPULAR_RECALL_TOP_K, build_popular_index, recall_popular  # 导入热门召回常量与函数

TRAIN_INTER = Path("data/processed/hm/hm.train.inter")  # 训练集交互文件路径
VALID_INTER = Path("data/processed/hm/hm.valid.inter")  # 验证集交互文件路径
TEST_INTER = Path("data/processed/hm/hm.test.inter")  # 测试集交互文件路径
OUTPUT_DIR = Path("outputs/recommendations")  # 召回结果输出目录

ChannelName = Literal["popular", "category_popular", "item2item"]  # 支持的规则召回通道名
ALL_CHANNELS: tuple[ChannelName, ...] = ("popular", "category_popular", "item2item")  # 全部规则召回通道


def _load_eval_users(eval_split: str) -> list[str]:  # 加载评估划分中的用户 ID 列表
    path = VALID_INTER if eval_split == "valid" else TEST_INTER  # 选择验证或测试交互文件
    if not path.exists():  # 若文件不存在
        raise FileNotFoundError(f"Missing eval split file: {path}")  # 抛出文件缺失异常
    df = pd.read_csv(path, sep="\t", usecols=["user_id:token"])  # 读取用户 ID 列
    return sorted(df["user_id:token"].astype(str).unique().tolist())  # 返回去重排序后的用户列表


def _history_paths(eval_split: str) -> list[Path]:  # 根据评估划分返回构建历史所需交互文件
    if eval_split == "valid":  # 验证集仅需训练历史
        return [TRAIN_INTER]  # 返回训练集路径
    return [TRAIN_INTER, VALID_INTER]  # 测试集需训练加验证历史


def default_output_path(channel: ChannelName, eval_split: str) -> Path:  # 生成默认召回 CSV 输出路径
    return OUTPUT_DIR / f"{channel}_{eval_split}.csv"  # 返回 {channel}_{split}.csv 路径


def _write_recall_csv(  # 将召回结果写入 CSV 文件
    output_path: Path,  # 输出文件路径
    channel: str,  # 召回通道名
    eval_split: str,  # 评估划分名称
    rows: list[tuple[str, str, float, int]],  # 召回行：(用户, 商品, 分数, 排名)
) -> None:  # 无返回值
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    with output_path.open("w", newline="", encoding="utf-8") as f:  # 以 UTF-8 打开输出文件
        writer = csv.writer(f)  # 创建 CSV 写入器
        writer.writerow(["user_id", "item_id", "score", "rank", "channel"])  # 写入表头
        for user_id, item_id, score, rank in rows:  # 遍历每条召回记录
            writer.writerow([user_id, item_id, score, rank, channel])  # 写入一行召回结果
    print(f"Saved {channel} recall ({eval_split}): {output_path} ({len(rows):,} rows)")  # 打印保存信息


def export_popular_recall(  # 导出热门召回 CSV
    eval_split: str,  # 评估划分：valid 或 test
    *,  # 以下为仅关键字参数
    top_k: int = POPULAR_RECALL_TOP_K,  # 召回数量上限
    output_path: Path | None = None,  # 可选输出路径
) -> Path:  # 返回输出文件路径
    history_paths = _history_paths(eval_split)  # 获取历史交互文件列表
    for path in history_paths:  # 校验历史文件是否存在
        if not path.exists():  # 若文件缺失
            raise FileNotFoundError(f"Missing history file: {path}")  # 抛出异常

    popular_index = build_popular_index(*history_paths)  # 构建时间衰减热门索引
    user_history_map = build_user_history(*history_paths)  # 构建用户历史映射
    eval_users = _load_eval_users(eval_split)  # 加载评估用户列表
    output_path = output_path or default_output_path("popular", eval_split)  # 解析输出路径

    rows: list[tuple[str, str, float, int]] = []  # 初始化召回结果行
    global_cands = recall_popular(popular_index, user_history=set(), top_k=top_k)  # 全局热门候选（兜底）
    for user_id in eval_users:  # 遍历每个评估用户
        history_set = set(user_history_map.get(user_id, []))  # 获取用户历史商品集合
        cands = recall_popular(popular_index, user_history=history_set, top_k=top_k) or global_cands  # 召回或回退全局热门
        for rank, (item_id, score) in enumerate(cands, start=1):  # 遍历候选并编号
            rows.append((user_id, str(item_id), float(score), rank))  # 追加一行召回记录

    _write_recall_csv(output_path, "popular", eval_split, rows)  # 写入 CSV
    print(f"Users: {len(eval_users):,}, index size: {len(popular_index):,}")  # 打印用户与索引规模
    return output_path  # 返回输出路径


def export_category_popular_recall(  # 导出类别热门召回 CSV
    eval_split: str,  # 评估划分：valid 或 test
    *,  # 以下为仅关键字参数
    top_k: int = CATEGORY_POPULAR_RECALL_TOP_K,  # 召回数量上限
    seed_items: int = CATEGORY_SEED_ITEMS,  # 种子商品数量
    output_path: Path | None = None,  # 可选输出路径
) -> Path:  # 返回输出文件路径
    history_paths = _history_paths(eval_split)  # 获取历史交互文件列表
    for path in history_paths:  # 校验历史文件是否存在
        if not path.exists():  # 若文件缺失
            raise FileNotFoundError(f"Missing history file: {path}")  # 抛出异常

    index = build_category_popular_index(history_paths)  # 构建类别热门索引
    user_history_map = build_user_history(*history_paths)  # 构建用户历史映射
    eval_users = _load_eval_users(eval_split)  # 加载评估用户列表
    output_path = output_path or default_output_path("category_popular", eval_split)  # 解析输出路径

    rows: list[tuple[str, str, float, int]] = []  # 初始化召回结果行
    for user_id in eval_users:  # 遍历每个评估用户
        history = user_history_map.get(user_id, [])  # 获取用户历史商品列表
        cands = recall_category_popular(history, index, seed_items=seed_items, top_k=top_k)  # 执行类别热门召回
        for rank, (item_id, score) in enumerate(cands, start=1):  # 遍历候选并编号
            rows.append((user_id, str(item_id), float(score), rank))  # 追加一行召回记录

    _write_recall_csv(output_path, "category_popular", eval_split, rows)  # 写入 CSV
    bucket_count = sum(len(values) for values in index.buckets.values())  # 统计类别桶数量
    print(f"Users: {len(eval_users):,}, category buckets: {bucket_count:,}")  # 打印用户与桶规模
    return output_path  # 返回输出路径


def export_item2item_recall(  # 导出商品共现召回 CSV
    eval_split: str,  # 评估划分：valid 或 test
    *,  # 以下为仅关键字参数
    top_k: int = ITEM2ITEM_RECALL_TOP_K,  # 召回数量上限
    seed_items: int = ITEM2ITEM_SEED_ITEMS,  # 种子商品数量
    cooccur_weeks: int = COOCCUR_WEEKS,  # 共现统计窗口周数
    top_sim_k: int = TOP_SIM_K,  # 每个商品保留的邻居数
    output_path: Path | None = None,  # 可选输出路径
) -> Path:  # 返回输出文件路径
    history_paths = _history_paths(eval_split)  # 获取历史交互文件列表
    for path in history_paths:  # 校验历史文件是否存在
        if not path.exists():  # 若文件缺失
            raise FileNotFoundError(f"Missing history file: {path}")  # 抛出异常

    index = build_item2item_index(  # 构建商品共现索引
        history_paths,  # 历史交互文件列表
        cooccur_weeks=cooccur_weeks,  # 共现窗口周数
        top_sim_k=top_sim_k,  # 邻居保留数量
    )  # 索引构建结束
    user_history_map = build_user_history(*history_paths)  # 构建用户历史映射
    eval_users = _load_eval_users(eval_split)  # 加载评估用户列表
    output_path = output_path or default_output_path("item2item", eval_split)  # 解析输出路径

    rows: list[tuple[str, str, float, int]] = []  # 初始化召回结果行
    for user_id in eval_users:  # 遍历每个评估用户
        history = user_history_map.get(user_id, [])  # 获取用户历史商品列表
        cands = recall_item2item(history, index, seed_items=seed_items, top_k=top_k)  # 执行商品共现召回
        for rank, (item_id, score) in enumerate(cands, start=1):  # 遍历候选并编号
            rows.append((user_id, str(item_id), float(score), rank))  # 追加一行召回记录

    _write_recall_csv(output_path, "item2item", eval_split, rows)  # 写入 CSV
    print(f"Users: {len(eval_users):,}, index size: {len(index):,}")  # 打印用户与索引规模
    return output_path  # 返回输出路径


_EXPORTERS = {  # 通道名到导出函数的映射
    "popular": export_popular_recall,  # 热门召回导出
    "category_popular": export_category_popular_recall,  # 类别热门召回导出
    "item2item": export_item2item_recall,  # 商品共现召回导出
}  # 导出函数映射结束


def export_rule_recalls(  # 批量导出所选规则召回通道
    eval_split: str = "valid",  # 评估划分：valid 或 test
    channels: tuple[ChannelName, ...] = ALL_CHANNELS,  # 要导出的通道列表
    top_k: int | None = None,  # 可选统一 Top-K 覆盖
) -> dict[str, Path]:  # 返回通道名到输出路径的映射
    """Export selected rule-based recall channels for one eval split."""  # 导出指定评估划分的所选规则召回通道
    if eval_split not in {"valid", "test"}:  # 校验评估划分名称
        raise ValueError("eval_split must be 'valid' or 'test'")  # 非法划分则报错

    outputs: dict[str, Path] = {}  # 初始化输出路径字典
    for channel in channels:  # 遍历每个通道
        if channel not in _EXPORTERS:  # 若通道名未知
            raise ValueError(f"Unknown channel: {channel}")  # 抛出异常
        kwargs: dict = {"eval_split": eval_split}  # 构建导出参数字典
        if top_k is not None:  # 若指定了 Top-K 覆盖
            kwargs["top_k"] = top_k  # 写入 Top-K 参数
        outputs[channel] = _EXPORTERS[channel](**kwargs)  # 调用对应导出函数
    return outputs  # 返回各通道输出路径


def main() -> None:  # 命令行入口函数
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description="Export Popular / Category Popular / Item2Item recall CSV files.",  # 程序描述
    )  # 参数解析器创建结束
    parser.add_argument(  # 添加评估划分参数
        "--eval-split",  # 参数名
        choices=["valid", "test", "both"],  # 可选值
        default="both",  # 默认导出 valid 与 test
        help="Which eval users to export recall for (default: both)",  # 帮助文本
    )  # 评估划分参数定义结束
    parser.add_argument(  # 添加通道选择参数
        "--channels",  # 参数名
        type=str,  # 字符串类型
        default="all",  # 默认导出全部通道
        help="Comma-separated: popular,category_popular,item2item or 'all' (default: all)",  # 帮助文本
    )  # 通道选择参数定义结束
    parser.add_argument(  # 添加 Top-K 覆盖参数
        "--top-k",  # 参数名
        type=int,  # 整数类型
        default=None,  # 默认使用各通道常量
        help="Override recall top-k for all channels (default: each channel's constant, usually 50)",  # 帮助文本
    )  # Top-K 参数定义结束
    args = parser.parse_args()  # 解析命令行参数

    if args.channels.strip().lower() == "all":  # 若选择全部通道
        channels: tuple[ChannelName, ...] = ALL_CHANNELS  # 使用全部通道列表
    else:  # 否则解析逗号分隔的通道名
        channels = tuple(ch.strip() for ch in args.channels.split(",") if ch.strip())  # 解析逗号分隔的通道名 # type: ignore[assignment]
        for ch in channels:  # 校验每个通道名
            if ch not in _EXPORTERS:  # 若通道未知
                raise ValueError(f"Unknown channel: {ch}. Choose from: {', '.join(ALL_CHANNELS)}")  # 抛出异常

    splits = ["valid", "test"] if args.eval_split == "both" else [args.eval_split]  # 确定要导出的划分列表
    started = time.perf_counter()  # 记录开始时间

    for split in splits:  # 遍历每个评估划分
        print(f"\n{'=' * 60}")  # 打印分隔线
        print(f"Eval split: {split}")  # 打印当前划分
        print("=" * 60)  # 打印分隔线
        export_rule_recalls(eval_split=split, channels=channels, top_k=args.top_k)  # 导出所选通道

    elapsed = time.perf_counter() - started  # 计算总耗时
    print(f"\nAll rule-based recalls finished in {elapsed:.1f}s")  # 打印总耗时
    print(f"Output directory: {OUTPUT_DIR.resolve()}")  # 打印输出目录绝对路径


if __name__ == "__main__":  # 脚本直接运行入口
    main()  # 执行主函数
