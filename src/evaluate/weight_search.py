"""Valid-set grid search for per-tier multi-channel fusion weights."""  # 验证集上按活跃度分层搜索多通道融合权重

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import copy  # 导入深拷贝模块
import json  # 导入 JSON 序列化模块
import sys  # 导入系统模块用于路径注入
from itertools import product  # 导入笛卡尔积生成器
from pathlib import Path  # 导入路径处理类
from typing import Any  # 导入任意类型注解

if __package__ is None or __package__ == "":  # 若以脚本方式直接运行
    project_root = Path(__file__).resolve().parents[2]  # 定位项目根目录
    if str(project_root) not in sys.path:  # 若根目录不在搜索路径中
        sys.path.insert(0, str(project_root))  # 注入项目根目录到 sys.path

from src.fusion.weighted_fusion import ACTIVITY_WEIGHTS, ActivityTier  # 导入默认权重表与活跃度分层类型
from src.evaluate.offline_eval import (  # 导入离线评估相关组件
    EVAL_OUT_DIR,  # 评估指标输出目录
    FusionEvalContext,  # 融合评估上下文
    build_fusion_eval_context,  # 构建融合评估上下文
    evaluate_fusion_map_at_k,  # 计算给定权重的平均 MAP@K
)  # 离线评估模块导入结束

DEFAULT_OUTPUT_JSON = EVAL_OUT_DIR / "best_fusion_weights.json"  # 最优权重 JSON 默认输出路径

# 各活跃度分层、各通道权重搜索范围（sequence 即 sasrecf）
TIER_WEIGHT_RANGES: dict[ActivityTier, dict[str, tuple[float, float]]] = {  # 各分层各通道权重搜索区间
    "high": {  # 高活跃分层
        "sequence": (0.45, 0.70),  # 序列模型通道权重范围
        "popular": (0.05, 0.25),  # 热门通道权重范围
        "category_popular": (0.05, 0.25),  # 类别热门通道权重范围
        "item2item": (0.10, 0.30),  # 商品共现通道权重范围
    },  # 高活跃分层范围结束
    "medium": {  # 中活跃分层
        "sequence": (0.30, 0.55),  # 序列模型通道权重范围
        "popular": (0.15, 0.40),  # 热门通道权重范围
        "category_popular": (0.10, 0.30),  # 类别热门通道权重范围
        "item2item": (0.10, 0.30),  # 商品共现通道权重范围
    },  # 中活跃分层范围结束
    "low": {  # 低活跃分层
        "sequence": (0.00, 0.25),  # 序列模型通道权重范围
        "popular": (0.40, 0.75),  # 热门通道权重范围
        "category_popular": (0.10, 0.35),  # 类别热门通道权重范围
        "item2item": (0.00, 0.20),  # 商品共现通道权重范围
    },  # 低活跃分层范围结束
    "cold_start": {  # 冷启动分层
        "sequence": (0.00, 0.00),  # 序列模型通道权重范围（固定为 0）
        "popular": (0.60, 0.90),  # 热门通道权重范围
        "category_popular": (0.10, 0.40),  # 类别热门通道权重范围
        "item2item": (0.00, 0.00),  # 商品共现通道权重范围（固定为 0）
    },  # 冷启动分层范围结束
}  # 搜索区间字典结束

CHANNEL_KEYS = ("sequence", "popular", "category_popular", "item2item")  # 四通道权重键名元组


def _grid_values(low: float, high: float, step: float) -> list[float]:  # 在区间内按步长生成网格值
    if low == high:  # 若上下界相等
        return [round(low, 4)]  # 返回单值列表
    values: list[float] = []  # 初始化网格值列表
    current = low  # 从下限开始
    while current <= high + 1e-9:  # 遍历至上限（含浮点容差）
        values.append(round(current, 4))  # 四舍五入后加入列表
        current += step  # 步进
    return values  # 返回网格值列表


def _normalize_tier_weights(raw: dict[str, float]) -> dict[str, float]:  # 将分层原始权重归一化为和为 1
    total = sum(raw[k] for k in CHANNEL_KEYS)  # 计算四通道权重之和
    if total <= 0:  # 若总和非正
        raise ValueError(f"Invalid tier weights (sum <= 0): {raw}")  # 抛出无效权重异常
    return {k: raw[k] / total for k in CHANNEL_KEYS}  # 按总和归一化并返回


def _in_range(value: float, bounds: tuple[float, float], tol: float = 1e-6) -> bool:  # 判断值是否在区间内（含容差）
    return bounds[0] - tol <= value <= bounds[1] + tol  # 返回是否在范围内


def generate_weight_candidates(  # 为单个活跃度分层生成归一化权重候选
    tier: ActivityTier,  # 目标活跃度分层
    step: float = 0.05,  # 网格搜索步长
) -> list[dict[str, float]]:  # 返回权重候选列表
    """Generate normalized weight tuples for one activity tier within search ranges."""  # 在搜索范围内为单分层生成归一化权重元组
    ranges = TIER_WEIGHT_RANGES[tier]  # 取该分层的搜索区间
    seq_vals = _grid_values(ranges["sequence"][0], ranges["sequence"][1], step)  # 序列通道网格值
    pop_vals = _grid_values(ranges["popular"][0], ranges["popular"][1], step)  # 热门通道网格值
    cat_vals = _grid_values(ranges["category_popular"][0], ranges["category_popular"][1], step)  # 类别热门通道网格值
    i2i_vals = _grid_values(ranges["item2item"][0], ranges["item2item"][1], step)  # 商品共现通道网格值

    # 冷启动：sasrecf 与 item2item 固定为 0，popular + category_popular = 1
    if tier == "cold_start":  # 冷启动分层特殊处理
        candidates: list[dict[str, float]] = []  # 初始化候选列表
        for pop in pop_vals:  # 遍历热门通道网格值
            cat = round(1.0 - pop, 4)  # 类别热门权重补足为 1
            if not _in_range(cat, ranges["category_popular"]):  # 若类别热门权重超出范围
                continue  # 跳过该组合
            candidates.append(  # 追加冷启动权重候选
                {  # 权重字典
                    "sequence": 0.0,  # 序列模型权重固定为 0
                    "popular": pop,  # 热门通道权重
                    "category_popular": cat,  # 类别热门通道权重
                    "item2item": 0.0,  # 商品共现权重固定为 0
                }  # 权重字典结束
            )  # 追加完成
        return candidates  # 返回冷启动候选列表

    candidates = []  # 初始化通用候选列表
    for seq, pop, cat, i2i in product(seq_vals, pop_vals, cat_vals, i2i_vals):  # 四通道笛卡尔积遍历
        raw = {"sequence": seq, "popular": pop, "category_popular": cat, "item2item": i2i}  # 组装原始权重
        total = sum(raw.values())  # 计算权重之和
        if abs(total - 1.0) > 1e-4:  # 若总和偏离 1 超过容差
            continue  # 跳过该组合
        if all(_in_range(raw[k], ranges[k]) for k in CHANNEL_KEYS):  # 若四通道均在搜索范围内
            candidates.append(raw)  # 加入候选列表

    if candidates:  # 若已有严格 sum=1 的候选
        return candidates  # 直接返回

    # 若严格 sum=1 无结果，退化为生成后归一化并校验范围
    for seq, pop, cat in product(seq_vals, pop_vals, cat_vals):  # 三通道笛卡尔积，item2item 由余量计算
        i2i = round(1.0 - seq - pop - cat, 4)  # 计算 item2item 余量权重
        if i2i < -1e-6:  # 若余量为负
            continue  # 跳过该组合
        raw = {"sequence": seq, "popular": pop, "category_popular": cat, "item2item": max(i2i, 0.0)}  # 组装原始权重
        normalized = _normalize_tier_weights(raw)  # 归一化权重
        if all(_in_range(normalized[k], ranges[k]) for k in CHANNEL_KEYS):  # 校验归一化后是否在范围内
            candidates.append(normalized)  # 加入候选列表
    return candidates  # 返回候选列表


def save_best_weights(payload: dict[str, Any], output_path: Path | None = None) -> Path:  # 保存最优权重到 JSON 文件
    output_path = output_path or DEFAULT_OUTPUT_JSON  # 使用默认或指定输出路径
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入 JSON 文件
    return output_path  # 返回输出路径


def load_best_weights(path: str | Path) -> dict[str, Any]:  # 从 JSON 文件加载最优权重
    path = Path(path)  # 转为 Path 对象
    data = json.loads(path.read_text(encoding="utf-8"))  # 读取并解析 JSON
    best = data.get("best_weights", data)  # 取 best_weights 字段或整个数据
    activity_weights: dict[ActivityTier, dict[str, float]] = {}  # 初始化分层权重字典
    for tier in ("high", "medium", "low", "cold_start"):  # 遍历四个活跃度分层
        if tier not in best:  # 若缺少某分层
            raise KeyError(f"Missing tier '{tier}' in weights file: {path}")  # 抛出键缺失异常
        activity_weights[tier] = {k: float(best[tier][k]) for k in CHANNEL_KEYS}  # 解析该分层四通道权重
    data["best_weights"] = activity_weights  # 将解析后的权重写回数据
    return data  # 返回完整数据


def search_best_weights(  # 坐标下降式网格搜索最优分层权重
    context: FusionEvalContext,  # 预计算的融合评估上下文
    step: float = 0.05,  # 网格搜索步长
    exclude_seen: bool = False,  # 融合时是否排除历史已购
    max_passes: int = 2,  # 最大坐标下降轮数
    verbose: bool = True,  # 是否打印搜索过程
) -> tuple[dict[ActivityTier, dict[str, float]], float]:  # 返回最优权重与对应 MAP@12
    """  # 函数文档字符串开始
    Coordinate-descent grid search: optimize each tier's weights on valid MAP@12.  # 坐标下降网格搜索：在验证集 MAP@12 上优化各分层权重

    Only fusion weights change; recall candidates are fixed in context.  # 仅融合权重变化，召回候选在上下文中固定
    """  # 坐标下降网格搜索：在固定召回候选下优化各分层权重以最大化验证集 MAP@12
    best_weights = copy.deepcopy(ACTIVITY_WEIGHTS)  # 以默认权重为搜索起点
    best_map = evaluate_fusion_map_at_k(context, best_weights, exclude_seen=exclude_seen)  # 计算基线 MAP@12
    if verbose:  # 若启用详细输出
        print(f"Baseline MAP@12={best_map:.6f} exclude_seen={exclude_seen}")  # 打印基线 MAP
        print(f"Baseline weights: {json.dumps(best_weights, ensure_ascii=False)}")  # 打印基线权重

    tier_order: list[ActivityTier] = ["high", "medium", "low", "cold_start"]  # 分层优化顺序

    for pass_idx in range(1, max_passes + 1):  # 遍历每一轮坐标下降
        if verbose:  # 若启用详细输出
            print(f"\n--- Pass {pass_idx}/{max_passes} ---")  # 打印当前轮次
        improved_any = False  # 标记本轮是否有改进
        for tier in tier_order:  # 逐分层优化
            candidates = generate_weight_candidates(tier, step=step)  # 生成该分层权重候选
            if verbose:  # 若启用详细输出
                print(f"Tier {tier}: {len(candidates)} candidate weight sets")  # 打印候选数量
            for candidate in candidates:  # 遍历每个权重候选
                trial_weights = copy.deepcopy(best_weights)  # 复制当前最优权重
                trial_weights[tier] = candidate  # 替换当前分层的权重
                trial_map = evaluate_fusion_map_at_k(  # 评估试用权重
                    context, trial_weights, exclude_seen=exclude_seen  # 传入上下文与排除已购标志
                )  # 评估完成
                if trial_map > best_map + 1e-9:  # 若 MAP 有显著提升
                    best_map = trial_map  # 更新最优 MAP
                    best_weights[tier] = candidate  # 更新该分层最优权重
                    improved_any = True  # 标记本轮有改进
                    if verbose:  # 若启用详细输出
                        print(  # 打印新的最优结果
                            f">> New best MAP@12={best_map:.6f} "  # 新的最优 MAP
                            f"tier={tier} "  # 改进的分层
                            f"trial_weights={json.dumps(candidate, ensure_ascii=False)} "  # 试用权重
                            f"all_tiers={json.dumps(best_weights, ensure_ascii=False)}"  # 全部层级权重
                        )  # 打印结束
        if not improved_any:  # 若本轮无任何改进
            if verbose:  # 若启用详细输出
                print("No improvement in this pass; stopping early.")  # 提示提前停止
            break  # 跳出坐标下降循环

    if verbose:  # 若启用详细输出
        print(f"\nFinal best MAP@12={best_map:.6f} exclude_seen={exclude_seen}")  # 打印最终最优 MAP
        print(f"Final best weights: {json.dumps(best_weights, ensure_ascii=False)}")  # 打印最终最优权重
    return best_weights, best_map  # 返回最优权重与 MAP


def run_weight_search(  # 运行完整权重搜索流程并保存结果
    eval_split: str = "valid",  # 评估划分（仅允许 valid）
    step: float = 0.05,  # 网格搜索步长
    max_passes: int = 2,  # 坐标下降最大轮数
    output_json: Path | None = None,  # 可选输出 JSON 路径
    sasrec_recall_csv: str | Path | None = None,  # 可选序列模型召回 CSV
    compare_exclude_seen: bool = True,  # 是否对比 exclude_seen 两种模式
    verbose: bool = True,  # 是否打印详细日志
) -> dict[str, Any]:  # 返回搜索结果载荷
    if eval_split != "valid":  # 权重搜索仅允许验证集
        raise ValueError("Weight search is only allowed on eval_split='valid'")  # 抛出非法划分异常

    if verbose:  # 若启用详细输出
        print("Building fusion eval context (recall computed once)...")  # 提示正在构建评估上下文
    context = build_fusion_eval_context(  # 构建融合评估上下文（召回只算一次）
        eval_split="valid",  # 固定使用验证集
        sasrec_recall_csv=sasrec_recall_csv,  # 传入序列模型召回 CSV
    )  # 上下文构建完成

    modes = [False, True] if compare_exclude_seen else [False]  # 确定要搜索的 exclude_seen 模式列表
    mode_results: dict[str, dict[str, Any]] = {}  # 初始化各模式搜索结果

    for exclude_seen in modes:  # 遍历每种 exclude_seen 模式
        label = "exclude_seen=true" if exclude_seen else "exclude_seen=false"  # 生成模式标签
        if verbose:  # 若启用详细输出
            print(f"\n========== Search mode: {label} ==========")  # 打印当前搜索模式
        best_weights, best_map = search_best_weights(  # 执行权重搜索
            context,  # 传入评估上下文
            step=step,  # 传入网格步长
            exclude_seen=exclude_seen,  # 传入排除已购标志
            max_passes=max_passes,  # 传入最大轮数
            verbose=verbose,  # 传入详细输出标志
        )  # 搜索完成
        mode_results[label] = {  # 记录该模式结果
            "exclude_seen": exclude_seen,  # 排除已购标志
            "best_map@12": best_map,  # 最优 MAP@12
            "best_weights": best_weights,  # 最优权重
        }  # 模式结果记录结束

    selected_label = max(mode_results, key=lambda k: mode_results[k]["best_map@12"])  # 选取 MAP 最高的模式
    selected = mode_results[selected_label]  # 取选中模式的结果

    payload: dict[str, Any] = {  # 组装输出载荷
        "protocol": "hm_fusion_weight_search",  # 协议标识
        "eval_split": "valid",  # 评估划分
        "sequence_channel": context.sequence_channel,  # 序列模型通道名
        "search_step": step,  # 搜索步长
        "max_passes": max_passes,  # 最大坐标下降轮数
        "exclude_seen": selected["exclude_seen"],  # 选中的排除已购模式
        "best_map@12": selected["best_map@12"],  # 选中模式的最优 MAP@12
        "best_weights": {  # 选中模式的最优权重
            tier: {k: float(selected["best_weights"][tier][k]) for k in CHANNEL_KEYS}  # 各分层四通道权重
            for tier in ("high", "medium", "low", "cold_start")  # 遍历四个分层
        },  # 最优权重结束
        "compared_exclude_seen": {  # 各 exclude_seen 模式对比结果
            label: {  # 单模式结果
                "exclude_seen": res["exclude_seen"],  # 排除已购标志
                "best_map@12": res["best_map@12"],  # 该模式最优 MAP@12
                "best_weights": {  # 该模式最优权重
                    tier: {k: float(res["best_weights"][tier][k]) for k in CHANNEL_KEYS}  # 各分层四通道权重
                    for tier in ("high", "medium", "low", "cold_start")  # 遍历四个分层
                },  # 最优权重结束
            }  # 单模式结果结束
            for label, res in mode_results.items()  # 遍历所有模式
        },  # 对比结果结束
        "selected_mode": selected_label,  # 最终选中的模式标签
    }  # 载荷组装结束

    out_path = save_best_weights(payload, output_json)  # 保存最优权重到 JSON
    if verbose:  # 若启用详细输出
        print(f"\nSaved best weights: {out_path}")  # 打印保存路径
        print(json.dumps(payload, ensure_ascii=False, indent=2))  # 打印完整载荷
    return payload  # 返回搜索结果载荷


def main() -> None:  # 命令行入口函数
    parser = argparse.ArgumentParser(description="Search fusion weights on valid MAP@12")  # 创建参数解析器
    parser.add_argument("--eval-split", choices=["valid"], default="valid")  # 评估划分参数（仅 valid）
    parser.add_argument("--step", type=float, default=0.05, help="Grid step (default: 0.05)")  # 网格步长参数
    parser.add_argument("--max-passes", type=int, default=2, help="Coordinate descent passes")  # 坐标下降轮数参数
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)  # 输出 JSON 路径参数
    parser.add_argument("--sasrec-recall-csv", type=Path, default=None)  # 可选序列模型召回 CSV 参数
    parser.add_argument(  # exclude_seen 搜索模式参数
        "--exclude-seen-only",  # 参数名
        choices=["both", "false", "true"],  # 可选值
        default="both",  # 默认同时搜索两种模式
        help="Search with exclude_seen=false, true, or both (default: both)",  # 帮助文本
    )  # exclude_seen 模式参数结束
    args = parser.parse_args()  # 解析命令行参数

    compare = args.exclude_seen_only == "both"  # 是否对比两种 exclude_seen 模式
    if args.exclude_seen_only == "false":  # 若仅搜索 exclude_seen=false
        compare = False  # 关闭模式对比
        # force only false - handled in run_weight_search by passing compare_exclude_seen=False
    elif args.exclude_seen_only == "true":  # 若仅搜索 exclude_seen=true
        # need custom single mode true - extend run_weight_search
        pass  # 下方单独处理 true 模式

    if args.exclude_seen_only == "true":  # 仅搜索 exclude_seen=true 模式
        context = build_fusion_eval_context(eval_split="valid", sasrec_recall_csv=args.sasrec_recall_csv)  # 构建评估上下文
        best_weights, best_map = search_best_weights(  # 执行权重搜索
            context, step=args.step, exclude_seen=True, max_passes=args.max_passes  # 固定排除已购
        )  # 搜索完成
        payload = {  # 组装输出载荷
            "protocol": "hm_fusion_weight_search",  # 协议标识
            "eval_split": "valid",  # 评估划分
            "sequence_channel": context.sequence_channel,  # 序列模型通道名
            "search_step": args.step,  # 搜索步长
            "exclude_seen": True,  # 排除已购标志
            "best_map@12": best_map,  # 最优 MAP@12
            "best_weights": {  # 最优权重
                tier: {k: float(best_weights[tier][k]) for k in CHANNEL_KEYS}  # 各分层四通道权重
                for tier in ("high", "medium", "low", "cold_start")  # 遍历四个分层
            },  # 最优权重结束
            "selected_mode": "exclude_seen=true",  # 选中模式标签
        }  # 载荷组装结束
        save_best_weights(payload, args.output_json)  # 保存最优权重
        print(json.dumps(payload, ensure_ascii=False, indent=2))  # 打印载荷
        return  # 结束主函数

    if args.exclude_seen_only == "false":  # 仅搜索 exclude_seen=false 模式
        context = build_fusion_eval_context(eval_split="valid", sasrec_recall_csv=args.sasrec_recall_csv)  # 构建评估上下文
        best_weights, best_map = search_best_weights(  # 执行权重搜索
            context, step=args.step, exclude_seen=False, max_passes=args.max_passes  # 不排除已购
        )  # 搜索完成
        payload = {  # 组装输出载荷
            "protocol": "hm_fusion_weight_search",  # 协议标识
            "eval_split": "valid",  # 评估划分
            "sequence_channel": context.sequence_channel,  # 序列模型通道名
            "search_step": args.step,  # 搜索步长
            "exclude_seen": False,  # 排除已购标志
            "best_map@12": best_map,  # 最优 MAP@12
            "best_weights": {  # 最优权重
                tier: {k: float(best_weights[tier][k]) for k in CHANNEL_KEYS}  # 各分层四通道权重
                for tier in ("high", "medium", "low", "cold_start")  # 遍历四个分层
            },  # 最优权重结束
            "selected_mode": "exclude_seen=false",  # 选中模式标签
        }  # 载荷组装结束
        save_best_weights(payload, args.output_json)  # 保存最优权重
        print(json.dumps(payload, ensure_ascii=False, indent=2))  # 打印载荷
        return  # 结束主函数

    run_weight_search(  # 默认：对比两种 exclude_seen 模式并选取最优
        eval_split=args.eval_split,  # 传入评估划分
        step=args.step,  # 传入网格步长
        max_passes=args.max_passes,  # 传入最大轮数
        output_json=args.output_json,  # 传入输出路径
        sasrec_recall_csv=args.sasrec_recall_csv,  # 传入序列模型召回 CSV
        compare_exclude_seen=True,  # 启用两种模式对比
    )  # 权重搜索流程结束


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 执行主函数
