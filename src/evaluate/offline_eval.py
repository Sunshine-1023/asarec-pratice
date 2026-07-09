"""Run multi-channel fusion and offline evaluation."""  # 运行多通道融合与离线评估

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import copy  # 深拷贝权重模板
import csv  # 导入 CSV 读写模块
import json  # 导入 JSON 序列化模块
import math  # 导入数学函数库
from collections import defaultdict  # 导入带默认值的字典
from dataclasses import dataclass  # 融合评估上下文
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas 数据分析库

from src.fusion.weighted_fusion import (  # 导入融合相关函数
    ACTIVITY_WEIGHTS,  # 默认活跃度权重模板
    ActivityTier,  # 活跃度分层类型
    build_user_history,  # 构建用户历史映射
    classify_activity_tier,  # 按历史长度分类活跃度
    fuse_candidates,  # 多通道候选融合
    get_channel_weights_for_user,  # 获取用户自适应通道权重
    infer_sequence_channel,  # 从文件名推断序列通道名
    load_channel_recall_csv,  # 加载通道召回 CSV
)  # 融合模块导入结束
from src.recall.category_popular import (  # 类别热门召回
    CATEGORY_POPULAR_RECALL_TOP_K,  # 类别热门召回 Top-K 常量
    SEED_ITEMS as CATEGORY_SEED_ITEMS,  # 类别热门种子商品数常量
    build_category_popular_index,  # 构建类别热门索引
    recall_category_popular,  # 执行类别热门召回
)  # 类别热门模块导入结束
from src.recall.item2item import (  # item2item 共现召回
    COOCCUR_WEEKS,  # 共现统计窗口（周）
    ITEM2ITEM_RECALL_TOP_K,  # item2item 召回 Top-K 常量
    SEED_ITEMS,  # 种子商品数常量
    TOP_SIM_K,  # 每个商品保留相似邻居数
    build_item2item_index,  # 构建 item2item 共现索引
    recall_item2item,  # 执行 item2item 召回
)  # item2item 模块导入结束
from src.recall.popular import POPULAR_RECALL_TOP_K, build_popular_index, recall_popular  # 导入热门召回函数


TRAIN_INTER = Path("data/processed/hm/hm.train.inter")  # 训练集交互文件路径
VALID_INTER = Path("data/processed/hm/hm.valid.inter")  # 验证集交互文件路径
TEST_INTER = Path("data/processed/hm/hm.test.inter")  # 测试集交互文件路径

SASREC_RECALL_DIR = Path("outputs/recommendations")  # SASRec 召回结果目录
FUSION_OUT_DIR = Path("outputs/recommendations")  # 融合推荐输出目录
EVAL_OUT_DIR = Path("outputs/evaluation")  # 评估指标输出目录


def default_sasrec_recall_csv(eval_split: str, prefer_sasrecf: bool = True) -> Path:  # 返回默认序列模型召回 CSV
    if prefer_sasrecf:  # 优先 SASRecF
        sasrecf_path = SASREC_RECALL_DIR / f"sasrecf_{eval_split}.csv"  # 构造 SASRecF 召回文件路径
        if sasrecf_path.exists():  # 若 SASRecF 文件存在
            return sasrecf_path  # 返回 SASRecF 路径
    return SASREC_RECALL_DIR / f"sasrec_{eval_split}.csv"  # 回退 SASRec


def _load_targets(path: Path) -> dict[str, set[str]]:  # 加载评估集真实标签
    df = pd.read_csv(path, sep="\t", usecols=["user_id:token", "item_id:token"])  # 读取用户与物品列
    grouped = (  # 按用户聚合真实物品集合
        df.groupby("user_id:token")["item_id:token"]  # 按用户分组并取物品列
        .apply(lambda s: {str(x) for x in s.tolist()})  # 将每组物品转为字符串集合
        .to_dict()  # 转为字典
    )  # 结束标签聚合
    return grouped  # 返回用户到真实物品集合的映射


def _recall_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 计算单用户 Recall@K
    if not actual:  # 若无真实标签
        return 0.0  # 返回 0
    return len(set(pred[:k]) & actual) / len(actual)  # 命中数除以真实物品总数


def _hit_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 计算单用户 Hit@K
    return 1.0 if set(pred[:k]) & actual else 0.0  # 有命中返回 1.0，否则返回 0.0


def _ndcg_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 计算单用户 NDCG@K
    dcg = 0.0  # 初始化折损累积增益
    for i, item in enumerate(pred[:k]):  # 遍历前 K 个预测物品
        if item in actual:  # 若该物品在真实标签中
            dcg += 1.0 / (math.log2(i + 2))  # 累加位置折损增益
    ideal_hits = min(len(actual), k)  # 理想命中数取真实数与 K 的较小值
    if ideal_hits == 0:  # 若无理想命中
        return 0.0  # 返回 0
    idcg = sum(1.0 / (math.log2(i + 2)) for i in range(ideal_hits))  # 计算理想折损累积增益
    return dcg / idcg if idcg > 0 else 0.0  # 返回 NDCG 比值


def _map_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 计算单用户 MAP@K
    """Mean Average Precision at K: AP@K = sum(P@i * rel_i) / min(|actual|, K)."""  # MAP@K 为前 K 位平均精确率均值
    if not actual:  # 若无真实标签
        return 0.0  # 返回 0

    hits = 0  # 初始化累计命中数
    ap_sum = 0.0  # 初始化精确率累加和
    for i, item in enumerate(pred[:k], start=1):  # 遍历前 K 个预测，排名从 1 开始
        if item in actual:  # 若当前物品命中
            hits += 1  # 命中数加一
            ap_sum += hits / i  # 累加当前位置的精确率

    denom = min(len(actual), k)  # 分母取真实数与 K 的较小值
    return ap_sum / denom if denom > 0 else 0.0  # 返回平均精确率


def map_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 公开 MAP@K（与 offline_eval 一致）
    return _map_at_k(actual, pred, k)  # 复用内部实现


@dataclass
class FusionEvalContext:  # 预计算召回候选，供权重搜索复用
    targets: dict[str, set[str]]  # 用户真实标签
    users: list[dict]  # 每用户 history / channel_candidates
    sequence_channel: str  # 序列模型通道名
    final_top_k: int  # 最终 Top-K


def build_fusion_eval_context(  # 构建融合评估上下文（召回只算一次）
    eval_split: str = "valid",
    recall_top_k: int = 100,
    popular_recall_top_k: int = POPULAR_RECALL_TOP_K,
    category_popular_recall_top_k: int = CATEGORY_POPULAR_RECALL_TOP_K,
    item2item_recall_top_k: int = ITEM2ITEM_RECALL_TOP_K,
    item2item_cooccur_weeks: int = COOCCUR_WEEKS,
    item2item_top_sim_k: int = TOP_SIM_K,
    item2item_seed_items: int = SEED_ITEMS,
    category_popular_seed_items: int = CATEGORY_SEED_ITEMS,
    final_top_k: int = 12,
    sasrec_recall_csv: str | Path | None = None,
    sequence_channel: str | None = None,
) -> FusionEvalContext:
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")

    sasrec_recall_csv = (
        Path(sasrec_recall_csv)
        if sasrec_recall_csv is not None
        else default_sasrec_recall_csv(eval_split)
    )
    eval_path = VALID_INTER if eval_split == "valid" else TEST_INTER
    history_paths = [TRAIN_INTER] if eval_split == "valid" else [TRAIN_INTER, VALID_INTER]

    user_history_map = build_user_history(*history_paths)
    targets = _load_targets(eval_path)
    popular_index = build_popular_index(*history_paths)
    category_popular_index = build_category_popular_index(history_paths)
    item2item_index = build_item2item_index(
        history_paths,
        cooccur_weeks=item2item_cooccur_weeks,
        top_sim_k=item2item_top_sim_k,
    )
    sasrec_map = load_channel_recall_csv(sasrec_recall_csv)
    resolved_sequence_channel = sequence_channel or infer_sequence_channel(sasrec_recall_csv)

    users: list[dict] = []
    for user_id, actual_items in targets.items():
        history = user_history_map.get(user_id, [])
        history_set = set(history)
        channel_candidates = {
            "popular": recall_popular(popular_index, user_history=history_set, top_k=popular_recall_top_k),
            "category_popular": recall_category_popular(
                history,
                category_popular_index,
                seed_items=category_popular_seed_items,
                top_k=category_popular_recall_top_k,
            ),
            "item2item": recall_item2item(
                history,
                item2item_index,
                seed_items=item2item_seed_items,
                top_k=item2item_recall_top_k,
            ),
            resolved_sequence_channel: [
                (iid, score) for iid, score, _ in sasrec_map.get(user_id, [])[:recall_top_k]
            ],
        }
        users.append(
            {
                "user_id": user_id,
                "actual_items": actual_items,
                "history": history,
                "history_set": history_set,
                "channel_candidates": channel_candidates,
            }
        )

    return FusionEvalContext(
        targets=targets,
        users=users,
        sequence_channel=resolved_sequence_channel,
        final_top_k=final_top_k,
    )


def evaluate_fusion_map_at_k(  # 给定权重模板计算平均 MAP@K
    context: FusionEvalContext,
    activity_weights: dict[ActivityTier, dict[str, float]],
    exclude_seen: bool = False,
) -> float:
    maps: list[float] = []
    for row in context.users:
        user_weights = get_channel_weights_for_user(
            len(row["history"]),
            context.sequence_channel,
            activity_weights=activity_weights,
        )
        fused = fuse_candidates(
            user_id=row["user_id"],
            user_history=row["history_set"],
            channel_candidates=row["channel_candidates"],
            channel_weights=user_weights,
            top_k=context.final_top_k,
            exclude_seen=exclude_seen,
        )
        pred_items = [item_id for item_id, _ in fused]
        maps.append(_map_at_k(row["actual_items"], pred_items, context.final_top_k))
    return float(sum(maps) / len(maps)) if maps else 0.0


def evaluate_fusion(  # 执行多通道融合并评估
    eval_split: str = "valid",  # 评估划分：valid 或 test
    recall_top_k: int = 100,  # 序列模型召回 Top-K
    popular_recall_top_k: int = POPULAR_RECALL_TOP_K,  # 全局热门召回 Top-K
    category_popular_recall_top_k: int = CATEGORY_POPULAR_RECALL_TOP_K,  # 类别热门召回 Top-K
    final_top_k: int = 12,  # 融合后最终 Top-K
    popular_weight: float = 0.15,  # 固定权重：热门通道
    category_popular_weight: float = 0.15,  # 固定权重：类别热门通道
    item2item_weight: float = 0.25,  # 固定权重：item2item 通道
    sasrec_weight: float = 0.45,  # 固定权重：序列模型通道
    item2item_recall_top_k: int = ITEM2ITEM_RECALL_TOP_K,  # item2item 召回 Top-K
    item2item_cooccur_weeks: int = COOCCUR_WEEKS,  # item2item 共现统计窗口（周）
    item2item_top_sim_k: int = TOP_SIM_K,  # 每个商品保留相似邻居数
    item2item_seed_items: int = SEED_ITEMS,  # 种子商品数（最近 N 个购买）
    category_popular_seed_items: int = CATEGORY_SEED_ITEMS,  # 类别热门种子商品数
    sasrec_recall_csv: str | Path | None = None,  # 可选序列模型召回 CSV 路径
    adaptive_weights: bool = True,  # 是否按用户历史长度自适应权重
    activity_weights: dict[ActivityTier, dict[str, float]] | None = None,  # 自定义分层权重
    exclude_seen: bool = False,  # 融合时是否排除历史已购
    sequence_channel: str | None = None,  # 序列通道名（sasrec / sasrecf），默认从 CSV 推断
) -> tuple[Path, Path, dict[str, float]]:  # 返回推荐文件路径、指标文件路径与指标字典
    """Run multi-channel recall fusion and evaluate on valid/test split."""  # 在 valid/test 划分上运行多通道召回融合并评估
    if eval_split not in {"valid", "test"}:  # 校验评估划分参数
        raise ValueError("eval_split must be 'valid' or 'test'")  # 非法划分时抛出异常

    sasrec_recall_csv = (  # 确定 SASRec 召回文件路径
        Path(sasrec_recall_csv)  # 若用户提供路径则转为 Path
        if sasrec_recall_csv is not None  # 判断路径是否非空
        else default_sasrec_recall_csv(eval_split)  # 否则使用默认路径
    )  # 结束路径选择

    eval_path = VALID_INTER if eval_split == "valid" else TEST_INTER  # 选择验证或测试交互文件
    history_paths = [TRAIN_INTER] if eval_split == "valid" else [TRAIN_INTER, VALID_INTER]  # 选择构建历史所用的交互文件

    user_history_map = build_user_history(*history_paths)  # 构建用户历史映射
    targets = _load_targets(eval_path)  # 加载评估集真实标签

    popular_index = build_popular_index(*history_paths)  # 构建热门召回索引
    category_popular_index = build_category_popular_index(history_paths)  # 构建类别热门索引
    item2item_index = build_item2item_index(  # 构建 item2item 共现索引
        history_paths,  # 传入历史交互文件路径
        cooccur_weeks=item2item_cooccur_weeks,  # 共现统计窗口
        top_sim_k=item2item_top_sim_k,  # 相似邻居保留数
    )  # item2item 索引构建完成
    if not sasrec_recall_csv.exists():  # 若 SASRec 召回文件不存在
        print(  # 打印警告信息
            f"Warning: SASRec recall file not found: {sasrec_recall_csv}. "  # 提示缺失文件路径
            "Fusion will run without SASRec channel."  # 说明将跳过 SASRec 通道
        )  # 结束警告输出
    sasrec_map = load_channel_recall_csv(sasrec_recall_csv)  # 加载序列模型召回结果
    resolved_sequence_channel = sequence_channel or infer_sequence_channel(sasrec_recall_csv)  # 推断通道名
    weights_table = activity_weights or ACTIVITY_WEIGHTS  # 使用的分层权重表

    fixed_weights = {  # 固定权重（adaptive_weights=False 时使用）
        "popular": popular_weight,  # 热门通道权重
        "category_popular": category_popular_weight,  # 类别热门通道权重
        "item2item": item2item_weight,  # item2item 通道权重
        resolved_sequence_channel: sasrec_weight,  # 序列模型通道权重
    }  # 固定权重字典结束

    tier_counts: dict[str, int] = defaultdict(int)  # 各活跃度分层用户数

    FUSION_OUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建融合输出目录
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建评估输出目录
    rec_out = FUSION_OUT_DIR / f"fusion_{eval_split}.csv"  # 融合推荐结果输出路径
    metric_out = EVAL_OUT_DIR / f"fusion_{eval_split}_metrics.json"  # 评估指标输出路径

    maps, recalls, ndcgs, hits = [], [], [], []  # 初始化各指标累计列表
    rows = []  # 初始化推荐结果行列表

    for user_id, actual_items in targets.items():  # 遍历每个评估用户
        history = user_history_map.get(user_id, [])  # 获取用户历史序列
        history_set = set(history)  # 转为集合（活跃度权重与 API 兼容，不再用于排除已购）

        pop_cands = recall_popular(popular_index, user_history=history_set, top_k=popular_recall_top_k)  # 热门通道召回
        category_pop_cands = recall_category_popular(  # 类别热门通道召回
            history,  # 用户历史序列
            category_popular_index,  # 类别热门索引
            seed_items=category_popular_seed_items,  # 种子商品数
            top_k=category_popular_recall_top_k,  # 召回 Top-K
        )  # 类别热门召回完成
        item2item_cands = recall_item2item(  # item2item 共现召回
            history,  # 用户历史序列
            item2item_index,  # item2item 共现索引
            seed_items=item2item_seed_items,  # 种子商品数
            top_k=item2item_recall_top_k,  # 召回 Top-K
        )  # item2item 召回完成
        sasrec_cands = [  # 序列模型通道召回
            (iid, score) for iid, score, _ in sasrec_map.get(user_id, [])[:recall_top_k]  # 截取 Top-K 候选
        ]  # 序列模型通道召回

        if adaptive_weights:  # 按历史长度自适应权重
            tier = classify_activity_tier(len(history))  # 判定活跃度
            tier_counts[tier] += 1  # 统计分层人数
            user_weights = get_channel_weights_for_user(
                len(history),
                resolved_sequence_channel,
                activity_weights=weights_table,
            )
        else:  # 全用户统一权重
            user_weights = fixed_weights  # 使用固定权重

        fused = fuse_candidates(  # 融合四通道候选
            user_id=user_id,  # 传入用户 ID
            user_history=history_set,  # 传入用户历史
            channel_candidates={  # 组装各通道候选
                "popular": pop_cands,  # 热门通道候选
                "category_popular": category_pop_cands,  # 类别热门通道候选
                "item2item": item2item_cands,  # item2item 共现候选
                resolved_sequence_channel: sasrec_cands,  # 序列模型通道候选
            },  # 结束通道候选字典
            channel_weights=user_weights,  # 传入通道权重
            top_k=final_top_k,  # 指定最终 Top-K
            exclude_seen=exclude_seen,  # 是否排除已购
        )  # 结束融合调用

        pred_items = [item_id for item_id, _ in fused]  # 提取预测物品 ID 列表
        maps.append(_map_at_k(actual_items, pred_items, final_top_k))  # 累计 MAP@K
        recalls.append(_recall_at_k(actual_items, pred_items, final_top_k))  # 累计 Recall@K
        ndcgs.append(_ndcg_at_k(actual_items, pred_items, final_top_k))  # 累计 NDCG@K
        hits.append(_hit_at_k(actual_items, pred_items, final_top_k))  # 累计 Hit@K

        for rank, (item_id, score) in enumerate(fused, start=1):  # 遍历融合结果并记录排名
            rows.append(  # 追加一行推荐记录
                {  # 构建推荐行字典
                    "user_id": user_id,  # 用户 ID
                    "item_id": item_id,  # 物品 ID
                    "score": score,  # 融合得分
                    "rank": rank,  # 推荐排名
                    "split": eval_split,  # 评估划分
                    "channel": "fusion",  # 渠道标识为融合
                }  # 结束行字典
            )  # 结束追加

    with rec_out.open("w", newline="", encoding="utf-8") as f:  # 打开推荐结果输出文件
        writer = csv.DictWriter(  # 创建字典 CSV 写入器
            f,  # 绑定输出文件
            fieldnames=["user_id", "item_id", "score", "rank", "split", "channel"],  # 指定列名
        )  # 结束写入器创建
        writer.writeheader()  # 写入表头
        writer.writerows(rows)  # 写入全部推荐行

    metrics = {  # 汇总评估指标
        f"MAP@{final_top_k}": float(sum(maps) / len(maps)) if maps else 0.0,  # 平均 MAP@K
        f"Recall@{final_top_k}": float(sum(recalls) / len(recalls)) if recalls else 0.0,  # 平均 Recall@K
        f"NDCG@{final_top_k}": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,  # 平均 NDCG@K
        f"Hit@{final_top_k}": float(sum(hits) / len(hits)) if hits else 0.0,  # 平均 Hit@K
        "users_evaluated": len(targets),  # 评估用户数量
        "adaptive_weights": adaptive_weights,  # 是否启用自适应权重
        "exclude_seen": exclude_seen,  # 是否排除已购商品
        "sequence_channel": resolved_sequence_channel,  # 序列模型通道名
        "activity_weights": (
            {tier: dict(w) for tier, w in weights_table.items()} if adaptive_weights else None
        ),
        "popular_recall_top_k": popular_recall_top_k,  # 热门召回 Top-K
        "category_popular_recall_top_k": category_popular_recall_top_k,  # 类别热门召回 Top-K
        "recall_top_k": recall_top_k,  # 序列模型召回 Top-K
        "item2item_recall_top_k": item2item_recall_top_k,  # item2item 召回 Top-K
        "item2item_cooccur_weeks": item2item_cooccur_weeks,  # item2item 共现统计窗口（周）
        "item2item_top_sim_k": item2item_top_sim_k,  # 每个商品保留相似邻居数
        "item2item_seed_items": item2item_seed_items,  # item2item 种子商品数
        "category_popular_seed_items": category_popular_seed_items,  # 类别热门种子商品数
        "weights": fixed_weights if not adaptive_weights else "per-user by activity tier",  # 权重说明
        "activity_tier_counts": dict(tier_counts) if adaptive_weights else {},  # 各分层用户数
        "eval_split": eval_split,  # 评估划分名称
    }  # 结束指标字典
    metric_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")  # 将指标写入 JSON 文件

    print(f"Saved fusion recommendations: {rec_out}")  # 打印推荐结果保存路径
    print(f"Saved evaluation metrics: {metric_out}")  # 打印指标文件保存路径
    print(json.dumps(metrics, ensure_ascii=False, indent=2))  # 打印指标 JSON
    return rec_out, metric_out, metrics  # 返回输出路径与指标


def main() -> None:  # 命令行入口函数
    parser = argparse.ArgumentParser(description="Multi-channel fusion offline evaluation")  # 创建参数解析器
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")  # 评估划分参数
    parser.add_argument("--recall-top-k", type=int, default=100)  # 序列模型召回 Top-K
    parser.add_argument(  # 热门召回 Top-K 参数
        "--popular-recall-top-k",  # 参数名
        type=int,  # 整数类型
        default=POPULAR_RECALL_TOP_K,  # 默认值
        help="Global popular recall top-k (default: 50)",  # 帮助文本
    )  # 热门召回参数结束
    parser.add_argument(  # 类别热门召回 Top-K 参数
        "--category-popular-recall-top-k",  # 参数名
        type=int,  # 整数类型
        default=CATEGORY_POPULAR_RECALL_TOP_K,  # 默认值
        help="Category popular recall top-k (default: 50)",  # 帮助文本
    )  # 类别热门召回参数结束
    parser.add_argument("--final-top-k", type=int, default=12)  # 融合最终 Top-K 参数
    parser.add_argument("--popular-weight", type=float, default=0.15)  # 固定权重：热门通道
    parser.add_argument("--category-popular-weight", type=float, default=0.15)  # 固定权重：类别热门
    parser.add_argument("--item2item-weight", type=float, default=0.25)  # 固定权重：item2item 通道
    parser.add_argument("--sasrec-weight", type=float, default=0.45)  # 固定权重：序列模型通道
    parser.add_argument(  # item2item 召回 Top-K 参数
        "--item2item-recall-top-k",  # 参数名
        type=int,  # 整数类型
        default=ITEM2ITEM_RECALL_TOP_K,  # 默认值
        help="Item2item recall top-k (default: 50)",  # 帮助文本
    )  # item2item 召回参数结束
    parser.add_argument("--item2item-cooccur-weeks", type=int, default=COOCCUR_WEEKS)  # item2item 共现窗口参数
    parser.add_argument("--item2item-top-sim-k", type=int, default=TOP_SIM_K)  # item2item 相似邻居数参数
    parser.add_argument("--item2item-seed-items", type=int, default=SEED_ITEMS)  # item2item 种子商品数参数
    parser.add_argument("--category-popular-seed-items", type=int, default=CATEGORY_SEED_ITEMS)  # 类别热门种子商品数参数
    parser.add_argument("--sasrec-recall-csv", type=Path, default=None)  # 可选序列模型召回 CSV
    parser.add_argument(  # 序列通道名参数
        "--sequence-channel",  # 参数名
        type=str,  # 字符串类型
        default=None,  # 默认从 CSV 文件名推断
        help="Sequence model channel key (sasrec/sasrecf); default inferred from recall csv filename",  # 帮助文本
    )  # 序列通道参数结束
    parser.add_argument(  # 禁用自适应权重开关
        "--no-adaptive-weights",  # 参数名
        action="store_true",  # 布尔开关
        help="Use fixed weights for all users instead of activity-based adaptive weights",  # 帮助文本
    )  # 自适应权重开关结束
    parser.add_argument(
        "--exclude-seen",
        action="store_true",
        help="Exclude items already in user history from fusion candidates",
    )
    parser.add_argument(
        "--weights-json",
        type=Path,
        default=None,
        help="Load per-tier fusion weights from JSON (e.g. outputs/evaluation/best_fusion_weights.json)",
    )
    args = parser.parse_args()  # 解析命令行参数

    loaded_weights = None
    exclude_seen = args.exclude_seen
    if args.weights_json is not None:
        from src.evaluate.weight_search import load_best_weights

        payload = load_best_weights(args.weights_json)
        loaded_weights = payload["best_weights"]
        if "exclude_seen" in payload and not args.exclude_seen:
            exclude_seen = bool(payload["exclude_seen"])

    evaluate_fusion(  # 调用融合评估主流程
        eval_split=args.eval_split,  # 传入评估划分
        recall_top_k=args.recall_top_k,  # 传入其他通道召回 Top-K
        popular_recall_top_k=args.popular_recall_top_k,  # 传入热门召回 Top-K
        category_popular_recall_top_k=args.category_popular_recall_top_k,  # 传入类别热门 Top-K
        final_top_k=args.final_top_k,  # 传入最终 Top-K
        popular_weight=args.popular_weight,  # 传入热门权重
        category_popular_weight=args.category_popular_weight,  # 传入类别热门权重
        item2item_weight=args.item2item_weight,  # 传入 item2item 权重
        sasrec_weight=args.sasrec_weight,  # 传入序列模型权重
        item2item_recall_top_k=args.item2item_recall_top_k,  # 传入 item2item 召回 Top-K
        item2item_cooccur_weeks=args.item2item_cooccur_weeks,  # 传入 item2item 共现窗口
        item2item_top_sim_k=args.item2item_top_sim_k,  # 传入 item2item 相似邻居数
        item2item_seed_items=args.item2item_seed_items,  # 传入 item2item 种子商品数
        category_popular_seed_items=args.category_popular_seed_items,  # 传入类别热门种子商品数
        sasrec_recall_csv=args.sasrec_recall_csv,  # 传入序列模型召回 CSV
        adaptive_weights=not args.no_adaptive_weights,  # 默认启用自适应权重
        activity_weights=loaded_weights,  # 可选：搜索得到的分层权重
        exclude_seen=exclude_seen,  # 是否排除已购
        sequence_channel=args.sequence_channel,  # 传入序列通道名
    )  # 结束评估调用


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 执行主函数
