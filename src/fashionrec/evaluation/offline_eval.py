"""Run multi-channel fusion and offline evaluation."""  # 运行多通道融合与离线评估

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import csv  # 导入 CSV 读写模块
import json  # 导入 JSON 序列化模块
import sys  # 导入系统模块用于路径注入
from collections import defaultdict  # 导入带默认值的字典
from dataclasses import dataclass  # 融合评估上下文
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas 数据分析库

if __package__ is None or __package__ == "":  # 若以脚本方式直接运行
    project_root = Path(__file__).resolve().parents[2]  # 定位项目根目录
    if str(project_root) not in sys.path:  # 若根目录不在搜索路径中
        sys.path.insert(0, str(project_root))  # 注入项目根目录到 sys.path

from fashionrec.data.split import (  # 时间切分与防泄漏路径约定
    TEST_INTER_FILE,  # 测试交互路径
    TRAIN_INTER_FILE,  # 训练交互路径
    VALID_INTER_FILE,  # 验证交互路径
    assert_history_paths_allowed,  # 检查召回索引未混入标签周
    history_paths_for_eval,  # 按评估划分选择历史路径
)  # 切分模块导入结束
from fashionrec.data.paths import ProcessedDataPaths
from fashionrec.evaluation.metrics import hit_at_k, map_at_k, ndcg_at_k, recall_at_k  # 统一指标实现
from fashionrec.domain.ids import canonical_item_id, canonical_user_id  # 统一 ID 契约
from fashionrec.ranking.fusion import (  # 导入融合相关函数
    ACTIVITY_WEIGHTS,  # 默认活跃度权重模板
    ActivityTier,  # 活跃度分层类型
    build_user_history,  # 构建用户历史映射
    classify_activity_tier,  # 按历史长度分类活跃度
    get_channel_weights_for_user,  # 获取用户自适应通道权重
    infer_sequence_channel,  # 从文件名推断序列通道名
    load_channel_recall_csv,  # 加载通道召回 CSV
)  # 融合模块导入结束
from fashionrec.recall.category_popular import (  # 类别热门召回
    CATEGORY_POPULAR_RECALL_TOP_K,  # 类别热门召回 Top-K 常量
    SEED_ITEMS as CATEGORY_SEED_ITEMS,  # 类别热门种子商品数常量
)  # 类别热门模块导入结束
from fashionrec.recall.item2item import (  # item2item 共现召回
    COOCCUR_WEEKS,  # 共现统计窗口（周）
    ITEM2ITEM_RECALL_TOP_K,  # item2item 召回 Top-K 常量
    SEED_ITEMS,  # 种子商品数常量
    TOP_SIM_K,  # 每个商品保留相似邻居数
)  # item2item 模块导入结束
from fashionrec.recall.popular import POPULAR_RECALL_TOP_K  # 导入热门召回 Top-K 常量
from fashionrec.recall.generator import generate_candidates, read_candidate_csv  # 与规则导出共享候选生成器
from fashionrec.recall.registry import PrecomputedChannel, build_rule_channel_registry  # 规则注册表与序列适配器
from fashionrec.ranking.weighted_rrf import WeightedRRFRanker  # 排序层基线实现


TRAIN_INTER = TRAIN_INTER_FILE  # 训练集交互文件路径
VALID_INTER = VALID_INTER_FILE  # 验证集交互文件路径
TEST_INTER = TEST_INTER_FILE  # 测试集交互文件路径

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
    df = pd.read_csv(  # 读取用户与物品列并保留 ID 文本
        path,
        sep="\t",
        usecols=["user_id:token", "item_id:token"],
        dtype={"user_id:token": "string", "item_id:token": "string"},
    )
    df["user_id:token"] = df["user_id:token"].map(canonical_user_id)  # 统一用户 ID
    df["item_id:token"] = df["item_id:token"].map(canonical_item_id)  # 统一商品 ID
    grouped = (  # 按用户聚合真实物品集合
        df.groupby("user_id:token")["item_id:token"]  # 按用户分组并取物品列
        .apply(lambda s: {canonical_item_id(x) for x in s.tolist()})  # 将每组物品转为规范化集合
        .to_dict()  # 转为字典
    )  # 结束标签聚合
    return grouped  # 返回用户到真实物品集合的映射


def _recall_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 兼容旧内部名称
    return recall_at_k(actual, pred, k)  # 转调统一实现


def _hit_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 兼容旧内部名称
    return hit_at_k(actual, pred, k)  # 转调统一实现


def _ndcg_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 兼容旧内部名称
    return ndcg_at_k(actual, pred, k)  # 转调统一实现


def _map_at_k(actual: set[str], pred: list[str], k: int) -> float:  # 兼容旧内部名称
    return map_at_k(actual, pred, k)  # 转调统一实现


@dataclass  # 数据类装饰器
class FusionEvalContext:  # 预计算召回候选，供权重搜索复用
    targets: dict[str, set[str]]  # 用户真实标签
    users: list[dict]  # 每用户 history / channel_candidates
    sequence_channel: str  # 序列模型通道名
    final_top_k: int  # 最终 Top-K


def build_fusion_eval_context(  # 构建融合评估上下文（召回只算一次）
    eval_split: str = "valid",  # 评估划分：valid 或 test
    recall_top_k: int = 100,  # 序列模型召回 Top-K
    popular_recall_top_k: int = POPULAR_RECALL_TOP_K,  # 全局热门召回 Top-K
    category_popular_recall_top_k: int = CATEGORY_POPULAR_RECALL_TOP_K,  # 类别热门召回 Top-K
    item2item_recall_top_k: int = ITEM2ITEM_RECALL_TOP_K,  # item2item 召回 Top-K
    item2item_cooccur_weeks: int = COOCCUR_WEEKS,  # item2item 共现统计窗口（周）
    item2item_top_sim_k: int = TOP_SIM_K,  # 每个商品保留相似邻居数
    item2item_seed_items: int = SEED_ITEMS,  # item2item 种子商品数
    category_popular_seed_items: int = CATEGORY_SEED_ITEMS,  # 类别热门种子商品数
    final_top_k: int = 12,  # 融合后最终 Top-K
    sasrec_recall_csv: str | Path | None = None,  # 可选序列模型召回 CSV 路径
    sequence_channel: str | None = None,  # 序列通道名（sasrec / sasrecf），默认从 CSV 推断
    strict: bool = False,  # 正式运行可要求缺失依赖立即失败
    candidate_csv: str | Path | None = None,  # 可直接消费已物化四路候选
    max_user_history: int = 100,  # 用户分层与排除已购使用的历史上限
    data_dir: str | Path | None = None,  # processed dataset root
) -> FusionEvalContext:  # 返回融合评估上下文
    if eval_split not in {"valid", "test"}:  # 校验评估划分参数
        raise ValueError("eval_split must be 'valid' or 'test'")  # 非法划分时抛出异常

    sasrec_recall_csv = (  # 确定序列模型召回文件路径
        Path(sasrec_recall_csv)  # 若用户提供路径则转为 Path
        if sasrec_recall_csv is not None  # 判断路径是否非空
        else default_sasrec_recall_csv(eval_split)  # 否则使用默认路径
    )  # 结束路径选择
    data_paths = ProcessedDataPaths.from_root(data_dir)
    eval_path = data_paths.valid_inter if eval_split == "valid" else data_paths.test_inter
    history_paths = history_paths_for_eval(eval_split, data_paths.train_inter, data_paths.valid_inter)
    assert_history_paths_allowed(eval_split, history_paths, data_paths.train_inter, data_paths.valid_inter, data_paths.test_inter)

    user_history_map = build_user_history(*history_paths, max_user_history=max_user_history)  # 构建统一长度的用户历史
    targets = _load_targets(eval_path)  # 加载评估集真实标签
    if candidate_csv is not None:  # 正式流程优先消费已物化候选
        candidate_path = Path(candidate_csv)
        if strict and not candidate_path.exists():
            raise FileNotFoundError(f"Missing required candidate artifact: {candidate_path}")
        generated = read_candidate_csv(candidate_path) if candidate_path.exists() else []
        bad_splits = sorted({candidate.split for candidate in generated if candidate.split != eval_split})
        if bad_splits:
            raise ValueError(f"Candidate artifact split mismatch: expected={eval_split}, found={bad_splits}")
        sequence_names = sorted({candidate.channel for candidate in generated if candidate.channel.startswith("sasrec")})
        resolved_sequence_channel = sequence_channel or (sequence_names[0] if sequence_names else "sasrecf")
        if strict and not generated:
            raise ValueError(f"Candidate artifact is empty: {candidate_path}")
        if strict and not sequence_names:
            raise ValueError(f"Candidate artifact has no SASRec/SASRecF channel: {candidate_path}")
    else:  # 兼容旧命令：现场生成相同 Candidate schema
        if strict and not sasrec_recall_csv.exists():
            raise FileNotFoundError(f"Missing required sequence recall: {sasrec_recall_csv}")
        sasrec_map = load_channel_recall_csv(sasrec_recall_csv)
        resolved_sequence_channel = sequence_channel or infer_sequence_channel(sasrec_recall_csv)
        registry = build_rule_channel_registry(
            history_paths,
            item2item_cooccur_weeks=item2item_cooccur_weeks,
            item2item_top_sim_k=item2item_top_sim_k,
            item2item_seed_items=item2item_seed_items,
            category_seed_items=category_popular_seed_items,
            item_file=data_paths.seq_item,
        )
        registry[resolved_sequence_channel] = PrecomputedChannel(
            resolved_sequence_channel,
            {user: [(item, score) for item, score, _ in rows] for user, rows in sasrec_map.items()},
        )
        generated = generate_candidates(
            eval_users=targets,
            user_history=user_history_map,
            channels=registry,
            split=eval_split,
            top_k_by_channel={
                "popular": popular_recall_top_k,
                "category_popular": category_popular_recall_top_k,
                "item2item": item2item_recall_top_k,
                resolved_sequence_channel: recall_top_k,
            },
        )
    candidates_by_user: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for candidate in generated:  # Candidate schema 转为兼容的融合输入
        candidates_by_user[candidate.user_id][candidate.channel].append((candidate.item_id, candidate.score))

    users: list[dict] = []  # 初始化用户评估数据列表
    for user_id, actual_items in targets.items():  # 遍历每个评估用户
        history = user_history_map.get(user_id, [])  # 获取用户历史序列
        history_set = set(history)  # 转为集合
        channel_candidates = dict(candidates_by_user.get(user_id, {}))  # 使用统一生成的候选
        users.append(  # 追加用户评估数据
            {  # 用户数据字典
                "user_id": user_id,  # 用户 ID
                "actual_items": actual_items,  # 真实标签集合
                "history": history,  # 历史序列
                "history_set": history_set,  # 历史集合
                "channel_candidates": channel_candidates,  # 各通道候选
            }  # 用户数据字典结束
        )  # 追加完成

    return FusionEvalContext(  # 构建并返回评估上下文
        targets=targets,  # 用户真实标签
        users=users,  # 用户评估数据列表
        sequence_channel=resolved_sequence_channel,  # 序列模型通道名
        final_top_k=final_top_k,  # 最终 Top-K
    )  # 上下文构建完成


def evaluate_fusion_map_at_k(  # 给定权重模板计算平均 MAP@K
    context: FusionEvalContext,  # 预计算的融合评估上下文
    activity_weights: dict[ActivityTier, dict[str, float]],  # 各分层通道权重模板
    exclude_seen: bool = False,  # 融合时是否排除历史已购
) -> float:  # 返回平均 MAP@K
    maps: list[float] = []  # 初始化各用户 MAP 列表
    for row in context.users:  # 遍历每个用户
        user_weights = get_channel_weights_for_user(  # 按历史长度获取用户通道权重
            len(row["history"]),  # 用户历史长度
            context.sequence_channel,  # 序列模型通道名
            activity_weights=activity_weights,  # 传入分层权重模板
        )  # 权重获取完成
        ranked = WeightedRRFRanker(user_weights, exclude_seen=exclude_seen).rank(  # 通过排序接口执行 RRF
            user_id=row["user_id"],  # 用户 ID
            user_history=row["history_set"],  # 用户历史集合
            channel_candidates=row["channel_candidates"],  # 各通道候选
            top_k=context.final_top_k,  # 最终 Top-K
        )  # 排序完成
        pred_items = [item.item_id for item in ranked]  # 提取预测物品 ID 列表
        maps.append(_map_at_k(row["actual_items"], pred_items, context.final_top_k))  # 累计 MAP@K
    return float(sum(maps) / len(maps)) if maps else 0.0  # 返回平均 MAP@K


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
    output_dir: str | Path = FUSION_OUT_DIR,  # 推荐输出目录，支持 run-scoped 路径
    evaluation_dir: str | Path = EVAL_OUT_DIR,  # 指标输出目录，支持 run-scoped 路径
    strict: bool = False,  # 严格模式禁止缺失序列召回
    candidate_csv: str | Path | None = None,  # 已物化候选输入
    max_user_history: int = 100,  # 用户分层与排除已购使用的历史上限
    data_dir: str | Path | None = None,  # processed dataset root
) -> tuple[Path, Path, dict[str, float]]:  # 返回推荐文件路径、指标文件路径与指标字典
    """Run multi-channel recall fusion and evaluate on valid/test split."""  # 在 valid/test 划分上运行多通道召回融合并评估
    if eval_split not in {"valid", "test"}:  # 校验评估划分参数
        raise ValueError("eval_split must be 'valid' or 'test'")  # 非法划分时抛出异常

    context = build_fusion_eval_context(  # 召回、候选与权重搜索共用同一上下文构建
        eval_split=eval_split,
        recall_top_k=recall_top_k,
        popular_recall_top_k=popular_recall_top_k,
        category_popular_recall_top_k=category_popular_recall_top_k,
        item2item_recall_top_k=item2item_recall_top_k,
        item2item_cooccur_weeks=item2item_cooccur_weeks,
        item2item_top_sim_k=item2item_top_sim_k,
        item2item_seed_items=item2item_seed_items,
        category_popular_seed_items=category_popular_seed_items,
        final_top_k=final_top_k,
        sasrec_recall_csv=sasrec_recall_csv,
        sequence_channel=sequence_channel,
        strict=strict,
        candidate_csv=candidate_csv,
        max_user_history=max_user_history,
        data_dir=data_dir,
    )
    targets = context.targets
    resolved_sequence_channel = context.sequence_channel
    weights_table = activity_weights or ACTIVITY_WEIGHTS  # 使用的分层权重表

    fixed_weights = {  # 固定权重（adaptive_weights=False 时使用）
        "popular": popular_weight,  # 热门通道权重
        "category_popular": category_popular_weight,  # 类别热门通道权重
        "item2item": item2item_weight,  # item2item 通道权重
        resolved_sequence_channel: sasrec_weight,  # 序列模型通道权重
    }  # 固定权重字典结束

    tier_counts: dict[str, int] = defaultdict(int)  # 各活跃度分层用户数

    resolved_output_dir = Path(output_dir)  # 解析推荐输出目录
    resolved_evaluation_dir = Path(evaluation_dir)  # 解析评估输出目录
    resolved_output_dir.mkdir(parents=True, exist_ok=True)  # 创建融合输出目录
    resolved_evaluation_dir.mkdir(parents=True, exist_ok=True)  # 创建评估输出目录
    rec_out = resolved_output_dir / f"fusion_{eval_split}.csv"  # 融合推荐结果输出路径
    metric_out = resolved_evaluation_dir / f"fusion_{eval_split}_metrics.json"  # 评估指标输出路径

    maps, recalls, ndcgs, hits = [], [], [], []  # 初始化各指标累计列表
    rows = []  # 初始化推荐结果行列表

    for row in context.users:  # 遍历统一候选上下文
        user_id = row["user_id"]
        actual_items = row["actual_items"]
        history = row["history"]
        history_set = row["history_set"]

        if adaptive_weights:  # 按历史长度自适应权重
            tier = classify_activity_tier(len(history))  # 判定活跃度
            tier_counts[tier] += 1  # 统计分层人数
            user_weights = get_channel_weights_for_user(  # 按历史长度获取用户通道权重
                len(history),  # 用户历史长度
                resolved_sequence_channel,  # 序列模型通道名
                activity_weights=weights_table,  # 传入分层权重表
            )  # 权重获取完成
        else:  # 全用户统一权重
            user_weights = fixed_weights  # 使用固定权重

        ranked = WeightedRRFRanker(user_weights, exclude_seen=exclude_seen).rank(  # 融合基线排序器
            user_id=user_id,  # 传入用户 ID
            user_history=history_set,  # 传入用户历史
            channel_candidates=row["channel_candidates"],  # 使用统一 Candidate 生成结果
            top_k=final_top_k,  # 指定最终 Top-K
        )  # 结束排序调用

        pred_items = [item.item_id for item in ranked]  # 提取预测物品 ID 列表
        maps.append(_map_at_k(actual_items, pred_items, final_top_k))  # 累计 MAP@K
        recalls.append(_recall_at_k(actual_items, pred_items, final_top_k))  # 累计 Recall@K
        ndcgs.append(_ndcg_at_k(actual_items, pred_items, final_top_k))  # 累计 NDCG@K
        hits.append(_hit_at_k(actual_items, pred_items, final_top_k))  # 累计 Hit@K

        for item in ranked:  # 遍历排序结果并记录排名
            rows.append(  # 追加一行推荐记录
                {  # 构建推荐行字典
                    "user_id": user_id,  # 用户 ID
                    "item_id": item.item_id,  # 物品 ID
                    "score": item.score,  # 融合得分
                    "rank": item.rank,  # 推荐排名
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
        "activity_weights": (  # 自适应权重详情
            {tier: dict(w) for tier, w in weights_table.items()} if adaptive_weights else None  # 各分层权重或 None
        ),  # 自适应权重详情结束
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


def main(argv: list[str] | None = None) -> None:  # 命令行入口函数
    parser = argparse.ArgumentParser(prog="fashionrec evaluate", description="Multi-channel fusion offline evaluation")  # 创建参数解析器
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
    parser.add_argument("--candidate-csv", type=Path, default=None)  # 已物化四路候选 CSV
    parser.add_argument("--output-dir", type=Path, default=FUSION_OUT_DIR)  # 推荐输出目录
    parser.add_argument("--evaluation-dir", type=Path, default=EVAL_OUT_DIR)  # 指标输出目录
    parser.add_argument("--strict", action="store_true")  # 缺失依赖直接失败
    parser.add_argument("--max-user-history", type=int, default=100)  # 历史长度上限
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed dataset root; defaults to data/processed.")
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
    parser.add_argument(  # 排除已购商品开关
        "--exclude-seen",  # 参数名
        action="store_true",  # 布尔开关
        help="Exclude items already in user history from fusion candidates",  # 帮助文本
    )  # 排除已购开关结束
    parser.add_argument(  # 从 JSON 加载分层权重参数
        "--weights-json",  # 参数名
        type=Path,  # 路径类型
        default=None,  # 默认不加载
        help="Load per-tier fusion weights from JSON (e.g. outputs/evaluation/best_fusion_weights.json)",  # 帮助文本
    )  # 权重 JSON 参数结束
    args = parser.parse_args(argv)  # 解析显式命令参数

    loaded_weights = None  # 初始化加载的分层权重
    exclude_seen = args.exclude_seen  # 默认使用命令行排除已购标志
    if args.weights_json is not None:  # 若指定了权重 JSON 文件
        from fashionrec.evaluation.weight_search import load_best_weights  # 导入权重加载函数

        payload = load_best_weights(args.weights_json)  # 加载最优权重载荷
        loaded_weights = payload["best_weights"]  # 提取分层权重
        if "exclude_seen" in payload and not args.exclude_seen:  # 若 JSON 含 exclude_seen 且命令行未指定
            exclude_seen = bool(payload["exclude_seen"])  # 采用 JSON 中的 exclude_seen 设置

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
        candidate_csv=args.candidate_csv,  # 已物化候选
        output_dir=args.output_dir,  # 推荐输出目录
        evaluation_dir=args.evaluation_dir,  # 指标输出目录
        strict=args.strict,  # 严格依赖检查
        max_user_history=args.max_user_history,  # 统一历史上限
        data_dir=args.data_dir,
    )  # 结束评估调用


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 执行主函数
