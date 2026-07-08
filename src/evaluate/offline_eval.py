"""Run multi-channel fusion and offline evaluation."""  # 运行多通道融合与离线评估

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import csv  # 导入 CSV 读写模块
import json  # 导入 JSON 序列化模块
import math  # 导入数学函数库
from pathlib import Path  # 导入路径处理类

import pandas as pd  # 导入 pandas 数据分析库

from src.fusion.weighted_fusion import build_user_history, fuse_candidates, load_channel_recall_csv  # 导入融合相关函数
from src.recall.itemcf import build_itemcf_index, recall_itemcf  # 导入 ItemCF 召回函数
from src.recall.popular import build_popular_index, recall_popular  # 导入热门召回函数


TRAIN_INTER = Path("data/processed/hm/hm.train.inter")  # 训练集交互文件路径
VALID_INTER = Path("data/processed/hm/hm.valid.inter")  # 验证集交互文件路径
TEST_INTER = Path("data/processed/hm/hm.test.inter")  # 测试集交互文件路径

SASREC_RECALL_DIR = Path("outputs/recommendations")  # SASRec 召回结果目录
FUSION_OUT_DIR = Path("outputs/recommendations")  # 融合推荐输出目录
EVAL_OUT_DIR = Path("outputs/evaluation")  # 评估指标输出目录


def default_sasrec_recall_csv(eval_split: str) -> Path:  # 返回默认 SASRec 召回 CSV 路径
    return SASREC_RECALL_DIR / f"sasrec_{eval_split}.csv"  # 按评估划分拼接文件名


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


def evaluate_fusion(  # 执行多通道融合并评估
    eval_split: str = "valid",  # 评估划分：valid 或 test
    recall_top_k: int = 100,  # 各通道召回 Top-K
    final_top_k: int = 12,  # 融合后最终 Top-K
    popular_weight: float = 0.2,  # 热门通道权重
    itemcf_weight: float = 0.3,  # ItemCF 通道权重
    sasrec_weight: float = 0.5,  # SASRec 通道权重
    itemcf_min_cooccur: int = 2,  # ItemCF 最小共现次数
    sasrec_recall_csv: str | Path | None = None,  # 可选 SASRec 召回 CSV 路径
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
    itemcf_index = build_itemcf_index(  # 构建 ItemCF 召回索引
        history_paths,  # 传入历史交互文件路径
        min_cooccur=itemcf_min_cooccur,  # 设置最小共现阈值
    )  # 结束 ItemCF 索引构建
    if not sasrec_recall_csv.exists():  # 若 SASRec 召回文件不存在
        print(  # 打印警告信息
            f"Warning: SASRec recall file not found: {sasrec_recall_csv}. "  # 提示缺失文件路径
            "Fusion will run without SASRec channel."  # 说明将跳过 SASRec 通道
        )  # 结束警告输出
    sasrec_map = load_channel_recall_csv(sasrec_recall_csv)  # 加载 SASRec 召回结果

    weights = {  # 定义各通道融合权重
        "popular": popular_weight,  # 热门通道权重
        "itemcf": itemcf_weight,  # ItemCF 通道权重
        "sasrec": sasrec_weight,  # SASRec 通道权重
    }  # 结束权重字典

    FUSION_OUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建融合输出目录
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建评估输出目录
    rec_out = FUSION_OUT_DIR / f"fusion_{eval_split}.csv"  # 融合推荐结果输出路径
    metric_out = EVAL_OUT_DIR / f"fusion_{eval_split}_metrics.json"  # 评估指标输出路径

    maps, recalls, ndcgs, hits = [], [], [], []  # 初始化各指标累计列表
    rows = []  # 初始化推荐结果行列表

    for user_id, actual_items in targets.items():  # 遍历每个评估用户
        history = user_history_map.get(user_id, [])  # 获取用户历史序列
        history_set = set(history)  # 转为集合便于过滤

        pop_cands = recall_popular(popular_index, user_history=history_set, top_k=recall_top_k)  # 热门通道召回
        itemcf_cands = recall_itemcf(history, itemcf_index, top_k=recall_top_k)  # ItemCF 通道召回
        sasrec_cands = [(iid, score) for iid, score, _ in sasrec_map.get(user_id, [])[:recall_top_k]]  # SASRec 通道召回

        fused = fuse_candidates(  # 融合三通道候选
            user_id=user_id,  # 传入用户 ID
            user_history=history_set,  # 传入用户历史集合
            channel_candidates={  # 组装各通道候选
                "popular": pop_cands,  # 热门通道候选
                "itemcf": itemcf_cands,  # ItemCF 通道候选
                "sasrec": sasrec_cands,  # SASRec 通道候选
            },  # 结束通道候选字典
            channel_weights=weights,  # 传入通道权重
            top_k=final_top_k,  # 指定最终 Top-K
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
        "weights": weights,  # 使用的通道权重
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
    parser.add_argument("--recall-top-k", type=int, default=100)  # 各通道召回 Top-K 参数
    parser.add_argument("--final-top-k", type=int, default=12)  # 融合最终 Top-K 参数
    parser.add_argument("--popular-weight", type=float, default=0.2)  # 热门通道权重参数
    parser.add_argument("--itemcf-weight", type=float, default=0.3)  # ItemCF 通道权重参数
    parser.add_argument("--sasrec-weight", type=float, default=0.5)  # SASRec 通道权重参数
    parser.add_argument("--itemcf-min-cooccur", type=int, default=2)  # ItemCF 最小共现参数
    parser.add_argument("--sasrec-recall-csv", type=Path, default=None)  # 可选 SASRec 召回 CSV 参数
    args = parser.parse_args()  # 解析命令行参数

    evaluate_fusion(  # 调用融合评估主流程
        eval_split=args.eval_split,  # 传入评估划分
        recall_top_k=args.recall_top_k,  # 传入召回 Top-K
        final_top_k=args.final_top_k,  # 传入最终 Top-K
        popular_weight=args.popular_weight,  # 传入热门权重
        itemcf_weight=args.itemcf_weight,  # 传入 ItemCF 权重
        sasrec_weight=args.sasrec_weight,  # 传入 SASRec 权重
        itemcf_min_cooccur=args.itemcf_min_cooccur,  # 传入 ItemCF 最小共现
        sasrec_recall_csv=args.sasrec_recall_csv,  # 传入 SASRec 召回 CSV
    )  # 结束评估调用


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 执行主函数
