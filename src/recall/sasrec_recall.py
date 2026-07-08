"""SASRec recall export utilities."""  # SASRec 召回导出工具模块

from __future__ import annotations  # 启用延迟注解评估

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pytorch_compat import patch_recbole_compat

patch_recbole_compat()

from recbole.quick_start import load_data_and_model

DEFAULT_CKPT_DIR = Path("outputs/checkpoints/sasrec")  # 默认 SASRec 检查点目录
DEFAULT_OUTPUT_DIR = Path("outputs/recommendations")  # 默认召回结果输出目录
VALID_INTER = Path("data/processed/hm/hm.valid.inter")  # 验证集交互文件路径
TEST_INTER = Path("data/processed/hm/hm.test.inter")  # 测试集交互文件路径


def default_output_path(eval_split: str) -> Path:  # 根据评估划分生成默认输出路径
    return DEFAULT_OUTPUT_DIR / f"sasrec_{eval_split}.csv"  # 返回 sasrec_{split}.csv 路径


def _latest_checkpoint(checkpoint_dir: Path) -> Path:  # 查找目录中最新的检查点文件
    candidates = sorted(  # 按修改时间降序排序检查点
        checkpoint_dir.glob("*.pth"),  # 匹配所有 .pth 文件
        key=lambda p: p.stat().st_mtime,  # 以文件修改时间为排序键
        reverse=True,  # 降序排列
    )  # 结束 sorted
    if not candidates:  # 若未找到任何检查点
        raise FileNotFoundError(  # 抛出文件未找到异常
            f"No checkpoint found in {checkpoint_dir}. Run SASRec training first."  # 提示先训练 SASRec
        )  # 结束异常消息
    return candidates[0]  # 返回最新检查点路径


def _batched(items: list[tuple[int, int]], batch_size: int) -> Iterable[list[tuple[int, int]]]:  # 将列表按批次切分
    for start in range(0, len(items), batch_size):  # 按 batch_size 步长遍历起始索引
        yield items[start : start + batch_size]  # 产出当前批次切片


def _as_str(x: object) -> str:  # 将对象安全转换为字符串
    if isinstance(x, bytes):  # 若为字节类型
        return x.decode("utf-8")  # 按 UTF-8 解码
    return str(x)  # 否则直接转为字符串


def _load_eval_users(eval_split: str) -> list[str]:  # 加载评估划分中的用户 ID 列表
    if eval_split not in {"valid", "test"}:  # 校验评估划分名称
        raise ValueError("eval_split must be 'valid' or 'test'")  # 非法划分则报错

    path = VALID_INTER if eval_split == "valid" else TEST_INTER  # 选择对应交互文件
    if not path.exists():  # 若文件不存在
        raise FileNotFoundError(f"Missing eval split file: {path}")  # 抛出文件缺失异常

    df = pd.read_csv(path, sep="\t", usecols=["user_id:token"])  # 读取用户 ID 列
    return sorted(df["user_id:token"].astype(str).unique().tolist())  # 返回去重排序后的用户列表


def _resolve_user_rows(eval_dataset, uid_internal_list: list[int]) -> list[tuple[int, int]]:  # 映射用户到数据集行索引
    """Map each user to the first interaction row in the eval dataset."""  # 将每个用户映射到评估集首条交互行
    uid_field = eval_dataset.uid_field  # 获取用户 ID 字段名
    uid_values = eval_dataset.inter_feat[uid_field].numpy()  # 读取全部内部用户 ID 数组

    rows: list[tuple[int, int]] = []  # 初始化 (内部用户ID, 行索引) 列表
    for uid in uid_internal_list:  # 遍历待评估用户
        matches = np.where(uid_values == uid)[0]  # 查找该用户在数据集中的行位置
        if len(matches) > 0:  # 若找到匹配行
            rows.append((uid, int(matches[0])))  # 记录用户与首条交互行索引
    return rows  # 返回用户行映射列表


def export_sasrec_recall(  # 导出 SASRec Top-K 召回结果到 CSV
    eval_split: str = "valid",  # 评估划分：valid 或 test
    model_file: str | Path | None = None,  # 模型检查点路径
    output_path: str | Path | None = None,  # 输出 CSV 路径
    top_k: int = 100,  # 每个用户召回数量
    batch_size: int = 512,  # 推理批大小
) -> Path:  # 返回输出文件路径
    """
    Export SASRec top-k recall for one eval split.

    valid:
      - uses RecBole valid_data (history = train)
      - targets users in hm.valid.inter
      - writes outputs/recommendations/sasrec_valid.csv

    test:
      - uses RecBole test_data (history = train + valid)
      - targets users in hm.test.inter
      - writes outputs/recommendations/sasrec_test.csv
    """  # 导出指定评估划分的 SASRec Top-K 召回 CSV
    if eval_split not in {"valid", "test"}:  # 校验评估划分名称
        raise ValueError("eval_split must be 'valid' or 'test'")  # 非法划分则报错

    output_path = Path(output_path or default_output_path(eval_split))  # 解析输出路径
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    if model_file is None:  # 若未指定检查点
        model_file = _latest_checkpoint(DEFAULT_CKPT_DIR)  # 使用最新检查点
    else:  # 若指定了检查点路径
        model_file = Path(model_file)  # 转为 Path 对象
        if not model_file.exists():  # 若检查点不存在
            raise FileNotFoundError(f"Checkpoint not found: {model_file}")  # 抛出异常

    config, model, _, _, valid_data, test_data = load_data_and_model(model_file=str(model_file))  # 加载模型与数据
    eval_data = valid_data if eval_split == "valid" else test_data  # 选择对应评估数据
    eval_dataset = eval_data.dataset  # 获取评估数据集对象
    uid_field = eval_dataset.uid_field  # 用户 ID 字段名
    iid_field = eval_dataset.iid_field  # 商品 ID 字段名
    device = config["device"]  # 获取推理设备

    eval_users = _load_eval_users(eval_split)  # 加载评估用户列表
    token2id = eval_dataset.field2token_id[uid_field]  # 获取用户 token 到内部 ID 映射
    internal_uids = [token2id[user_id] for user_id in eval_users if user_id in token2id]  # 过滤并转换内部用户 ID
    user_rows = _resolve_user_rows(eval_dataset, internal_uids)  # 解析用户对应的数据行

    missing_users = len(eval_users) - len(user_rows)  # 统计未匹配到数据集的用户数
    if missing_users > 0:  # 若存在缺失用户
        print(f"Warning: {missing_users:,} eval users not found in RecBole {eval_split} dataset.")  # 打印警告

    model.eval()  # 切换模型为评估模式
    total_rows = 0  # 初始化导出总行数计数
    with output_path.open("w", newline="", encoding="utf-8") as f:  # 打开输出 CSV 文件
        writer = csv.writer(f)  # 创建 CSV 写入器
        writer.writerow(["user_id", "item_id", "score", "rank", "channel"])  # 写入表头

        for batch in _batched(user_rows, batch_size):  # 按批次遍历用户
            uids = [uid for uid, _ in batch]  # 提取批次内用户内部 ID
            row_indices = torch.tensor([idx for _, idx in batch], dtype=torch.long)  # 构建行索引张量
            input_interaction = eval_dataset[row_indices].to(device)  # 取交互特征并移到设备

            with torch.no_grad():  # 关闭梯度计算
                scores = model.full_sort_predict(input_interaction)  # 全量商品打分预测
            scores = scores.view(-1, eval_dataset.item_num)  # 重塑为 (batch, item_num)
            scores[:, 0] = -np.inf  # 屏蔽 padding 商品（索引 0）

            topk_scores, topk_iids = torch.topk(scores, top_k)  # 取 Top-K 分数与商品 ID
            uid_tokens = eval_dataset.id2token(uid_field, np.array(uids))  # 内部用户 ID 转 token
            iid_tokens = eval_dataset.id2token(iid_field, topk_iids.cpu().numpy())  # 内部商品 ID 转 token
            score_mat = topk_scores.cpu().numpy()  # 分数矩阵转到 CPU NumPy

            for i in range(len(batch)):  # 遍历批次内每个用户
                user_id = _as_str(uid_tokens[i])  # 获取用户 token 字符串
                for rank in range(top_k):  # 遍历 Top-K 排名
                    item_id = _as_str(iid_tokens[i][rank])  # 获取商品 token 字符串
                    score = float(score_mat[i][rank])  # 获取预测分数
                    writer.writerow([user_id, item_id, score, rank + 1, "sasrec"])  # 写入一行召回结果
                    total_rows += 1  # 累计导出行数

    print(f"Eval split: {eval_split}")  # 打印评估划分
    print(f"Model checkpoint: {model_file}")  # 打印模型检查点路径
    print(f"Users exported: {len(user_rows):,}")  # 打印导出用户数
    print(f"Rows exported: {total_rows:,}")  # 打印导出总行数
    print(f"Saved SASRec recall to {output_path}")  # 打印保存路径
    return output_path  # 返回输出文件路径


def main() -> None:  # 命令行入口函数
    parser = argparse.ArgumentParser(description="Export SASRec top-k recall to CSV")  # 创建参数解析器
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")  # 评估划分参数
    parser.add_argument("--model-file", type=Path, default=None)  # 模型检查点路径参数
    parser.add_argument("--output-path", type=Path, default=None)  # 输出 CSV 路径参数
    parser.add_argument("--top-k", type=int, default=100)  # Top-K 召回数量参数
    parser.add_argument("--batch-size", type=int, default=512)  # 推理批大小参数
    args = parser.parse_args()  # 解析命令行参数

    export_sasrec_recall(  # 调用导出函数
        eval_split=args.eval_split,  # 传入评估划分
        model_file=args.model_file,  # 传入模型路径
        output_path=args.output_path,  # 传入输出路径
        top_k=args.top_k,  # 传入 Top-K
        batch_size=args.batch_size,  # 传入批大小
    )  # 结束 export_sasrec_recall 调用


if __name__ == "__main__":  # 脚本直接运行入口
    main()  # 执行主函数
