"""SASRec / SASRecF recall export utilities."""  # SASRec 系列模型召回导出工具模块

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析
import csv  # 导入 CSV 读写
from pathlib import Path  # 导入路径处理类
import sys  # 导入系统模块用于路径注入
from typing import Iterable  # 导入可迭代类型

import numpy as np  # 导入 NumPy 数值计算库
import pandas as pd  # 导入 pandas 数据分析库
import torch  # 导入 PyTorch 深度学习框架
import yaml  # 导入 YAML 配置文件解析

if __package__ is None or __package__ == "":  # 若以脚本方式直接运行
    project_root = Path(__file__).resolve().parents[2]  # 定位项目根目录
    if str(project_root) not in sys.path:  # 若根目录不在搜索路径中
        sys.path.insert(0, str(project_root))  # 注入项目根目录到 sys.path

from src.pytorch_compat import patch_recbole_compat  # 导入 RecBole 兼容性补丁

patch_recbole_compat()  # 应用 RecBole 兼容性补丁

from recbole.quick_start import load_data_and_model  # 导入 RecBole 模型与数据加载

DEFAULT_CONFIG = Path("configs/sasrec.yaml")  # 默认 SASRec 配置文件
DEFAULT_CKPT_DIR = Path("outputs/checkpoints/sasrec")  # 默认 SASRec 检查点目录
DEFAULT_RECALL_TOP_K = 100  # 默认召回 Top-K
DEFAULT_OUTPUT_DIR = Path("outputs/recommendations")  # 默认召回结果输出目录
VALID_INTER = Path("data/processed/hm/hm.valid.inter")  # 验证集交互文件路径
TEST_INTER = Path("data/processed/hm/hm.test.inter")  # 测试集交互文件路径


def _load_recall_settings(config_path: Path | None) -> tuple[Path, int, str]:  # 从 YAML 配置读取召回设置
    """Read checkpoint_dir, recall_top_k, and channel name from a yaml config."""  # 读取检查点目录、召回 Top-K 与通道名
    config_path = config_path or DEFAULT_CONFIG  # 使用默认配置文件
    if not config_path.exists():  # 若配置文件不存在
        raise FileNotFoundError(f"Config not found: {config_path}")  # 抛出文件未找到异常

    with config_path.open("r", encoding="utf-8") as f:  # 以 UTF-8 打开配置文件
        cfg = yaml.safe_load(f) or {}  # 安全解析 YAML，空文件则用空字典

    checkpoint_dir = Path(cfg.get("checkpoint_dir", DEFAULT_CKPT_DIR))  # 解析检查点目录
    recall_top_k = int(cfg.get("recall_top_k", DEFAULT_RECALL_TOP_K))  # 解析召回 Top-K
    model_name = str(cfg.get("model", "SASRec"))  # 解析模型名称
    channel = model_name.lower()  # 通道名取模型名小写
    return checkpoint_dir, recall_top_k, channel  # 返回解析结果


def default_output_path(eval_split: str, channel: str = "sasrec") -> Path:  # 根据评估划分生成默认输出路径
    return DEFAULT_OUTPUT_DIR / f"{channel}_{eval_split}.csv"  # 返回 {channel}_{split}.csv 路径


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


def export_sasrec_recall(  # 导出 SASRec / SASRecF Top-K 召回结果到 CSV
    eval_split: str = "valid",  # 评估划分：valid 或 test
    model_file: str | Path | None = None,  # 模型检查点路径
    output_path: str | Path | None = None,  # 输出 CSV 路径
    top_k: int | None = None,  # 每个用户召回数量（默认从 config 读取 recall_top_k）
    batch_size: int = 512,  # 推理批大小
    config_path: str | Path | None = None,  # 模型配置文件路径
    checkpoint_dir: str | Path | None = None,  # 检查点目录（优先于 config）
    channel: str | None = None,  # 召回通道名（默认从 config 的 model 推导）
) -> Path:  # 返回输出文件路径
    """  # 函数文档字符串开始
    Export SASRec-family top-k recall for one eval split.  # 导出指定评估划分的 SASRec 系列 Top-K 召回

    Strategy: recall Top-100 first, then truncate to Top-12 in fusion/final eval.  # 先召回 Top-100，融合阶段再截断为 Top-12

    valid:  # 验证集导出策略
      - uses RecBole valid_data (history = train)  # 使用 RecBole 验证集（历史为训练集）
      - targets users in hm.valid.inter  # 目标用户来自 hm.valid.inter
      - writes outputs/recommendations/{channel}_valid.csv  # 输出到 valid 召回 CSV

    test:  # 测试集导出策略
      - uses RecBole test_data (history = train + valid)  # 使用 RecBole 测试集（历史为训练+验证）
      - targets users in hm.test.inter  # 目标用户来自 hm.test.inter
      - writes outputs/recommendations/{channel}_test.csv  # 输出到 test 召回 CSV
    """  # 导出指定评估划分的 SASRec 系列 Top-K 召回 CSV
    if eval_split not in {"valid", "test"}:  # 校验评估划分名称
        raise ValueError("eval_split must be 'valid' or 'test'")  # 非法划分则报错

    cfg_checkpoint_dir, cfg_recall_top_k, cfg_channel = _load_recall_settings(  # 从配置读取默认设置
        Path(config_path) if config_path is not None else None  # 解析配置路径
    )  # 配置加载结束
    resolved_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else cfg_checkpoint_dir  # 解析检查点目录
    resolved_top_k = cfg_recall_top_k if top_k is None else top_k  # 解析召回 Top-K
    resolved_channel = channel or cfg_channel  # 解析通道名

    output_path = Path(  # 解析输出路径
        output_path or default_output_path(eval_split, channel=resolved_channel)  # 使用默认或指定路径
    )  # 路径解析结束
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    if model_file is None:  # 若未指定检查点
        model_file = _latest_checkpoint(resolved_checkpoint_dir)  # 使用最新检查点
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

            topk_scores, topk_iids = torch.topk(scores, resolved_top_k)  # 取 Top-K 分数与商品 ID
            uid_tokens = eval_dataset.id2token(uid_field, np.array(uids))  # 内部用户 ID 转 token
            iid_tokens = eval_dataset.id2token(iid_field, topk_iids.cpu().numpy())  # 内部商品 ID 转 token
            score_mat = topk_scores.cpu().numpy()  # 分数矩阵转到 CPU NumPy

            for i in range(len(batch)):  # 遍历批次内每个用户
                user_id = _as_str(uid_tokens[i])  # 获取用户 token 字符串
                for rank in range(resolved_top_k):  # 遍历 Top-K 排名
                    item_id = _as_str(iid_tokens[i][rank])  # 获取商品 token 字符串
                    score = float(score_mat[i][rank])  # 获取预测分数
                    writer.writerow([user_id, item_id, score, rank + 1, resolved_channel])  # 写入一行召回结果
                    total_rows += 1  # 累计导出行数

    print(f"Eval split: {eval_split}")  # 打印评估划分
    print(f"Recall top-k: {resolved_top_k}")  # 打印召回 Top-K
    print(f"Model checkpoint: {model_file}")  # 打印模型检查点路径
    print(f"Users exported: {len(user_rows):,}")  # 打印导出用户数
    print(f"Rows exported: {total_rows:,}")  # 打印导出总行数
    print(f"Saved SASRec recall to {output_path}")  # 打印保存路径
    return output_path  # 返回输出文件路径


def main() -> None:  # 命令行入口函数
    parser = argparse.ArgumentParser(description="Export SASRec/SASRecF top-k recall to CSV")  # 创建参数解析器
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)  # 模型配置文件路径
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")  # 评估划分参数
    parser.add_argument("--model-file", type=Path, default=None)  # 模型检查点路径参数
    parser.add_argument("--checkpoint-dir", type=Path, default=None)  # 检查点目录参数
    parser.add_argument("--output-path", type=Path, default=None)  # 输出 CSV 路径参数
    parser.add_argument(  # 添加 Top-K 参数
        "--top-k",  # 参数名
        type=int,  # 整数类型
        default=None,  # 默认从配置读取
        help="Recall top-k (default: recall_top_k from config, usually 100)",  # 帮助文本
    )  # Top-K 参数定义结束
    parser.add_argument("--batch-size", type=int, default=512)  # 推理批大小参数
    args = parser.parse_args()  # 解析命令行参数

    export_sasrec_recall(  # 调用导出函数
        eval_split=args.eval_split,  # 传入评估划分
        model_file=args.model_file,  # 传入模型路径
        output_path=args.output_path,  # 传入输出路径
        top_k=args.top_k,  # 传入 Top-K
        batch_size=args.batch_size,  # 传入批大小
        config_path=args.config,  # 传入配置文件
        checkpoint_dir=args.checkpoint_dir,  # 传入检查点目录
    )  # 结束 export_sasrec_recall 调用


if __name__ == "__main__":  # 脚本直接运行入口
    main()  # 执行主函数
