"""Unified RecBole evaluation for SASRec / Pop / ItemKNN."""  # SASRec / Pop / ItemKNN 统一 RecBole 评估模块

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import json  # 导入 JSON 序列化模块
from pathlib import Path  # 导入路径处理类
from typing import Any  # 导入任意类型注解

import torch  # 导入 PyTorch 深度学习框架

from src.pytorch_compat import patch_recbole_compat  # 导入 RecBole 兼容性补丁

patch_recbole_compat()  # 应用 RecBole 兼容性补丁

from recbole.config import Config  # 导入 RecBole 配置类
from recbole.data import create_dataset, data_preparation  # 导入数据集创建与划分函数
from recbole.utils import get_model, get_trainer, init_seed  # 导入模型、训练器与种子初始化

DEFAULT_CONFIG = Path("configs/sasrec.yaml")  # 默认 SASRec 配置文件路径
DEFAULT_CKPT_DIR = Path("outputs/checkpoints/sasrec")  # 默认检查点目录
DEFAULT_OUTPUT_JSON = Path("outputs/evaluation/recbole_channel_metrics.json")  # 默认通道指标输出路径
DEFAULT_OUTPUT_MD = Path("outputs/evaluation/recbole_channel_comparison.md")  # 默认对比 Markdown 输出路径
DEFAULT_FUSION_JSON = Path("outputs/evaluation/recbole_fusion_weight_search.json")  # 默认融合搜索结果路径


def _latest_checkpoint(checkpoint_dir: Path) -> Path:  # 获取目录中最新的检查点文件
    candidates = sorted(  # 按修改时间排序候选检查点
        checkpoint_dir.glob("*.pth"),  # 匹配所有 .pth 检查点文件
        key=lambda p: p.stat().st_mtime,  # 以文件修改时间为排序键
        reverse=True,  # 降序排列，最新在前
    )  # 排序完成
    if not candidates:  # 若无候选检查点
        raise FileNotFoundError(  # 抛出文件未找到异常
            f"No checkpoint found in {checkpoint_dir}. Run SASRec training first."  # 提示需先训练 SASRec
        )  # 异常消息结束
    return candidates[0]  # 返回最新检查点路径


def _to_float_metrics(metrics: dict[str, Any]) -> dict[str, float]:  # 将指标字典值转为浮点数
    return {k: float(v) for k, v in metrics.items()}  # 逐项转换并返回新字典


def _normalized_path(path_like: str | Path) -> str:  # 规范化路径用于一致性比较
    return str(Path(path_like).resolve()).replace("\\", "/").lower()  # 解析为绝对路径、统一分隔符并转小写


def build_shared_context(config_path: Path) -> dict[str, Any]:  # 构建所有模型共享的数据上下文
    """Build one shared dataset/dataloader context for all models."""  # 为所有模型构建共享数据集与数据加载器上下文
    base_config = Config(model="SASRec", config_file_list=[str(config_path)])  # 以 SASRec 为基准创建配置
    init_seed(base_config["seed"], base_config["reproducibility"])  # 初始化随机种子
    dataset = create_dataset(base_config)  # 创建 RecBole 数据集
    train_data, valid_data, test_data = data_preparation(base_config, dataset)  # 划分训练/验证/测试数据
    return {  # 返回共享上下文字典
        "config": base_config,  # 基准配置对象
        "dataset": train_data._dataset,  # 底层数据集对象
        "train_data": train_data,  # 训练数据加载器
        "valid_data": valid_data,  # 验证数据加载器
        "test_data": test_data,  # 测试数据加载器
    }  # 上下文字典结束


def build_sasrec_model(shared: dict[str, Any], model_file: Path):  # 从检查点构建 SASRec 模型
    config = shared["config"]  # 获取共享配置
    model = get_model("SASRec")(config, shared["dataset"]).to(config["device"])  # 创建 SASRec 模型并移至设备

    checkpoint = torch.load(str(model_file), map_location=config["device"])  # 加载检查点文件
    if "state_dict" not in checkpoint:  # 检查点缺少 state_dict 键
        raise KeyError(f"Checkpoint missing 'state_dict': {model_file}")  # 抛出键错误
    model.load_state_dict(checkpoint["state_dict"])  # 加载模型权重
    return model  # 返回加载好的模型


def evaluate_sasrec_recbole(shared: dict[str, Any], model_file: Path) -> dict[str, dict[str, float]]:  # 评估 SASRec 模型
    config = shared["config"]  # 获取共享配置
    model = build_sasrec_model(shared, model_file)  # 从检查点构建 SASRec 模型
    trainer = get_trainer(config["MODEL_TYPE"], "SASRec")(config, model)  # 创建 SASRec 训练器
    valid_metrics = trainer.evaluate(shared["valid_data"], load_best_model=False, show_progress=False)  # 验证集评估
    test_metrics = trainer.evaluate(shared["test_data"], load_best_model=False, show_progress=False)  # 测试集评估
    return {  # 返回验证与测试指标
        "valid": _to_float_metrics(valid_metrics),  # 验证集浮点指标
        "test": _to_float_metrics(test_metrics),  # 测试集浮点指标
    }  # 指标字典结束


def evaluate_traditional_recbole(  # 评估传统推荐模型（Pop / ItemKNN）
    model_name: str,  # 模型名称
    shared: dict[str, Any],  # 共享数据上下文
    config_path: Path,  # 配置文件路径
    fit_epochs: int = 1,  # 训练轮数
) -> dict[str, dict[str, float]]:  # 返回验证与测试指标字典
    """Evaluate traditional model on the same shared dataset/token space."""  # 在相同共享数据集与词元空间上评估传统模型
    base_cfg = shared["config"]  # 获取基准配置
    config = Config(  # 创建传统模型专用配置
        model=model_name,  # 指定模型名称
        config_file_list=[str(config_path)],  # 加载配置文件
        config_dict={  # 覆盖配置字典
            "model": model_name,  # 模型名称
            "epochs": fit_epochs,  # 训练轮数
            "use_gpu": base_cfg["use_gpu"],  # 是否使用 GPU
            "gpu_id": base_cfg["gpu_id"],  # GPU 设备 ID
        },  # 配置字典结束
    )  # 配置创建完成
    init_seed(config["seed"], config["reproducibility"])  # 初始化随机种子

    model = get_model(model_name)(config, shared["dataset"]).to(base_cfg["device"])  # 创建模型并移至设备
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)  # 创建训练器
    trainer.fit(shared["train_data"], shared["valid_data"], saved=False, show_progress=False)  # 训练模型

    valid_metrics = trainer.evaluate(shared["valid_data"], load_best_model=False, show_progress=False)  # 验证集评估
    test_metrics = trainer.evaluate(shared["test_data"], load_best_model=False, show_progress=False)  # 测试集评估
    return {  # 返回验证与测试指标
        "valid": _to_float_metrics(valid_metrics),  # 验证集浮点指标
        "test": _to_float_metrics(test_metrics),  # 测试集浮点指标
    }  # 指标字典结束


def _fmt(value: float | None) -> str:  # 格式化指标值为四位小数字符串
    return "-" if value is None else f"{value:.4f}"  # None 显示为横线，否则保留四位小数


def _is_fusion_payload_compatible(  # 检查融合搜索结果是否与当前评估兼容
    fusion_payload: dict[str, Any],  # 融合搜索结果载荷
    checkpoint_path: Path,  # 当前 SASRec 检查点路径
    dataset_name: str,  # 当前数据集名称
) -> tuple[bool, str]:  # 返回是否兼容及原因
    if fusion_payload.get("protocol") != "recbole_full_ranking_hm_seq":  # 检查评估协议是否一致
        return False, "protocol mismatch"  # 协议不匹配

    fusion_ckpt = fusion_payload.get("checkpoint")  # 获取融合结果中的检查点路径
    if not fusion_ckpt:  # 融合结果缺少检查点信息
        return False, "missing checkpoint in fusion payload"  # 缺少检查点
    if _normalized_path(fusion_ckpt) != _normalized_path(checkpoint_path):  # 比较检查点路径
        return False, "checkpoint mismatch"  # 检查点不匹配

    fusion_dataset = fusion_payload.get("dataset")  # 获取融合结果中的数据集名称
    if fusion_dataset and fusion_dataset != dataset_name:  # 比较数据集名称
        return False, "dataset mismatch"  # 数据集不匹配
    return True, ""  # 兼容，无跳过原因


def write_comparison_markdown(  # 生成通道对比 Markdown 报告
    output_path: Path,  # 输出文件路径
    channel_metrics: dict[str, dict[str, dict[str, float]]],  # 各通道指标字典
    checkpoint_path: Path,  # SASRec 检查点路径
    dataset_name: str,  # 数据集名称
    fusion_file: Path,  # 融合搜索结果 JSON 路径
) -> None:  # 无返回值
    rows: list[tuple[str, str, str, str, str, str, str]] = []  # 初始化表格行列表
    for model_name in ("SASRec", "Pop", "ItemKNN"):  # 遍历三个通道模型
        metrics = channel_metrics[model_name]  # 获取该模型指标
        for split in ("valid", "test"):  # 遍历验证与测试划分
            m = metrics[split]  # 获取该划分的指标
            rows.append(  # 追加一行表格数据
                (  # 表格行元组
                    model_name,  # 模型名称
                    split,  # 数据划分
                    _fmt(m.get("map@12")),  # MAP@12 格式化值
                    _fmt(m.get("recall@12")),  # Recall@12 格式化值
                    _fmt(m.get("ndcg@12")),  # NDCG@12 格式化值
                    _fmt(m.get("hit@12")),  # Hit@12 格式化值
                    _fmt(m.get("precision@12")),  # Precision@12 格式化值
                )  # 表格行结束
            )  # 追加完成

    fusion_section = ""  # 初始化融合结果 Markdown 片段
    fusion_skip_reason = ""  # 初始化跳过融合段落的原因
    if fusion_file.exists():  # 若融合搜索结果文件存在
        fusion_payload = json.loads(fusion_file.read_text(encoding="utf-8"))  # 读取并解析融合 JSON
        ok, reason = _is_fusion_payload_compatible(  # 检查融合结果是否兼容
            fusion_payload=fusion_payload,  # 融合载荷
            checkpoint_path=checkpoint_path,  # 当前检查点路径
            dataset_name=dataset_name,  # 当前数据集名称
        )  # 兼容性检查完成
        if ok:  # 兼容则生成融合段落
            best = fusion_payload.get("best", {})  # 获取最佳权重结果
            w = best.get("weights", {})  # 获取最佳权重
            valid_m = best.get("valid_metrics", {})  # 获取最佳验证集指标
            test_m = best.get("test_metrics", {})  # 获取最佳测试集指标
            fusion_section = (  # 构建融合 Markdown 段落
                "\n## Fusion (RecBole)\n\n"  # 融合章节标题
                f"- Best weights: `popular={w.get('popular', 0):.2f}, "  # 最佳热门权重
                f"itemknn={w.get('itemknn', 0):.2f}, sasrec={w.get('sasrec', 0):.2f}`\n\n"  # 最佳 ItemKNN 与 SASRec 权重
                "| Model | Split | MAP@12 | Recall@12 | NDCG@12 | Hit@12 | Precision@12 |\n"  # 表头行
                "|-------|-------|-------:|----------:|--------:|-------:|-------------:|\n"  # 表头分隔行
                f"| Fusion | valid | {_fmt(valid_m.get('map@12'))} | {_fmt(valid_m.get('recall@12'))} | "  # 融合验证集行
                f"{_fmt(valid_m.get('ndcg@12'))} | {_fmt(valid_m.get('hit@12'))} | "  # 融合验证集 NDCG 与 Hit
                f"{_fmt(valid_m.get('precision@12'))} |\n"  # 融合验证集 Precision
                f"| Fusion | test | {_fmt(test_m.get('map@12'))} | {_fmt(test_m.get('recall@12'))} | "  # 融合测试集行
                f"{_fmt(test_m.get('ndcg@12'))} | {_fmt(test_m.get('hit@12'))} | "  # 融合测试集 NDCG 与 Hit
                f"{_fmt(test_m.get('precision@12'))} |\n"  # 融合测试集 Precision
            )  # 融合段落结束
        else:  # 不兼容则记录跳过原因
            fusion_skip_reason = reason  # 保存跳过原因

    lines = [  # 构建 Markdown 文档行列表
        "# RecBole Unified Comparison",  # 文档标题
        "",  # 空行
        "Unified protocol: RecBole full ranking on `hm_seq` benchmark split.",  # 协议说明
        "",  # 空行
        f"- Dataset: `{dataset_name}`",  # 数据集信息
        f"- SASRec checkpoint: `{checkpoint_path}`",  # 检查点信息
        "",  # 空行
        "## Channel Metrics",  # 通道指标章节标题
        "",  # 空行
        "| Model | Split | MAP@12 | Recall@12 | NDCG@12 | Hit@12 | Precision@12 |",  # 表头行
        "|-------|-------|-------:|----------:|--------:|-------:|-------------:|",  # 表头分隔行
    ]  # 文档头部行结束
    for row in rows:  # 遍历各通道表格行
        lines.append("| " + " | ".join(row) + " |")  # 追加 Markdown 表格行

    if fusion_section:  # 若有融合段落则追加
        lines.extend(["", fusion_section.strip(), ""])  # 追加融合章节
    elif fusion_skip_reason:  # 若因不兼容跳过融合
        lines.extend(  # 追加跳过说明
            [  # 跳过说明行列表
                "",  # 空行
                "## Fusion (RecBole)",  # 融合章节标题
                "",  # 空行
                f"- Skipped fusion section due to `{fusion_skip_reason}`.",  # 跳过原因说明
                "",  # 空行
            ]  # 跳过说明结束
        )  # 追加完成

    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写入 Markdown 文件


def main() -> None:  # 命令行入口函数
    parser = argparse.ArgumentParser(description="Evaluate SASRec/Pop/ItemKNN in RecBole protocol")  # 创建参数解析器
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)  # 配置文件路径参数
    parser.add_argument("--model-file", type=Path, default=None)  # 可选模型检查点文件参数
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)  # 检查点目录参数
    parser.add_argument("--fit-epochs", type=int, default=1)  # 传统模型训练轮数参数
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)  # 指标 JSON 输出路径参数
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)  # 对比 Markdown 输出路径参数
    parser.add_argument("--fusion-json", type=Path, default=DEFAULT_FUSION_JSON)  # 融合搜索结果路径参数
    args = parser.parse_args()  # 解析命令行参数

    model_file = args.model_file or _latest_checkpoint(args.checkpoint_dir)  # 确定 SASRec 检查点路径
    if not model_file.exists():  # 检查点文件不存在
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")  # 抛出文件未找到异常

    shared = build_shared_context(args.config)  # 构建共享数据上下文
    dataset_name = str(shared["config"]["dataset"])  # 获取数据集名称

    sasrec_metrics = evaluate_sasrec_recbole(shared, model_file)  # 评估 SASRec 通道
    pop_metrics = evaluate_traditional_recbole(  # 评估热门通道
        "Pop", shared, args.config, fit_epochs=args.fit_epochs  # 传入模型名与训练轮数
    )  # 热门通道评估完成
    itemknn_metrics = evaluate_traditional_recbole(  # 评估 ItemKNN 通道
        "ItemKNN", shared, args.config, fit_epochs=args.fit_epochs  # 传入模型名与训练轮数
    )  # ItemKNN 通道评估完成

    payload = {  # 构建输出载荷字典
        "protocol": "recbole_full_ranking_hm_seq",  # 评估协议标识
        "dataset": dataset_name,  # 数据集名称
        "checkpoint": str(model_file.resolve()),  # SASRec 检查点绝对路径
        "models": {  # 各通道模型指标
            "SASRec": sasrec_metrics,  # SASRec 通道指标
            "Pop": pop_metrics,  # 热门通道指标
            "ItemKNN": itemknn_metrics,  # ItemKNN 通道指标
        },  # 模型指标字典结束
    }  # 载荷字典结束
    args.output_json.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入 JSON 文件
    print(f"Saved RecBole channel metrics: {args.output_json}")  # 打印 JSON 保存路径

    write_comparison_markdown(  # 生成对比 Markdown 报告
        output_path=args.output_md,  # 输出路径
        channel_metrics=payload["models"],  # 各通道指标
        checkpoint_path=model_file,  # 检查点路径
        dataset_name=dataset_name,  # 数据集名称
        fusion_file=args.fusion_json,  # 融合搜索结果路径
    )  # Markdown 生成完成
    print(f"Saved RecBole comparison markdown: {args.output_md}")  # 打印 Markdown 保存路径


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 执行主函数
