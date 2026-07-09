"""RecBole-protocol weighted fusion with grid search."""  # RecBole 协议加权融合与网格搜索模块

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
DEFAULT_OUTPUT_JSON = Path("outputs/evaluation/recbole_fusion_weight_search.json")  # 默认融合搜索结果输出路径


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


def _build_traditional_model(  # 构建并训练传统推荐模型
    model_name: str,  # 模型名称（Pop / ItemKNN）
    shared: dict[str, Any],  # 共享数据上下文
    config_path: Path,  # 配置文件路径
    fit_epochs: int = 1,  # 训练轮数
):  # 函数签名结束
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
    return model  # 返回训练好的模型


def _build_sasrec_model(shared: dict[str, Any], model_file: Path):  # 从检查点加载 SASRec 模型
    config = shared["config"]  # 获取共享配置
    model = get_model("SASRec")(config, shared["dataset"]).to(config["device"])  # 创建 SASRec 模型并移至设备
    checkpoint = torch.load(str(model_file), map_location=config["device"])  # 加载检查点文件
    if "state_dict" not in checkpoint:  # 检查点缺少 state_dict 键
        raise KeyError(f"Checkpoint missing 'state_dict': {model_file}")  # 抛出键错误
    model.load_state_dict(checkpoint["state_dict"])  # 加载模型权重
    return model  # 返回加载好的模型


class WeightedFusionModel(torch.nn.Module):  # 加权融合模型包装类
    """Fusion wrapper that combines three full-sort score vectors."""  # 融合三个全排序得分向量的包装器

    def __init__(  # 初始化融合模型
        self,  # 实例自身
        sasrec_model: torch.nn.Module,  # SASRec 子模型
        pop_model: torch.nn.Module,  # 热门推荐子模型
        itemknn_model: torch.nn.Module,  # ItemKNN 子模型
        weights: tuple[float, float, float],  # 三通道权重元组
    ) -> None:  # 返回类型为 None
        super().__init__()  # 调用父类初始化
        self.sasrec_model = sasrec_model  # 保存 SASRec 子模型
        self.pop_model = pop_model  # 保存热门子模型
        self.itemknn_model = itemknn_model  # 保存 ItemKNN 子模型
        self.set_weights(weights)  # 设置融合权重

    def set_weights(self, weights: tuple[float, float, float]) -> None:  # 更新三通道融合权重
        self.w_pop, self.w_itemknn, self.w_sasrec = weights  # 解包并赋值权重

    def to(self, device):  # noqa: ANN001  # 将子模型移至指定设备
        self.sasrec_model.to(device)  # 移动 SASRec 模型
        self.pop_model.to(device)  # 移动热门模型
        self.itemknn_model.to(device)  # 移动 ItemKNN 模型
        return self  # 返回自身以支持链式调用

    def eval(self):  # 将所有子模型设为评估模式
        self.sasrec_model.eval()  # SASRec 进入评估模式
        self.pop_model.eval()  # 热门模型进入评估模式
        self.itemknn_model.eval()  # ItemKNN 进入评估模式
        return self  # 返回自身

    def full_sort_predict(self, interaction):  # 全排序预测并融合三通道得分
        sasrec_scores = self.sasrec_model.full_sort_predict(interaction)  # 获取 SASRec 全排序得分
        pop_scores = self.pop_model.full_sort_predict(interaction)  # 获取热门全排序得分
        itemknn_scores = self.itemknn_model.full_sort_predict(interaction)  # 获取 ItemKNN 全排序得分

        target_numel = sasrec_scores.numel()  # 以 SASRec 得分元素数为基准
        if pop_scores.numel() != target_numel or itemknn_scores.numel() != target_numel:  # 检查得分形状是否一致
            raise RuntimeError(  # 形状不一致时抛出运行时错误
                "Fusion score shape mismatch: "  # 错误信息前缀
                f"sasrec={tuple(sasrec_scores.shape)}, "  # SASRec 得分形状
                f"pop={tuple(pop_scores.shape)}, "  # 热门得分形状
                f"itemknn={tuple(itemknn_scores.shape)}"  # ItemKNN 得分形状
            )  # 错误信息结束

        pop_scores = pop_scores.reshape(sasrec_scores.shape)  # 将热门得分重塑为与 SASRec 相同形状
        itemknn_scores = itemknn_scores.reshape(sasrec_scores.shape)  # 将 ItemKNN 得分重塑为相同形状
        fused_scores = (  # 计算加权融合得分
            self.w_pop * pop_scores  # 热门通道加权得分
            + self.w_itemknn * itemknn_scores  # ItemKNN 通道加权得分
            + self.w_sasrec * sasrec_scores  # SASRec 通道加权得分
        )  # 融合得分计算完成
        return fused_scores.reshape(-1)  # 展平为一维向量并返回


def parse_weight_grid(weight_grid: str | None) -> list[tuple[float, float, float]]:  # 解析权重网格搜索空间
    def normalize(weights: tuple[float, float, float]) -> tuple[float, float, float]:  # 归一化权重使总和为 1
        if any(w < 0 for w in weights):  # 检查是否有负权重
            raise ValueError(f"Weight must be >= 0, got {weights}")  # 负权重非法
        total = sum(weights)  # 计算权重总和
        if total <= 0:  # 总和必须为正
            raise ValueError(f"Weight sum must be > 0, got {weights}")  # 总和为零或负非法
        return (weights[0] / total, weights[1] / total, weights[2] / total)  # 返回归一化后的权重

    if not weight_grid:  # 未提供自定义权重网格时使用默认组合
        defaults = [  # 默认权重组合列表
            (0.10, 0.20, 0.70),  # 默认组合 1
            (0.10, 0.30, 0.60),  # 默认组合 2
            (0.10, 0.40, 0.50),  # 默认组合 3
            (0.20, 0.20, 0.60),  # 默认组合 4
            (0.20, 0.30, 0.50),  # 默认组合 5
            (0.20, 0.40, 0.40),  # 默认组合 6
            (0.30, 0.20, 0.50),  # 默认组合 7
            (0.30, 0.30, 0.40),  # 默认组合 8
        ]  # 默认组合列表结束
        return [normalize(w) for w in defaults]  # 归一化所有默认组合并返回
    combos: list[tuple[float, float, float]] = []  # 初始化自定义组合列表
    for part in weight_grid.split(";"):  # 按分号分割各权重元组
        p, i, s = (x.strip() for x in part.split(","))  # 按逗号分割并去除空白
        combos.append(normalize((float(p), float(i), float(s))))  # 解析浮点数并归一化后追加
    return combos  # 返回全部权重组合


def main() -> None:  # 命令行入口函数
    parser = argparse.ArgumentParser(description="RecBole-protocol fusion grid search")  # 创建参数解析器
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)  # 配置文件路径参数
    parser.add_argument("--model-file", type=Path, default=None)  # 可选模型检查点文件参数
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)  # 检查点目录参数
    parser.add_argument("--fit-epochs", type=int, default=1)  # 传统模型训练轮数参数
    parser.add_argument(  # 权重网格参数
        "--weight-grid",  # 参数名
        type=str,  # 字符串类型
        default=None,  # 默认使用内置网格
        help="Semicolon-separated tuples: p,i,s;p,i,s",  # 帮助文本
    )  # 权重网格参数结束
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)  # 输出 JSON 路径参数
    args = parser.parse_args()  # 解析命令行参数

    model_file = args.model_file or _latest_checkpoint(args.checkpoint_dir)  # 确定 SASRec 检查点路径
    if not model_file.exists():  # 检查点文件不存在
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")  # 抛出文件未找到异常

    shared = build_shared_context(args.config)  # 构建共享数据上下文
    sas_config = shared["config"]  # 获取基准配置
    sas_model = _build_sasrec_model(shared, model_file)  # 加载 SASRec 模型
    pop_model = _build_traditional_model(  # 构建并训练热门模型
        "Pop", shared, args.config, fit_epochs=args.fit_epochs  # 传入模型名与训练轮数
    )  # 热门模型构建完成
    itemknn_model = _build_traditional_model(  # 构建并训练 ItemKNN 模型
        "ItemKNN", shared, args.config, fit_epochs=args.fit_epochs  # 传入模型名与训练轮数
    )  # ItemKNN 模型构建完成

    fusion_model = WeightedFusionModel(  # 创建加权融合模型
        sasrec_model=sas_model,  # 传入 SASRec 子模型
        pop_model=pop_model,  # 传入热门子模型
        itemknn_model=itemknn_model,  # 传入 ItemKNN 子模型
        weights=(0.2, 0.3, 0.5),  # 初始权重（后续由网格搜索覆盖）
    ).to(sas_config["device"]).eval()  # 移至设备并设为评估模式

    fusion_trainer = get_trainer(sas_config["MODEL_TYPE"], sas_config["model"])(  # 创建融合模型训练器
        sas_config, fusion_model  # 传入配置与融合模型
    )  # 训练器创建完成

    grid = parse_weight_grid(args.weight_grid)  # 解析权重搜索网格
    rows = []  # 初始化各权重组合的评估结果列表
    best_row = None  # 初始化最佳结果行
    for w_pop, w_itemknn, w_sasrec in grid:  # 遍历每个权重组合
        print(  # 打印当前评估的权重
            f"Evaluating normalized weights: popular={w_pop:.4f}, "  # 热门权重
            f"itemknn={w_itemknn:.4f}, sasrec={w_sasrec:.4f}"  # ItemKNN 与 SASRec 权重
        )  # 打印结束
        fusion_model.set_weights((w_pop, w_itemknn, w_sasrec))  # 设置当前权重组合
        valid_metrics = _to_float_metrics(  # 在验证集上评估并转为浮点指标
            fusion_trainer.evaluate(  # 调用训练器评估
                shared["valid_data"], load_best_model=False, show_progress=False  # 验证集评估参数
            )  # 评估调用结束
        )  # 指标转换完成
        row = {  # 构建当前权重组合的结果行
            "weights": {  # 权重字典
                "popular": w_pop,  # 热门权重
                "itemknn": w_itemknn,  # ItemKNN 权重
                "sasrec": w_sasrec,  # SASRec 权重
            },  # 权重字典结束
            "valid_metrics": valid_metrics,  # 验证集指标
        }  # 结果行结束
        rows.append(row)  # 追加到结果列表
        if best_row is None or row["valid_metrics"].get("map@12", 0.0) > best_row[  # 比较 MAP@12 更新最佳结果
            "valid_metrics"  # 最佳行的验证指标
        ].get("map@12", 0.0):  # 最佳行的 MAP@12 值
            best_row = row  # 更新最佳结果行

    assert best_row is not None  # 断言至少有一个评估结果
    w = best_row["weights"]  # 获取最佳权重
    fusion_model.set_weights((w["popular"], w["itemknn"], w["sasrec"]))  # 将融合模型设为最佳权重
    test_metrics = _to_float_metrics(  # 在测试集上评估最佳权重组合
        fusion_trainer.evaluate(shared["test_data"], load_best_model=False, show_progress=False)  # 测试集评估
    )  # 测试指标转换完成

    payload = {  # 构建输出载荷字典
        "protocol": "recbole_full_ranking_hm_seq",  # 评估协议标识
        "dataset": str(sas_config["dataset"]),  # 数据集名称
        "checkpoint": str(model_file.resolve()),  # SASRec 检查点绝对路径
        "grid": rows,  # 全部权重组合的验证集结果
        "best": {  # 最佳权重组合详情
            "weights": best_row["weights"],  # 最佳权重
            "valid_metrics": best_row["valid_metrics"],  # 最佳验证集指标
            "test_metrics": test_metrics,  # 最佳权重在测试集上的指标
        },  # 最佳结果结束
    }  # 载荷字典结束
    args.output_json.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入 JSON 文件
    print(f"Saved RecBole fusion grid search: {args.output_json}")  # 打印保存路径
    print(json.dumps(payload["best"], ensure_ascii=False, indent=2))  # 打印最佳结果


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 执行主函数
