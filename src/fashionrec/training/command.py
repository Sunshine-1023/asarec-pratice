"""Train SASRec model via RecBole."""  # 通过 RecBole 训练 SASRec 模型的脚本

import argparse  # 导入命令行参数解析模块
import json  # 导入 JSON 序列化模块
from logging import getLogger  # 导入日志记录器获取函数
from pathlib import Path  # 导入路径处理类
import torch  # 导入 PyTorch 深度学习框架

from fashionrec.data.paths import ProcessedDataPaths
from fashionrec.training.checkpoints import install_validation_checkpoint_shortlist


def _read_model_name(config_path: Path) -> str:  # 从配置文件读取模型名称
    marker = "model:"  # 配置项前缀
    for line in config_path.read_text(encoding="utf-8").splitlines():  # 逐行读取配置
        striped = line.strip()  # 去除首尾空白
        if striped.startswith(marker):  # 匹配模型配置项
            model_name = striped.split(":", 1)[1].split("#", 1)[0].strip()  # 去注释后解析模型名
            if model_name:  # 模型名非空
                return model_name  # 返回模型名
    raise ValueError("model not found in config.")  # 未找到模型配置时报错


def _assert_benchmark_dataset_layout(config_path: Path) -> None:  # 校验基准数据集配置
    text = config_path.read_text(encoding="utf-8")  # 读取配置文件全文
    if "dataset: hm_seq" not in text:  # 检查数据集名称
        raise FileNotFoundError(  # 数据集配置不正确
            "Current config must use dataset: hm_seq for benchmark split training."  # 错误提示信息
        )  # 异常抛出
    if "benchmark_filename: [train, valid, test]" not in text:  # 检查基准文件名配置
        raise ValueError(  # 基准文件名配置不正确
            "Current config must set benchmark_filename: [train, valid, test]."  # 错误提示信息
        )  # 异常抛出


def _assert_prepared_training_data(model_name: str, data_paths: ProcessedDataPaths) -> None:
    required = [data_paths.seq_train_inter, data_paths.seq_valid_inter, data_paths.seq_test_inter]
    if model_name.upper() == "SASRECF":
        required.append(data_paths.seq_item)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Training data is not prepared. Run `make data RUN_ID=<id>` first. "
            f"Missing: {missing}"
        )


def _select_device() -> torch.device:  # 选择训练设备
    """Select training device with priority: cuda > mps > cpu."""  # 按 cuda > mps > cpu 优先级选择训练设备
    if torch.cuda.is_available():  # 若 CUDA 可用
        return torch.device("cuda")  # 使用 CUDA 设备
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # 若 MPS 可用
        return torch.device("mps")  # 使用 MPS 设备
    return torch.device("cpu")  # 默认使用 CPU 设备


def _patch_tqdm_single_line() -> None:  # 强制 tqdm 在单行内更新进度条
    """Force tqdm to update in one terminal line."""  # 强制 tqdm 在单行终端内更新
    try:  # 尝试导入 tqdm
        from tqdm.std import tqdm as tqdm_cls  # 导入 tqdm 标准类
    except Exception:  # 导入失败则跳过补丁
        return  # 直接返回

    if getattr(tqdm_cls, "_single_line_patch_applied", False):  # 若已应用过补丁
        return  # 避免重复打补丁

    original_init = tqdm_cls.__init__  # 保存原始初始化方法

    def _patched_init(self, *args, **kwargs):  # 定义补丁后的初始化方法
        kwargs.setdefault("leave", False)  # 默认完成后清除进度条
        kwargs.setdefault("position", 0)  # 默认固定在第一行
        kwargs.setdefault("dynamic_ncols", True)  # 默认动态列宽
        kwargs.setdefault("mininterval", 0.2)  # 默认最小刷新间隔 0.2 秒
        return original_init(self, *args, **kwargs)  # 调用原始初始化

    tqdm_cls.__init__ = _patched_init  # 替换 tqdm 初始化方法
    tqdm_cls._single_line_patch_applied = True  # 标记补丁已应用


def fit_model_without_test_evaluation(trainer, train_data, valid_data, *, show_progress: bool):
    """Fit and select coarse checkpoints on validation only; test is not accepted."""
    return trainer.fit(train_data, valid_data, saved=True, show_progress=show_progress)


def run_sasrec_with_device(  # 在选定设备上运行 SASRec 训练与验证
    config_path: Path,  # 配置文件路径
    model_name: str,  # 模型名称
    data_dir: Path | None = None,  # 明确的数据根目录；未给时保持 baseline 默认
    seed: int | None = None,  # 可选随机种子
    checkpoint_dir: Path | None = None,  # 可选 run-scoped checkpoint 目录
    checkpoint_shortlist_size: int = 5,
) -> tuple[float, dict, list[str]]:  # 返回 RecBole 验证结果与粗筛 checkpoint 列表
    from fashionrec.pytorch_compat import patch_recbole_compat  # 仅训练时需要 RecBole 兼容补丁

    patch_recbole_compat()  # 在导入 RecBole 前应用兼容补丁
    from recbole.config import Config  # 延迟导入可选训练依赖
    from recbole.data import create_dataset, data_preparation  # 延迟导入数据集工具
    from recbole.utils import get_model, get_trainer, init_logger, init_seed  # 延迟导入训练工具

    _patch_tqdm_single_line()  # 应用 tqdm 单行补丁

    selected_device = _select_device()  # 选择训练设备
    use_gpu = selected_device.type == "cuda"  # 判断是否使用 GPU
    gpu_id = "0" if use_gpu else ""  # 设置 GPU 设备 ID

    config_dict = {"use_gpu": use_gpu, "gpu_id": gpu_id}  # 构建设备相关配置字典
    if data_dir is not None:
        config_dict["data_path"] = str(data_dir)
    if seed is not None:  # 若指定了随机种子
        config_dict["seed"] = seed  # 将种子写入配置
    if checkpoint_dir is not None:  # 正式流水线覆盖全局 checkpoint 目录
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config_dict["checkpoint_dir"] = str(checkpoint_dir)

    config = Config(  # 创建 RecBole 配置对象
        model=model_name,  # 指定模型名称
        config_file_list=[str(config_path)],  # 加载配置文件
        config_dict=config_dict,  # 传入覆盖配置
    )  # 配置创建完成
    config.final_config_dict["device"] = selected_device  # 强制设置训练设备
    # Force tqdm progress bars in terminal during train/eval.  # 训练与评估时在终端显示 tqdm 进度条
    config.final_config_dict["show_progress"] = True  # 启用进度条显示

    init_seed(config["seed"], config["reproducibility"])  # 初始化全局随机种子
    init_logger(config)  # 初始化 RecBole 日志
    logger = getLogger()  # 获取日志记录器
    logger.info(f"Selected device: {selected_device}")  # 记录所选设备
    logger.info(f"Seed: {config['seed']}")  # 记录随机种子

    dataset = create_dataset(config)  # 创建数据集
    train_data, valid_data, _test_data = data_preparation(config, dataset)  # test 仅由 RecBole 构建，不在训练阶段评估

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])  # 再次初始化种子保证可复现
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])  # 创建模型并移至设备
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)  # 创建训练器
    shortlist_dir = Path(config["checkpoint_dir"]) / f"shortlist_seed_{config['seed']}"
    snapshots = install_validation_checkpoint_shortlist(
        trainer,
        shortlist_dir,
        max_candidates=checkpoint_shortlist_size,
    )
    best_valid_score, best_valid_result = fit_model_without_test_evaluation(
        trainer,
        train_data,
        valid_data,
        show_progress=config["show_progress"],
    )

    logger.info(f"best valid score: {best_valid_score}")  # 记录最佳验证分数
    logger.info(f"best valid result: {best_valid_result}")  # 记录最佳验证结果
    logger.info(f"checkpoint shortlist: {[str(path) for path in snapshots]}")
    return best_valid_score, best_valid_result, [str(path.resolve()) for path in snapshots]


def _parse_seeds(seed: int | None, seeds: str | None) -> list[int]:  # 解析单种子或多种子参数
    if seeds:  # 若提供了多种子字符串
        parsed = [int(token.strip()) for token in seeds.split(",") if token.strip()]  # 按逗号分割并解析整数
        if not parsed:  # 解析结果为空
            raise ValueError("--seeds is provided but empty.")  # 抛出值错误
        return parsed  # 返回种子列表
    if seed is not None:  # 若提供了单种子
        return [seed]  # 返回单元素种子列表
    return []  # 未指定种子则返回空列表


def _metrics_to_float_dict(metrics: dict) -> dict:  # 将指标字典值转为浮点数
    return {key: float(value) for key, value in metrics.items()}  # 逐项转换并返回


def main(  # 命令行入口函数
    argv: list[str] | None = None,  # 显式命令参数
    *,  # 后续参数仅允许关键字传递
    default_config: str | Path = "configs/sasrec.yaml",  # 默认模型配置
) -> None:  # 无返回值
    parser = argparse.ArgumentParser(prog="fashionrec train", description="Train SASRec/SASRecF on H&M data")  # 创建参数解析器
    parser.add_argument("--config", default=str(default_config))  # 配置文件路径参数
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Processed dataset root containing hm/ and hm_seq/; defaults to data/processed for baseline compatibility.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Run one custom seed")  # 单种子参数
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Override RecBole checkpoint directory")
    parser.add_argument(
        "--checkpoint-shortlist-size",
        type=int,
        default=5,
        help="Keep the best N RecBole-validation checkpoints for user-week selection",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/evaluation/sasrec_multi_seed_results.json"),
        help="Training result JSON path",
    )
    parser.add_argument(  # 多种子参数
        "--seeds",  # 参数名
        type=str,  # 字符串类型
        default=None,  # 默认不指定
        help="Run multiple seeds, comma-separated (e.g., 2024,2025,2026)",  # 帮助文本
    )  # 多种子参数结束
    args = parser.parse_args(argv)  # 解析显式命令参数

    config_path = Path(args.config)  # 配置文件路径对象
    if not config_path.exists():  # 配置文件不存在
        raise FileNotFoundError(f"Config not found: {config_path}")  # 抛出文件未找到错误

    _assert_benchmark_dataset_layout(config_path)  # 校验基准数据集配置

    model_name = _read_model_name(config_path)  # 从配置读取模型名称
    data_paths = ProcessedDataPaths.from_root(args.data_dir)
    _assert_prepared_training_data(model_name, data_paths)  # 训练不再隐式重建或覆盖数据产物

    seed_list = _parse_seeds(args.seed, args.seeds)  # 解析种子列表
    if not seed_list:  # 未指定种子时单次运行
        run_sasrec_with_device(
            config_path,
            model_name=model_name,
            data_dir=data_paths.root,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_shortlist_size=args.checkpoint_shortlist_size,
        )
        return  # 直接返回

    all_results: list[dict] = []  # 初始化多种子结果列表
    for run_seed in seed_list:  # 遍历每个种子
        best_valid_score, best_valid_result, checkpoint_candidates = run_sasrec_with_device(  # 以当前种子训练验证
            config_path,
            model_name=model_name,
            data_dir=data_paths.root,
            seed=run_seed,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_shortlist_size=args.checkpoint_shortlist_size,
        )  # 单次种子运行完成
        all_results.append(  # 追加当前种子结果
            {  # 结果字典
                "seed": run_seed,  # 当前种子值
                "best_valid_score": float(best_valid_score),  # 最佳验证分数
                "best_valid_result": _metrics_to_float_dict(best_valid_result),  # 最佳验证指标
                "checkpoint_candidates": checkpoint_candidates,
            }  # 结果字典结束
        )  # 追加完成

    report_path = args.report_path  # 多种子结果报告路径
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    report_path.write_text(  # 写入多种子结果 JSON
        json.dumps({"config": str(config_path), "results": all_results}, indent=2),  # 序列化结果数据
        encoding="utf-8",  # 使用 UTF-8 编码
    )  # 写入完成
    print(f"Saved multi-seed results: {report_path}")  # 打印保存路径


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用主函数
