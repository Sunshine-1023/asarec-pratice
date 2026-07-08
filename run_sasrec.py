"""Train SASRec model via RecBole."""  # 通过 RecBole 训练 SASRec 模型的脚本

import argparse
import csv
import json
import re
from logging import getLogger
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch

from src.pytorch_compat import patch_recbole_compat

patch_recbole_compat()

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed

from src.data.preprocess import build_inter_file  # 构建交互文件
from src.data.split import split_by_time  # 按时间划分数据集

SOURCE_DIR = Path("data/processed/hm")  # 原始划分数据目录
TARGET_DIR = Path("data/processed/hm_seq")  # 序列化数据输出目录
TRAIN_SPLIT_FILE = SOURCE_DIR / "hm.train.inter"  # 训练集划分文件
VALID_SPLIT_FILE = SOURCE_DIR / "hm.valid.inter"  # 验证集划分文件
TEST_SPLIT_FILE = SOURCE_DIR / "hm.test.inter"  # 测试集划分文件
RECB_TRAIN_FILE = TARGET_DIR / "hm_seq.train.inter"  # RecBole 训练文件
RECB_VALID_FILE = TARGET_DIR / "hm_seq.valid.inter"  # RecBole 验证文件
RECB_TEST_FILE = TARGET_DIR / "hm_seq.test.inter"  # RecBole 测试文件


def _normalize_item_id(value: object) -> str:  # 统一商品 ID 格式为 10 位字符串
    text = str(value).strip()  # 转字符串并去掉首尾空白
    text = re.sub(r"\.0+$", "", text)  # 兼容 CSV 里被解析成浮点字符串的情况
    return text.zfill(10)  # 左侧补零到 10 位


def _convert_to_seq_samples(  # 将交互数据转换为序列样本
    source_path: Path,  # 源文件路径
    target_path: Path,  # 目标文件路径
    history_map: dict[str, list[str]],  # 用户历史记录映射
    max_item_list_length: int,  # 最大序列长度
    rolling_within_split: bool,  # 是否在划分内滚动更新历史
    advance_history_after_split: bool,  # 是否在划分结束后批量更新历史
) -> int:  # 返回写入行数
    df = pd.read_csv(  # 读取源 CSV 文件
        source_path,  # 源文件路径
        sep="\t",  # 制表符分隔
        usecols=["user_id:token", "item_id:token", "timestamp:float"],  # 只读取需要的列
    )  # 读取完成
    df = df.sort_values(["user_id:token", "timestamp:float"])  # 按用户和时间戳排序

    rows_written = 0  # 初始化写入行计数
    split_items_by_user: dict[str, list[str]] = defaultdict(list)  # 划分内待追加的历史物品
    with target_path.open("w", newline="", encoding="utf-8") as f:  # 打开目标文件写入
        writer = csv.writer(f, delimiter="\t")  # 创建 TSV 写入器
        writer.writerow(  # 写入表头
            [  # 表头列名列表
                "user_id:token",  # 用户 ID
                "item_id_list:token_seq",  # 历史物品序列
                "item_length:float",  # 序列长度
                "item_id:token",  # 目标物品 ID
                "timestamp:float",  # 时间戳
            ]  # 表头列名列表结束
        )  # 表头写入完成

        for user_id, item_id, timestamp in df.itertuples(index=False, name=None):  # 遍历每条交互记录
            user_id = str(user_id)  # 用户 ID 转字符串
            item_id = _normalize_item_id(item_id)  # 物品 ID 统一为 10 位字符串
            hist = history_map[user_id]  # 获取该用户当前历史

            if hist:  # 若历史非空则生成样本
                seq_items = hist[-max_item_list_length:]  # 截取最近 max_item_list_length 个物品
                writer.writerow(  # 写入一行序列样本
                    [user_id, " ".join(seq_items), len(seq_items), item_id, timestamp]  # 样本字段
                )  # 行写入完成
                rows_written += 1  # 写入行数加一

            if rolling_within_split:  # 划分内滚动模式
                hist.append(item_id)  # 立即将当前物品加入历史
            elif advance_history_after_split:  # 划分结束后批量更新模式
                split_items_by_user[user_id].append(item_id)  # 暂存当前物品待后续追加

    if advance_history_after_split and split_items_by_user:  # 若有待批量追加的历史
        for user_id, items in split_items_by_user.items():  # 遍历每个用户
            history_map[user_id].extend(items)  # 将暂存物品追加到历史映射

    return rows_written  # 返回写入的总行数


def prepare_recbole_benchmark_files(max_item_list_length: int) -> tuple[Path, Path, Path]:  # 准备 RecBole 基准数据集文件
    """Build benchmark train/valid/test files with SASRec sequence columns."""  # 构建含 SASRec 序列列的基准训练/验证/测试文件
    for path in (TRAIN_SPLIT_FILE, VALID_SPLIT_FILE, TEST_SPLIT_FILE):  # 检查划分文件是否存在
        if not path.exists():  # 文件缺失
            raise FileNotFoundError(f"Missing {path}. Run preprocessing/splitting first.")  # 抛出文件未找到错误

    targets = (RECB_TRAIN_FILE, RECB_VALID_FILE, RECB_TEST_FILE)  # 目标输出文件元组

    TARGET_DIR.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    history_map: dict[str, list[str]] = defaultdict(list)  # 初始化用户历史映射

    train_rows = _convert_to_seq_samples(  # 转换训练集
        TRAIN_SPLIT_FILE,  # 训练源文件
        RECB_TRAIN_FILE,  # 训练目标文件
        history_map,  # 用户历史映射
        max_item_list_length,  # 最大序列长度
        rolling_within_split=True,  # 训练集内滚动更新历史
        advance_history_after_split=False,  # 不在划分结束后批量更新
    )  # 训练集转换完成
    valid_rows = _convert_to_seq_samples(  # 转换验证集
        VALID_SPLIT_FILE,  # 验证源文件
        RECB_VALID_FILE,  # 验证目标文件
        history_map,  # 用户历史映射
        max_item_list_length,  # 最大序列长度
        rolling_within_split=False,  # 验证集不滚动更新
        advance_history_after_split=True,  # 验证集结束后批量更新历史
    )  # 验证集转换完成
    test_rows = _convert_to_seq_samples(  # 转换测试集
        TEST_SPLIT_FILE,  # 测试源文件
        RECB_TEST_FILE,  # 测试目标文件
        history_map,  # 用户历史映射
        max_item_list_length,  # 最大序列长度
        rolling_within_split=False,  # 测试集不滚动更新
        advance_history_after_split=False,  # 测试集不批量更新历史
    )  # 测试集转换完成

    print(f"Prepared benchmark train file: {RECB_TRAIN_FILE} ({train_rows:,} rows)")  # 打印训练文件信息
    print(f"Prepared benchmark valid file: {RECB_VALID_FILE} ({valid_rows:,} rows)")  # 打印验证文件信息
    print(f"Prepared benchmark test file: {RECB_TEST_FILE} ({test_rows:,} rows)")  # 打印测试文件信息
    return targets  # 返回三个目标文件路径


def _read_max_item_list_length(config_path: Path) -> int:  # 从配置文件读取最大序列长度
    marker = "MAX_ITEM_LIST_LENGTH:"  # 配置项前缀
    for line in config_path.read_text(encoding="utf-8").splitlines():  # 逐行读取配置文件
        striped = line.strip()  # 去除首尾空白
        if striped.startswith(marker):  # 匹配目标配置项
            value = striped.split(":", 1)[1].split("#", 1)[0].strip()  # 去注释后解析数值
            return int(value)  # 返回整数值
    raise ValueError("MAX_ITEM_LIST_LENGTH not found in config.")  # 未找到配置项则报错


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


def _select_device() -> torch.device:  # 选择训练设备
    """Select training device with priority: cuda > mps > cpu."""  # 按 cuda > mps > cpu 优先级选择训练设备
    if torch.cuda.is_available():  # 若 CUDA 可用
        return torch.device("cuda")  # 使用 CUDA 设备
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # 若 MPS 可用
        return torch.device("mps")  # 使用 MPS 设备
    return torch.device("cpu")  # 默认使用 CPU 设备


def _patch_tqdm_single_line() -> None:
    """Force tqdm to update in one terminal line."""
    try:
        from tqdm.std import tqdm as tqdm_cls
    except Exception:
        return

    if getattr(tqdm_cls, "_single_line_patch_applied", False):
        return

    original_init = tqdm_cls.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("leave", False)
        kwargs.setdefault("position", 0)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 0.2)
        return original_init(self, *args, **kwargs)

    tqdm_cls.__init__ = _patched_init
    tqdm_cls._single_line_patch_applied = True


def run_sasrec_with_device(
    config_path: Path,
    model_name: str,
    seed: int | None = None,
) -> tuple[float, dict, dict]:
    _patch_tqdm_single_line()

    selected_device = _select_device()
    use_gpu = selected_device.type == "cuda"
    gpu_id = "0" if use_gpu else ""

    config_dict = {"use_gpu": use_gpu, "gpu_id": gpu_id}
    if seed is not None:
        config_dict["seed"] = seed

    config = Config(
        model=model_name,
        config_file_list=[str(config_path)],
        config_dict=config_dict,
    )
    config.final_config_dict["device"] = selected_device
    # Force tqdm progress bars in terminal during train/eval.
    config.final_config_dict["show_progress"] = True

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(f"Selected device: {selected_device}")
    logger.info(f"Seed: {config['seed']}")

    dataset = create_dataset(config)  # 创建数据集
    train_data, valid_data, test_data = data_preparation(config, dataset)  # 划分训练/验证/测试数据

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])  # 再次初始化种子保证可复现
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])  # 创建模型并移至设备
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)  # 创建训练器
    best_valid_score, best_valid_result = trainer.fit(  # 训练模型
        train_data, valid_data, saved=True, show_progress=config["show_progress"]  # 训练参数
    )  # 训练完成，返回最佳验证分数与结果
    test_result = trainer.evaluate(  # 在测试集上评估
        test_data, load_best_model=True, show_progress=config["show_progress"]  # 评估参数
    )  # 评估完成

    logger.info(f"best valid score: {best_valid_score}")
    logger.info(f"best valid result: {best_valid_result}")
    logger.info(f"test result: {test_result}")
    return best_valid_score, best_valid_result, test_result


def _parse_seeds(seed: int | None, seeds: str | None) -> list[int]:
    if seeds:
        parsed = [int(token.strip()) for token in seeds.split(",") if token.strip()]
        if not parsed:
            raise ValueError("--seeds is provided but empty.")
        return parsed
    if seed is not None:
        return [seed]
    return []


def _metrics_to_float_dict(metrics: dict) -> dict:
    return {key: float(value) for key, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Train SASRec on H&M data")
    parser.add_argument("--config", default="configs/sasrec.yaml")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Run one custom seed")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Run multiple seeds, comma-separated (e.g., 2024,2025,2026)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)  # 配置文件路径对象
    if not config_path.exists():  # 配置文件不存在
        raise FileNotFoundError(f"Config not found: {config_path}")  # 抛出文件未找到错误

    _assert_benchmark_dataset_layout(config_path)  # 校验基准数据集配置

    if not args.skip_preprocess:  # 未跳过预处理时
        build_inter_file()  # 构建交互文件
        split_by_time()  # 按时间划分数据集

    max_item_list_length = _read_max_item_list_length(config_path)  # 读取最大序列长度
    prepare_recbole_benchmark_files(max_item_list_length)  # 准备 RecBole 基准文件

    model_name = _read_model_name(config_path)
    seed_list = _parse_seeds(args.seed, args.seeds)
    if not seed_list:
        run_sasrec_with_device(config_path, model_name=model_name)
        return

    all_results: list[dict] = []
    for run_seed in seed_list:
        best_valid_score, best_valid_result, test_result = run_sasrec_with_device(
            config_path, model_name=model_name, seed=run_seed
        )
        all_results.append(
            {
                "seed": run_seed,
                "best_valid_score": float(best_valid_score),
                "best_valid_result": _metrics_to_float_dict(best_valid_result),
                "test_result": _metrics_to_float_dict(test_result),
            }
        )

    report_path = Path("outputs/evaluation/sasrec_multi_seed_results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"config": str(config_path), "results": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved multi-seed results: {report_path}")


if __name__ == "__main__":  # 脚本直接运行时
    main()  # 调用主函数
