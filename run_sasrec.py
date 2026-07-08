"""Train SASRec model via RecBole."""

import argparse
import csv
from logging import getLogger
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed

from src.data.preprocess import build_inter_file
from src.data.split import split_by_time

SOURCE_DIR = Path("data/processed/hm")
TARGET_DIR = Path("data/processed/hm_seq")
TRAIN_SPLIT_FILE = SOURCE_DIR / "hm.train.inter"
VALID_SPLIT_FILE = SOURCE_DIR / "hm.valid.inter"
TEST_SPLIT_FILE = SOURCE_DIR / "hm.test.inter"
RECB_TRAIN_FILE = TARGET_DIR / "hm_seq.train.inter"
RECB_VALID_FILE = TARGET_DIR / "hm_seq.valid.inter"
RECB_TEST_FILE = TARGET_DIR / "hm_seq.test.inter"


def _convert_to_seq_samples(
    source_path: Path,
    target_path: Path,
    history_map: dict[str, list[str]],
    max_item_list_length: int,
    rolling_within_split: bool,
    advance_history_after_split: bool,
) -> int:
    df = pd.read_csv(
        source_path,
        sep="\t",
        usecols=["user_id:token", "item_id:token", "timestamp:float"],
    )
    df = df.sort_values(["user_id:token", "timestamp:float"])

    rows_written = 0
    split_items_by_user: dict[str, list[str]] = defaultdict(list)
    with target_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "user_id:token",
                "item_id_list:token_seq",
                "item_length:float",
                "item_id:token",
                "timestamp:float",
            ]
        )

        for user_id, item_id, timestamp in df.itertuples(index=False, name=None):
            user_id = str(user_id)
            item_id = str(item_id)
            hist = history_map[user_id]

            if hist:
                seq_items = hist[-max_item_list_length:]
                writer.writerow(
                    [user_id, " ".join(seq_items), len(seq_items), item_id, timestamp]
                )
                rows_written += 1

            if rolling_within_split:
                hist.append(item_id)
            elif advance_history_after_split:
                split_items_by_user[user_id].append(item_id)

    if advance_history_after_split and split_items_by_user:
        for user_id, items in split_items_by_user.items():
            history_map[user_id].extend(items)

    return rows_written


def prepare_recbole_benchmark_files(max_item_list_length: int) -> tuple[Path, Path, Path]:
    """Build benchmark train/valid/test files with SASRec sequence columns."""
    for path in (TRAIN_SPLIT_FILE, VALID_SPLIT_FILE, TEST_SPLIT_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run preprocessing/splitting first.")

    targets = (RECB_TRAIN_FILE, RECB_VALID_FILE, RECB_TEST_FILE)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    history_map: dict[str, list[str]] = defaultdict(list)

    train_rows = _convert_to_seq_samples(
        TRAIN_SPLIT_FILE,
        RECB_TRAIN_FILE,
        history_map,
        max_item_list_length,
        rolling_within_split=True,
        advance_history_after_split=False,
    )
    valid_rows = _convert_to_seq_samples(
        VALID_SPLIT_FILE,
        RECB_VALID_FILE,
        history_map,
        max_item_list_length,
        rolling_within_split=False,
        advance_history_after_split=True,
    )
    test_rows = _convert_to_seq_samples(
        TEST_SPLIT_FILE,
        RECB_TEST_FILE,
        history_map,
        max_item_list_length,
        rolling_within_split=False,
        advance_history_after_split=False,
    )

    print(f"Prepared benchmark train file: {RECB_TRAIN_FILE} ({train_rows:,} rows)")
    print(f"Prepared benchmark valid file: {RECB_VALID_FILE} ({valid_rows:,} rows)")
    print(f"Prepared benchmark test file: {RECB_TEST_FILE} ({test_rows:,} rows)")
    return targets


def _read_max_item_list_length(config_path: Path) -> int:
    marker = "MAX_ITEM_LIST_LENGTH:"
    for line in config_path.read_text(encoding="utf-8").splitlines():
        striped = line.strip()
        if striped.startswith(marker):
            return int(striped.split(":", 1)[1].strip())
    raise ValueError("MAX_ITEM_LIST_LENGTH not found in config.")


def _assert_benchmark_dataset_layout(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    if "dataset: hm_seq" not in text:
        raise FileNotFoundError(
            "Current config must use dataset: hm_seq for benchmark split training."
        )
    if "benchmark_filename: [train, valid, test]" not in text:
        raise ValueError(
            "Current config must set benchmark_filename: [train, valid, test]."
        )


def _select_device() -> torch.device:
    """Select training device with priority: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_sasrec_with_device(config_path: Path) -> None:
    selected_device = _select_device()
    use_gpu = selected_device.type == "cuda"
    gpu_id = "0" if use_gpu else ""

    config = Config(
        model="SASRec",
        config_file_list=[str(config_path)],
        config_dict={"use_gpu": use_gpu, "gpu_id": gpu_id},
    )
    config.final_config_dict["device"] = selected_device

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(f"Selected device: {selected_device}")

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    logger.info(f"best valid score: {best_valid_score}")
    logger.info(f"best valid result: {best_valid_result}")
    logger.info(f"test result: {test_result}")


def main():
    parser = argparse.ArgumentParser(description="Train SASRec on H&M data")
    parser.add_argument("--config", default="configs/sasrec.yaml")
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _assert_benchmark_dataset_layout(config_path)

    if not args.skip_preprocess:
        build_inter_file()
        split_by_time()

    max_item_list_length = _read_max_item_list_length(config_path)
    prepare_recbole_benchmark_files(max_item_list_length)

    run_sasrec_with_device(config_path)


if __name__ == "__main__":
    main()
