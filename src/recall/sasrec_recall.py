"""SASRec recall export utilities."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from recbole.quick_start import load_data_and_model

DEFAULT_CKPT_DIR = Path("outputs/checkpoints/sasrec")
DEFAULT_OUTPUT_DIR = Path("outputs/recommendations")
VALID_INTER = Path("data/processed/hm/hm.valid.inter")
TEST_INTER = Path("data/processed/hm/hm.test.inter")


def default_output_path(eval_split: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"sasrec_{eval_split}.csv"


def _latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(
        checkpoint_dir.glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. Run SASRec training first."
        )
    return candidates[0]


def _batched(items: list[tuple[int, int]], batch_size: int) -> Iterable[list[tuple[int, int]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _as_str(x: object) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _load_eval_users(eval_split: str) -> list[str]:
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")

    path = VALID_INTER if eval_split == "valid" else TEST_INTER
    if not path.exists():
        raise FileNotFoundError(f"Missing eval split file: {path}")

    df = pd.read_csv(path, sep="\t", usecols=["user_id:token"])
    return sorted(df["user_id:token"].astype(str).unique().tolist())


def _resolve_user_rows(eval_dataset, uid_internal_list: list[int]) -> list[tuple[int, int]]:
    """Map each user to the first interaction row in the eval dataset."""
    uid_field = eval_dataset.uid_field
    uid_values = eval_dataset.inter_feat[uid_field].numpy()

    rows: list[tuple[int, int]] = []
    for uid in uid_internal_list:
        matches = np.where(uid_values == uid)[0]
        if len(matches) > 0:
            rows.append((uid, int(matches[0])))
    return rows


def export_sasrec_recall(
    eval_split: str = "valid",
    model_file: str | Path | None = None,
    output_path: str | Path | None = None,
    top_k: int = 100,
    batch_size: int = 512,
) -> Path:
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
    """
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")

    output_path = Path(output_path or default_output_path(eval_split))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_file is None:
        model_file = _latest_checkpoint(DEFAULT_CKPT_DIR)
    else:
        model_file = Path(model_file)
        if not model_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_file}")

    config, model, _, _, valid_data, test_data = load_data_and_model(model_file=str(model_file))
    eval_data = valid_data if eval_split == "valid" else test_data
    eval_dataset = eval_data.dataset
    uid_field = eval_dataset.uid_field
    iid_field = eval_dataset.iid_field
    device = config["device"]

    eval_users = _load_eval_users(eval_split)
    token2id = eval_dataset.field2token_id[uid_field]
    internal_uids = [token2id[user_id] for user_id in eval_users if user_id in token2id]
    user_rows = _resolve_user_rows(eval_dataset, internal_uids)

    missing_users = len(eval_users) - len(user_rows)
    if missing_users > 0:
        print(f"Warning: {missing_users:,} eval users not found in RecBole {eval_split} dataset.")

    model.eval()
    total_rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "score", "rank", "channel"])

        for batch in _batched(user_rows, batch_size):
            uids = [uid for uid, _ in batch]
            row_indices = torch.tensor([idx for _, idx in batch], dtype=torch.long)
            input_interaction = eval_dataset[row_indices].to(device)

            with torch.no_grad():
                scores = model.full_sort_predict(input_interaction)
            scores = scores.view(-1, eval_dataset.item_num)
            scores[:, 0] = -np.inf

            topk_scores, topk_iids = torch.topk(scores, top_k)
            uid_tokens = eval_dataset.id2token(uid_field, np.array(uids))
            iid_tokens = eval_dataset.id2token(iid_field, topk_iids.cpu().numpy())
            score_mat = topk_scores.cpu().numpy()

            for i in range(len(batch)):
                user_id = _as_str(uid_tokens[i])
                for rank in range(top_k):
                    item_id = _as_str(iid_tokens[i][rank])
                    score = float(score_mat[i][rank])
                    writer.writerow([user_id, item_id, score, rank + 1, "sasrec"])
                    total_rows += 1

    print(f"Eval split: {eval_split}")
    print(f"Model checkpoint: {model_file}")
    print(f"Users exported: {len(user_rows):,}")
    print(f"Rows exported: {total_rows:,}")
    print(f"Saved SASRec recall to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SASRec top-k recall to CSV")
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")
    parser.add_argument("--model-file", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    export_sasrec_recall(
        eval_split=args.eval_split,
        model_file=args.model_file,
        output_path=args.output_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
