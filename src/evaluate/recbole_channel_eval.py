"""Unified RecBole evaluation for SASRec / Pop / ItemKNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.pytorch_compat import patch_recbole_compat

patch_recbole_compat()

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed

DEFAULT_CONFIG = Path("configs/sasrec.yaml")
DEFAULT_CKPT_DIR = Path("outputs/checkpoints/sasrec")
DEFAULT_OUTPUT_JSON = Path("outputs/evaluation/recbole_channel_metrics.json")
DEFAULT_OUTPUT_MD = Path("outputs/evaluation/recbole_channel_comparison.md")
DEFAULT_FUSION_JSON = Path("outputs/evaluation/recbole_fusion_weight_search.json")


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


def _to_float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def _normalized_path(path_like: str | Path) -> str:
    return str(Path(path_like).resolve()).replace("\\", "/").lower()


def build_shared_context(config_path: Path) -> dict[str, Any]:
    """Build one shared dataset/dataloader context for all models."""
    base_config = Config(model="SASRec", config_file_list=[str(config_path)])
    init_seed(base_config["seed"], base_config["reproducibility"])
    dataset = create_dataset(base_config)
    train_data, valid_data, test_data = data_preparation(base_config, dataset)
    return {
        "config": base_config,
        "dataset": train_data._dataset,
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
    }


def build_sasrec_model(shared: dict[str, Any], model_file: Path):
    config = shared["config"]
    model = get_model("SASRec")(config, shared["dataset"]).to(config["device"])

    checkpoint = torch.load(str(model_file), map_location=config["device"])
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing 'state_dict': {model_file}")
    model.load_state_dict(checkpoint["state_dict"])
    return model


def evaluate_sasrec_recbole(shared: dict[str, Any], model_file: Path) -> dict[str, dict[str, float]]:
    config = shared["config"]
    model = build_sasrec_model(shared, model_file)
    trainer = get_trainer(config["MODEL_TYPE"], "SASRec")(config, model)
    valid_metrics = trainer.evaluate(shared["valid_data"], load_best_model=False, show_progress=False)
    test_metrics = trainer.evaluate(shared["test_data"], load_best_model=False, show_progress=False)
    return {
        "valid": _to_float_metrics(valid_metrics),
        "test": _to_float_metrics(test_metrics),
    }


def evaluate_traditional_recbole(
    model_name: str,
    shared: dict[str, Any],
    config_path: Path,
    fit_epochs: int = 1,
) -> dict[str, dict[str, float]]:
    """Evaluate traditional model on the same shared dataset/token space."""
    base_cfg = shared["config"]
    config = Config(
        model=model_name,
        config_file_list=[str(config_path)],
        config_dict={
            "model": model_name,
            "epochs": fit_epochs,
            "use_gpu": base_cfg["use_gpu"],
            "gpu_id": base_cfg["gpu_id"],
        },
    )
    init_seed(config["seed"], config["reproducibility"])

    model = get_model(model_name)(config, shared["dataset"]).to(base_cfg["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.fit(shared["train_data"], shared["valid_data"], saved=False, show_progress=False)

    valid_metrics = trainer.evaluate(shared["valid_data"], load_best_model=False, show_progress=False)
    test_metrics = trainer.evaluate(shared["test_data"], load_best_model=False, show_progress=False)
    return {
        "valid": _to_float_metrics(valid_metrics),
        "test": _to_float_metrics(test_metrics),
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _is_fusion_payload_compatible(
    fusion_payload: dict[str, Any],
    checkpoint_path: Path,
    dataset_name: str,
) -> tuple[bool, str]:
    if fusion_payload.get("protocol") != "recbole_full_ranking_hm_seq":
        return False, "protocol mismatch"

    fusion_ckpt = fusion_payload.get("checkpoint")
    if not fusion_ckpt:
        return False, "missing checkpoint in fusion payload"
    if _normalized_path(fusion_ckpt) != _normalized_path(checkpoint_path):
        return False, "checkpoint mismatch"

    fusion_dataset = fusion_payload.get("dataset")
    if fusion_dataset and fusion_dataset != dataset_name:
        return False, "dataset mismatch"
    return True, ""


def write_comparison_markdown(
    output_path: Path,
    channel_metrics: dict[str, dict[str, dict[str, float]]],
    checkpoint_path: Path,
    dataset_name: str,
    fusion_file: Path,
) -> None:
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for model_name in ("SASRec", "Pop", "ItemKNN"):
        metrics = channel_metrics[model_name]
        for split in ("valid", "test"):
            m = metrics[split]
            rows.append(
                (
                    model_name,
                    split,
                    _fmt(m.get("map@12")),
                    _fmt(m.get("recall@12")),
                    _fmt(m.get("ndcg@12")),
                    _fmt(m.get("hit@12")),
                    _fmt(m.get("precision@12")),
                )
            )

    fusion_section = ""
    fusion_skip_reason = ""
    if fusion_file.exists():
        fusion_payload = json.loads(fusion_file.read_text(encoding="utf-8"))
        ok, reason = _is_fusion_payload_compatible(
            fusion_payload=fusion_payload,
            checkpoint_path=checkpoint_path,
            dataset_name=dataset_name,
        )
        if ok:
            best = fusion_payload.get("best", {})
            w = best.get("weights", {})
            valid_m = best.get("valid_metrics", {})
            test_m = best.get("test_metrics", {})
            fusion_section = (
                "\n## Fusion (RecBole)\n\n"
                f"- Best weights: `popular={w.get('popular', 0):.2f}, "
                f"itemknn={w.get('itemknn', 0):.2f}, sasrec={w.get('sasrec', 0):.2f}`\n\n"
                "| Model | Split | MAP@12 | Recall@12 | NDCG@12 | Hit@12 | Precision@12 |\n"
                "|-------|-------|-------:|----------:|--------:|-------:|-------------:|\n"
                f"| Fusion | valid | {_fmt(valid_m.get('map@12'))} | {_fmt(valid_m.get('recall@12'))} | "
                f"{_fmt(valid_m.get('ndcg@12'))} | {_fmt(valid_m.get('hit@12'))} | "
                f"{_fmt(valid_m.get('precision@12'))} |\n"
                f"| Fusion | test | {_fmt(test_m.get('map@12'))} | {_fmt(test_m.get('recall@12'))} | "
                f"{_fmt(test_m.get('ndcg@12'))} | {_fmt(test_m.get('hit@12'))} | "
                f"{_fmt(test_m.get('precision@12'))} |\n"
            )
        else:
            fusion_skip_reason = reason

    lines = [
        "# RecBole Unified Comparison",
        "",
        "Unified protocol: RecBole full ranking on `hm_seq` benchmark split.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- SASRec checkpoint: `{checkpoint_path}`",
        "",
        "## Channel Metrics",
        "",
        "| Model | Split | MAP@12 | Recall@12 | NDCG@12 | Hit@12 | Precision@12 |",
        "|-------|-------|-------:|----------:|--------:|-------:|-------------:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    if fusion_section:
        lines.extend(["", fusion_section.strip(), ""])
    elif fusion_skip_reason:
        lines.extend(
            [
                "",
                "## Fusion (RecBole)",
                "",
                f"- Skipped fusion section due to `{fusion_skip_reason}`.",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SASRec/Pop/ItemKNN in RecBole protocol")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-file", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--fit-epochs", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--fusion-json", type=Path, default=DEFAULT_FUSION_JSON)
    args = parser.parse_args()

    model_file = args.model_file or _latest_checkpoint(args.checkpoint_dir)
    if not model_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")

    shared = build_shared_context(args.config)
    dataset_name = str(shared["config"]["dataset"])

    sasrec_metrics = evaluate_sasrec_recbole(shared, model_file)
    pop_metrics = evaluate_traditional_recbole(
        "Pop", shared, args.config, fit_epochs=args.fit_epochs
    )
    itemknn_metrics = evaluate_traditional_recbole(
        "ItemKNN", shared, args.config, fit_epochs=args.fit_epochs
    )

    payload = {
        "protocol": "recbole_full_ranking_hm_seq",
        "dataset": dataset_name,
        "checkpoint": str(model_file.resolve()),
        "models": {
            "SASRec": sasrec_metrics,
            "Pop": pop_metrics,
            "ItemKNN": itemknn_metrics,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved RecBole channel metrics: {args.output_json}")

    write_comparison_markdown(
        output_path=args.output_md,
        channel_metrics=payload["models"],
        checkpoint_path=model_file,
        dataset_name=dataset_name,
        fusion_file=args.fusion_json,
    )
    print(f"Saved RecBole comparison markdown: {args.output_md}")


if __name__ == "__main__":
    main()
