"""RecBole-protocol weighted fusion with grid search."""

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
DEFAULT_OUTPUT_JSON = Path("outputs/evaluation/recbole_fusion_weight_search.json")


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


def _build_traditional_model(
    model_name: str,
    shared: dict[str, Any],
    config_path: Path,
    fit_epochs: int = 1,
):
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
    return model


def _build_sasrec_model(shared: dict[str, Any], model_file: Path):
    config = shared["config"]
    model = get_model("SASRec")(config, shared["dataset"]).to(config["device"])
    checkpoint = torch.load(str(model_file), map_location=config["device"])
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing 'state_dict': {model_file}")
    model.load_state_dict(checkpoint["state_dict"])
    return model


class WeightedFusionModel(torch.nn.Module):
    """Fusion wrapper that combines three full-sort score vectors."""

    def __init__(
        self,
        sasrec_model: torch.nn.Module,
        pop_model: torch.nn.Module,
        itemknn_model: torch.nn.Module,
        weights: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.sasrec_model = sasrec_model
        self.pop_model = pop_model
        self.itemknn_model = itemknn_model
        self.set_weights(weights)

    def set_weights(self, weights: tuple[float, float, float]) -> None:
        self.w_pop, self.w_itemknn, self.w_sasrec = weights

    def to(self, device):  # noqa: ANN001
        self.sasrec_model.to(device)
        self.pop_model.to(device)
        self.itemknn_model.to(device)
        return self

    def eval(self):
        self.sasrec_model.eval()
        self.pop_model.eval()
        self.itemknn_model.eval()
        return self

    def full_sort_predict(self, interaction):
        sasrec_scores = self.sasrec_model.full_sort_predict(interaction)
        pop_scores = self.pop_model.full_sort_predict(interaction)
        itemknn_scores = self.itemknn_model.full_sort_predict(interaction)

        target_numel = sasrec_scores.numel()
        if pop_scores.numel() != target_numel or itemknn_scores.numel() != target_numel:
            raise RuntimeError(
                "Fusion score shape mismatch: "
                f"sasrec={tuple(sasrec_scores.shape)}, "
                f"pop={tuple(pop_scores.shape)}, "
                f"itemknn={tuple(itemknn_scores.shape)}"
            )

        pop_scores = pop_scores.reshape(sasrec_scores.shape)
        itemknn_scores = itemknn_scores.reshape(sasrec_scores.shape)
        fused_scores = (
            self.w_pop * pop_scores
            + self.w_itemknn * itemknn_scores
            + self.w_sasrec * sasrec_scores
        )
        return fused_scores.reshape(-1)


def parse_weight_grid(weight_grid: str | None) -> list[tuple[float, float, float]]:
    def normalize(weights: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(w < 0 for w in weights):
            raise ValueError(f"Weight must be >= 0, got {weights}")
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"Weight sum must be > 0, got {weights}")
        return (weights[0] / total, weights[1] / total, weights[2] / total)

    if not weight_grid:
        defaults = [
            (0.10, 0.20, 0.70),
            (0.10, 0.30, 0.60),
            (0.10, 0.40, 0.50),
            (0.20, 0.20, 0.60),
            (0.20, 0.30, 0.50),
            (0.20, 0.40, 0.40),
            (0.30, 0.20, 0.50),
            (0.30, 0.30, 0.40),
        ]
        return [normalize(w) for w in defaults]
    combos: list[tuple[float, float, float]] = []
    for part in weight_grid.split(";"):
        p, i, s = (x.strip() for x in part.split(","))
        combos.append(normalize((float(p), float(i), float(s))))
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="RecBole-protocol fusion grid search")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-file", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--fit-epochs", type=int, default=1)
    parser.add_argument(
        "--weight-grid",
        type=str,
        default=None,
        help="Semicolon-separated tuples: p,i,s;p,i,s",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    model_file = args.model_file or _latest_checkpoint(args.checkpoint_dir)
    if not model_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")

    shared = build_shared_context(args.config)
    sas_config = shared["config"]
    sas_model = _build_sasrec_model(shared, model_file)
    pop_model = _build_traditional_model(
        "Pop", shared, args.config, fit_epochs=args.fit_epochs
    )
    itemknn_model = _build_traditional_model(
        "ItemKNN", shared, args.config, fit_epochs=args.fit_epochs
    )

    fusion_model = WeightedFusionModel(
        sasrec_model=sas_model,
        pop_model=pop_model,
        itemknn_model=itemknn_model,
        weights=(0.2, 0.3, 0.5),
    ).to(sas_config["device"]).eval()

    fusion_trainer = get_trainer(sas_config["MODEL_TYPE"], sas_config["model"])(
        sas_config, fusion_model
    )

    grid = parse_weight_grid(args.weight_grid)
    rows = []
    best_row = None
    for w_pop, w_itemknn, w_sasrec in grid:
        print(
            f"Evaluating normalized weights: popular={w_pop:.4f}, "
            f"itemknn={w_itemknn:.4f}, sasrec={w_sasrec:.4f}"
        )
        fusion_model.set_weights((w_pop, w_itemknn, w_sasrec))
        valid_metrics = _to_float_metrics(
            fusion_trainer.evaluate(
                shared["valid_data"], load_best_model=False, show_progress=False
            )
        )
        row = {
            "weights": {
                "popular": w_pop,
                "itemknn": w_itemknn,
                "sasrec": w_sasrec,
            },
            "valid_metrics": valid_metrics,
        }
        rows.append(row)
        if best_row is None or row["valid_metrics"].get("map@12", 0.0) > best_row[
            "valid_metrics"
        ].get("map@12", 0.0):
            best_row = row

    assert best_row is not None
    w = best_row["weights"]
    fusion_model.set_weights((w["popular"], w["itemknn"], w["sasrec"]))
    test_metrics = _to_float_metrics(
        fusion_trainer.evaluate(shared["test_data"], load_best_model=False, show_progress=False)
    )

    payload = {
        "protocol": "recbole_full_ranking_hm_seq",
        "dataset": str(sas_config["dataset"]),
        "checkpoint": str(model_file.resolve()),
        "grid": rows,
        "best": {
            "weights": best_row["weights"],
            "valid_metrics": best_row["valid_metrics"],
            "test_metrics": test_metrics,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved RecBole fusion grid search: {args.output_json}")
    print(json.dumps(payload["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
