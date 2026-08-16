"""Preserve a true Top-N RecBole-validation checkpoint shortlist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def discover_checkpoint_candidates(checkpoint_dir: str | Path) -> list[Path]:
    """Return deterministic coarse-shortlist candidates, falling back to legacy checkpoints."""
    root = Path(checkpoint_dir)
    shortlist = sorted(root.rglob("candidate_eval_*.pth")) if root.exists() else []
    if not shortlist and root.exists():
        shortlist = sorted(root.rglob("candidate_epoch_*.pth"))  # 兼容旧 shortlist
    if shortlist:
        return shortlist
    legacy = sorted(root.glob("*.pth")) if root.exists() else []
    if not legacy:
        raise FileNotFoundError(f"No checkpoint candidates found in {root}")
    return legacy


def install_validation_checkpoint_shortlist(
    trainer: Any,
    output_dir: str | Path,
    *,
    max_candidates: int = 5,
) -> list[Path]:
    """Snapshot every validation epoch and keep the metric-ranked Top-N states."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    destination = Path(output_dir)
    if destination.exists():
        existing = sorted(path.name for path in destination.iterdir())
        if existing:
            raise FileExistsError(
                f"Checkpoint shortlist directory must be empty: {destination}. "
                "Use a new run ID/checkpoint directory, or skip training to reuse the existing shortlist. "
                f"Existing entries: {existing}"
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
    required = [name for name in ("_valid_epoch", "model", "config") if not hasattr(trainer, name)]
    if required:
        raise TypeError(f"trainer missing validation checkpoint attributes: {required}")

    manifest_path = destination / "shortlist_manifest.json"
    original_valid_epoch = trainer._valid_epoch
    snapshots: list[Path] = []
    records: list[dict[str, Any]] = []
    validation_index = 0
    try:
        valid_metric_bigger = bool(trainer.config["valid_metric_bigger"])
    except (KeyError, TypeError):
        valid_metric_bigger = True

    def validate_and_snapshot(*args: Any, **kwargs: Any) -> Any:
        nonlocal validation_index
        result = original_valid_epoch(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("trainer._valid_epoch must return (valid_score, valid_result)")
        valid_score = float(result[0])
        validation_index += 1
        snapshot = destination / f"candidate_eval_{validation_index:04d}.pth"

        import torch  # 训练依赖；保持模块导入与 CLI help 不依赖 RecBole

        state = {
            "config": trainer.config,
            "epoch": validation_index - 1,
            "cur_step": getattr(trainer, "cur_step", 0),
            "best_valid_score": valid_score,
            "state_dict": trainer.model.state_dict(),
            "other_parameter": (
                trainer.model.other_parameter() if hasattr(trainer.model, "other_parameter") else None
            ),
        }
        if hasattr(trainer, "optimizer"):
            state["optimizer"] = trainer.optimizer.state_dict()
        torch.save(state, snapshot)
        records.append(
            {
                "validation_index": validation_index,
                "coarse_valid_score": valid_score,
                "checkpoint": str(snapshot.resolve()),
            }
        )

        ranked = sorted(
            records,
            key=lambda row: (
                -row["coarse_valid_score"] if valid_metric_bigger else row["coarse_valid_score"],
                row["validation_index"],
            ),
        )
        kept = ranked[:max_candidates]
        kept_paths = {Path(row["checkpoint"]) for row in kept}
        for row in records:
            path = Path(row["checkpoint"])
            if path not in kept_paths and path.exists():
                path.unlink()
        records[:] = kept
        snapshots[:] = [Path(row["checkpoint"]) for row in kept]
        manifest_path.write_text(
            json.dumps(
                {
                    "valid_metric_bigger": valid_metric_bigger,
                    "max_candidates": max_candidates,
                    "candidates": kept,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    trainer._valid_epoch = validate_and_snapshot
    return snapshots
