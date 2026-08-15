"""Preserve a small RecBole-valid checkpoint shortlist without importing RecBole."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def discover_checkpoint_candidates(checkpoint_dir: str | Path) -> list[Path]:
    """Return deterministic coarse-shortlist candidates, falling back to legacy checkpoints."""
    root = Path(checkpoint_dir)
    shortlist = sorted(root.rglob("candidate_epoch_*.pth")) if root.exists() else []
    if shortlist:
        return shortlist
    legacy = sorted(root.glob("*.pth")) if root.exists() else []
    if not legacy:
        raise FileNotFoundError(f"No checkpoint candidates found in {root}")
    return legacy


def install_improving_checkpoint_snapshots(
    trainer: Any,
    output_dir: str | Path,
    *,
    max_candidates: int = 5,
) -> list[Path]:
    """Snapshot each RecBole-valid improvement and retain the latest Top-N improvements.

    RecBole's trainer writes its best checkpoint to one stable filename. Its private
    ``_save_checkpoint`` hook is called when the coarse validation score improves;
    wrapping that hook preserves several improving states for later user-week MAP@K
    selection while leaving RecBole's normal save/early-stop behavior unchanged.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if not hasattr(trainer, "_save_checkpoint") or not hasattr(trainer, "saved_model_file"):
        raise TypeError("trainer must expose _save_checkpoint and saved_model_file")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    original_save = trainer._save_checkpoint
    snapshots: list[Path] = []

    def save_and_snapshot(*args: Any, **kwargs: Any) -> Any:
        result = original_save(*args, **kwargs)
        epoch_raw = kwargs.get("epoch", args[0] if args else len(snapshots))
        try:
            epoch = int(epoch_raw)
        except (TypeError, ValueError):
            epoch = len(snapshots)
        source = Path(trainer.saved_model_file)
        if not source.exists():
            raise FileNotFoundError(f"RecBole did not write expected checkpoint: {source}")
        snapshot = destination / f"candidate_epoch_{epoch:04d}.pth"
        shutil.copy2(source, snapshot)
        if snapshot not in snapshots:
            snapshots.append(snapshot)
        while len(snapshots) > max_candidates:
            removed = snapshots.pop(0)
            if removed.exists():
                removed.unlink()
        return result

    trainer._save_checkpoint = save_and_snapshot
    return snapshots
