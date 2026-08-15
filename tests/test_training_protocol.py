"""Validation-only checkpoint production and user-week selection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_sasrec import fit_model_without_test_evaluation
from src.training.checkpoint_selection import (
    load_valid_user_week_targets,
    select_checkpoint_by_score,
    user_week_map_at_k,
)
from src.training.checkpoints import install_improving_checkpoint_snapshots


class _FitOnlyTrainer:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.evaluate_calls = 0

    def fit(self, train_data, valid_data, **kwargs):
        self.fit_calls += 1
        return 0.2, {"MAP@12": 0.2}

    def evaluate(self, *args, **kwargs):
        self.evaluate_calls += 1
        raise AssertionError("test evaluation must not run during training")


def test_training_helper_never_invokes_test_evaluation() -> None:
    trainer = _FitOnlyTrainer()
    result = fit_model_without_test_evaluation(trainer, object(), object(), show_progress=False)
    assert result == (0.2, {"MAP@12": 0.2})
    assert trainer.fit_calls == 1
    assert trainer.evaluate_calls == 0


def test_improving_checkpoint_hook_preserves_bounded_shortlist(tmp_path: Path) -> None:
    class FakeTrainer:
        saved_model_file = tmp_path / "recbole_best.pth"

        def _save_checkpoint(self, epoch: int, **_kwargs) -> None:
            self.saved_model_file.write_bytes(f"epoch-{epoch}".encode())

    trainer = FakeTrainer()
    snapshots = install_improving_checkpoint_snapshots(trainer, tmp_path / "shortlist", max_candidates=2)
    for epoch in (1, 2, 3):
        trainer._save_checkpoint(epoch)

    assert [path.name for path in snapshots] == ["candidate_epoch_0002.pth", "candidate_epoch_0003.pth"]
    assert not (tmp_path / "shortlist" / "candidate_epoch_0001.pth").exists()


def test_checkpoint_selection_uses_complete_valid_user_week_map(tmp_path: Path) -> None:
    valid_path = tmp_path / "hm.valid.inter"
    pd.DataFrame(
        [
            ("u1", "1", 1),
            ("u1", "2", 2),
            ("u2", "3", 3),
        ],
        columns=["user_id:token", "item_id:token", "timestamp:float"],
    ).to_csv(valid_path, sep="\t", index=False)
    targets = load_valid_user_week_targets(valid_path)
    assert user_week_map_at_k(
        targets,
        {"u1": ["1", "2"], "u2": []},
        k=12,
    ) == 0.5

    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    report = tmp_path / "selection.json"
    selected = tmp_path / "selected.pth"

    winner = select_checkpoint_by_score(
        [first, second],
        lambda path: {"first.pth": 0.1, "second.pth": 0.3}[path.name],
        output_json=report,
        selected_model_path=selected,
        k=12,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert winner == selected
    assert selected.read_bytes() == b"second"
    assert payload["selection_split"] == "valid"
    assert payload["selection_unit"] == "user_week"
    assert payload["metric"] == "MAP@12"
    assert set(payload) == {
        "selection_split",
        "selection_unit",
        "metric",
        "selected_source_checkpoint",
        "selected_model_path",
        "scores",
    }
