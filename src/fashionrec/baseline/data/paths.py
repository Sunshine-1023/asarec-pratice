"""Resolved paths for one processed FashionRec dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_DIR = Path("data/processed")


@dataclass(frozen=True, slots=True)
class ProcessedDataPaths:
    """Keep every downstream reader on one explicit processed-data root."""

    root: Path

    @classmethod
    def from_root(cls, data_dir: str | Path | None = None) -> "ProcessedDataPaths":
        return cls(Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR)

    @property
    def hm_dir(self) -> Path:
        return self.root / "hm"

    @property
    def seq_dir(self) -> Path:
        return self.root / "hm_seq"

    @property
    def train_inter(self) -> Path:
        return self.hm_dir / "hm.train.inter"

    @property
    def model_train_inter(self) -> Path:
        return self.hm_dir / "hm.model_train.inter"

    @property
    def valid_inter(self) -> Path:
        return self.hm_dir / "hm.valid.inter"

    @property
    def test_inter(self) -> Path:
        return self.hm_dir / "hm.test.inter"

    @property
    def seq_train_inter(self) -> Path:
        return self.seq_dir / "hm_seq.train.inter"

    @property
    def seq_valid_inter(self) -> Path:
        return self.seq_dir / "hm_seq.valid.inter"

    @property
    def seq_test_inter(self) -> Path:
        return self.seq_dir / "hm_seq.test.inter"

    @property
    def seq_item(self) -> Path:
        return self.seq_dir / "hm_seq.item"
