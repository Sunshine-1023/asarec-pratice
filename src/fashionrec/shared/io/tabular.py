"""Small, schema-agnostic CSV and Parquet primitives."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def write_csv_rows(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    fieldnames: Sequence[str],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"CSV file not found: {source}")
    with source.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_parquet_frame(frame: pd.DataFrame, output_path: str | Path, **kwargs: Any) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, **kwargs)
    return path


def read_parquet_frame(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Parquet file not found: {source}")
    return pd.read_parquet(source, **kwargs)
