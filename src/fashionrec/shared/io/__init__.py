"""Neutral tabular I/O helpers shared by both applications."""

from fashionrec.shared.io.tabular import read_csv_rows, read_parquet_frame, write_csv_rows, write_parquet_frame

__all__ = ["read_csv_rows", "read_parquet_frame", "write_csv_rows", "write_parquet_frame"]
