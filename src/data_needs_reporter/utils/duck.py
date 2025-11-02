from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import polars as pl

_IDENTIFIER_CLEAN_RE = re.compile(r"[^A-Za-z0-9_]")


def open_db(db_path_or_none: Optional[Path]) -> duckdb.DuckDBPyConnection:
    if db_path_or_none is None:
        return duckdb.connect(database=":memory:")
    Path(db_path_or_none).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=str(db_path_or_none))


def attach_parquet_dir(
    conn: duckdb.DuckDBPyConnection, name: str, dir_path: Path
) -> None:
    directory = Path(dir_path)
    if not directory.exists():
        raise FileNotFoundError(f"Parquet directory not found: {dir_path}")

    parquet_files = sorted(directory.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dir_path}")

    for file_path in parquet_files:
        view_name = _make_identifier(f"{name}_{file_path.stem}")
        file_literal = str(file_path).replace("'", "''")
        conn.execute(
            f"CREATE VIEW IF NOT EXISTS {view_name} AS SELECT * FROM read_parquet('{file_literal}')"
        )


def safe_query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: Optional[Iterable] = None,
) -> pl.DataFrame:
    _guard_sql(sql)
    result = conn.execute(sql, params or ())
    return result.pl()


def _guard_sql(sql: str) -> None:
    lowered = sql.lower()
    if ";" in sql:
        raise ValueError("Semicolons are not permitted in safe_query.")
    forbidden = ("copy", "attach", "install")
    if any(keyword in lowered for keyword in forbidden):
        raise ValueError("Potentially unsafe SQL detected.")


def _make_identifier(raw: str) -> str:
    if not raw:
        raise ValueError("Identifier cannot be empty.")
    sanitized = _IDENTIFIER_CLEAN_RE.sub("_", raw)
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized
