from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duckdb")
pl = pytest.importorskip("polars")

# isort: off
from data_needs_reporter.utils.duck import (  # noqa: E402
    attach_parquet_dir,
    open_db,
    safe_query,
)

# isort: on
from data_needs_reporter.utils.io import write_parquet_atomic  # noqa: E402


def test_attach_parquet_dir_and_query(tmp_path: Path) -> None:
    df = pl.DataFrame({"id": [1, 2, 3]})
    data_dir = tmp_path / "parquet"
    parquet_path = data_dir / "sample.parquet"
    write_parquet_atomic(parquet_path, df)

    conn = open_db(None)
    attach_parquet_dir(conn, "sample", data_dir)

    result = safe_query(conn, "SELECT COUNT(*) AS cnt FROM sample_sample")
    assert result.to_dict(False) == {"cnt": [3]}


def test_safe_query_blocks_forbidden_sql() -> None:
    conn = open_db(None)
    with pytest.raises(ValueError):
        safe_query(conn, "ATTACH 'danger'")
