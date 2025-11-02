from __future__ import annotations

from pathlib import Path

import pytest

pl = pytest.importorskip("polars")

from data_needs_reporter.utils.io import (
    read_json,
    read_parquet,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)


def test_write_json_atomic_creates_directories_and_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "config.json"
    payload = {"key": "value", "nested": {"answer": 42}}

    write_json_atomic(target, payload)
    assert target.is_file()
    assert read_json(target) == payload

    updated_payload = {"key": "updated"}
    write_json_atomic(target, updated_payload)
    assert read_json(target) == updated_payload
    assert not any(target.parent.glob("*.tmp"))


def test_write_parquet_atomic_round_trip(tmp_path: Path) -> None:
    df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    target = tmp_path / "data" / "sample.parquet"

    write_parquet_atomic(target, df)
    assert target.is_file()

    round_trip = read_parquet(target)
    assert round_trip.sort("id").to_dict(False) == df.sort("id").to_dict(False)


def test_write_csv_atomic_round_trip(tmp_path: Path) -> None:
    df = pl.DataFrame({"name": ["alice", "bob"], "score": [10, 20]})
    target = tmp_path / "exports" / "sample.csv"

    write_csv_atomic(target, df)
    assert target.is_file()

    round_trip = pl.read_csv(target)
    assert round_trip.sort("name").to_dict(False) == df.sort("name").to_dict(False)
