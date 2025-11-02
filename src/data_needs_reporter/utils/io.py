from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

try:  # pragma: no cover - optional dependency handling
    import polars as _pl
except ImportError:  # pragma: no cover
    _pl = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import polars as pl  # noqa: F401

    PolarsDataFrame = pl.DataFrame
else:
    PolarsDataFrame = Any
    pl = _pl  # type: ignore[assignment]

PathLike = Union[str, Path]


def write_json_atomic(path: PathLike, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with _tempfile(target) as tmp_path:
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, indent=2, sort_keys=True)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, target)


def read_json(path: PathLike) -> Any:
    with open(Path(path), "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_parquet_atomic(path: PathLike, dataframe: PolarsDataFrame) -> None:
    _ensure_polars()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with _tempfile(target, suffix=".parquet") as tmp_path:
        dataframe.write_parquet(tmp_path)
        _fsync_path(tmp_path)
        os.replace(tmp_path, target)


def read_parquet(path: PathLike):
    _ensure_polars()
    return pl.read_parquet(path)


def write_csv_atomic(path: PathLike, dataframe: PolarsDataFrame) -> None:
    _ensure_polars()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with _tempfile(target, suffix=".csv") as tmp_path:
        dataframe.write_csv(tmp_path)
        _fsync_path(tmp_path)
        os.replace(tmp_path, target)


class _AtomicTempFile:
    def __init__(self, temp_path: Path):
        self.temp_path = temp_path

    def __enter__(self) -> Path:
        return self.temp_path

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self.temp_path.exists():
            self.temp_path.unlink(missing_ok=True)


def _tempfile(target: Path, suffix: str = "") -> _AtomicTempFile:
    fd, tmp = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.tmp-",
        suffix=suffix,
    )
    os.close(fd)
    return _AtomicTempFile(Path(tmp))


def _fsync_path(temp_path: Path) -> None:
    with open(temp_path, "rb") as fp:
        fp.flush()
        os.fsync(fp.fileno())


def _ensure_polars() -> None:
    if pl is None:
        raise RuntimeError("polars is required for parquet or csv helpers.")
