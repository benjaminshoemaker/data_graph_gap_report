from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff"
    b"\x87\x0f\x0d\x1d\x00\x00\x00\x0cIDAT\x08\xd7c```\x00\x00\x00\x04\x00\x01"
    b"\x0b\xe7\x02\x9a\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PLACEHOLDER_PNG)


def plot_lag_p95_daily(data: Sequence[Mapping], path: Path) -> None:
    _write_placeholder(path)


def plot_key_null_pct_daily(data: Sequence[Mapping], path: Path) -> None:
    _write_placeholder(path)


def plot_orphan_pct_daily(data: Sequence[Mapping], path: Path) -> None:
    _write_placeholder(path)


def plot_dup_key_pct_bar(data: Mapping[str, float], path: Path) -> None:
    _write_placeholder(path)


def plot_theme_demand_monthly(data: Sequence[Mapping], path: Path) -> None:
    _write_placeholder(path)


__all__ = [
    "plot_lag_p95_daily",
    "plot_key_null_pct_daily",
    "plot_orphan_pct_daily",
    "plot_dup_key_pct_bar",
    "plot_theme_demand_monthly",
]
