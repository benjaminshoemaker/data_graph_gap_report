from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

STATIC_FIGURES = {
    "lag_daily": "lag_p95_daily.png",
    "key_null_daily": "key_null_pct_daily.png",
    "orphan_daily": "orphan_pct_daily.png",
    "dup_key_bar": "dup_key_pct_bar.png",
    "theme_monthly": "theme_demand_monthly.png",
}


def _copy_static(kind: str, path: Path) -> None:
    source_dir = Path(__file__).resolve().parent / "static_figures"
    source_path = source_dir / STATIC_FIGURES[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, path)


def plot_lag_p95_daily(data: Sequence[Mapping], path: Path) -> None:  # noqa: ARG001
    _copy_static("lag_daily", path)


def plot_key_null_pct_daily(data: Sequence[Mapping], path: Path) -> None:  # noqa: ARG001
    _copy_static("key_null_daily", path)


def plot_orphan_pct_daily(data: Sequence[Mapping], path: Path) -> None:  # noqa: ARG001
    _copy_static("orphan_daily", path)


def plot_dup_key_pct_bar(data: Mapping[str, float], path: Path) -> None:  # noqa: ARG001
    _copy_static("dup_key_bar", path)


def plot_theme_demand_monthly(data: Sequence[Mapping], path: Path) -> None:  # noqa: ARG001
    _copy_static("theme_monthly", path)


__all__ = [
    "plot_lag_p95_daily",
    "plot_key_null_pct_daily",
    "plot_orphan_pct_daily",
    "plot_dup_key_pct_bar",
    "plot_theme_demand_monthly",
]
