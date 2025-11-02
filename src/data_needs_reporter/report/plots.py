from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_lag_p95_daily(data: Sequence[Mapping], path: Path) -> None:
    days = [entry["day"] for entry in data]
    values = [entry["p95_lag_min"] for entry in data]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(days, values, marker="o")
    ax.set_title("Daily p95 Ingest Lag")
    ax.set_xlabel("Day")
    ax.set_ylabel("Lag (min)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_key_null_pct_daily(data: Sequence[Mapping], path: Path) -> None:
    days = [entry["day"] for entry in data]
    values = [entry["key_null_pct"] for entry in data]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(days, values, color="#d95f02")
    ax.set_title("Daily Key Null %")
    ax.set_xlabel("Day")
    ax.set_ylabel("Null %")
    fig.autofmt_xdate()
    ax.set_ylim(0, max(values + [1]) * 1.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_orphan_pct_daily(data: Sequence[Mapping], path: Path) -> None:
    days = [entry["day"] for entry in data]
    values = [entry["orphan_pct"] for entry in data]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(days, values, marker="s", linestyle="--", color="#1b9e77")
    ax.set_title("Daily Orphan %")
    ax.set_xlabel("Day")
    ax.set_ylabel("Orphan %")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dup_key_pct_bar(data: Mapping[str, float], path: Path) -> None:
    tables = list(data.keys())
    values = [data[table] for table in tables]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(tables, values, color="#7570b3")
    ax.set_title("Duplicate Key % by Table")
    ax.set_xlabel("Dup %")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_theme_demand_monthly(data: Sequence[Mapping], path: Path) -> None:
    months = [entry["month"] for entry in data]
    themes = sorted({key for entry in data for key in entry if key != "month"})
    fig, ax = plt.subplots(figsize=(6, 3))
    for theme in themes:
        theme_values = [entry.get(theme, 0) for entry in data]
        ax.plot(months, theme_values, marker="o", label=theme)
    ax.set_title("Theme Demand (Monthly)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Demand Weight")
    ax.legend(loc="upper right", fontsize="small")
    fig.autofmt_xdate()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "plot_lag_p95_daily",
    "plot_key_null_pct_daily",
    "plot_orphan_pct_daily",
    "plot_dup_key_pct_bar",
    "plot_theme_demand_monthly",
]
