from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


def _require_polars() -> pl.DataFrame:
    if pl is None:  # pragma: no cover
        raise RuntimeError("polars is required for metrics computation")
    return pl


def key_null_pct(df: "pl.DataFrame", columns: Sequence[str]) -> float:
    polars = _require_polars()
    if not columns or df.height == 0:
        return 0.0
    exprs = [polars.col(col).is_null() for col in columns]
    null_any = polars.max_horizontal(exprs)
    total_nulls = df.select(null_any.alias("is_null")).sum().item()
    return (total_nulls / df.height) * 100 if df.height else 0.0


def fk_metrics(
    fact_df: "pl.DataFrame",
    fk_column: str,
    valid_ids: Iterable[int],
) -> Dict[str, float]:
    polars = _require_polars()
    if fact_df.height == 0:
        return {"fk_success_pct": 100.0, "fk_orphan_pct": 0.0}
    valid = polars.Series("valid", list(valid_ids))
    fk_series = fact_df.select(polars.col(fk_column)).to_series()
    is_null = fk_series.is_null()
    success = fk_series.is_in(valid)
    success_count = success.sum()
    failure = (~success) & (~is_null)
    orphan_count = failure.sum()
    total = fk_series.len()
    success_pct = (success_count / total) * 100 if total else 0.0
    orphan_pct = (orphan_count / total) * 100 if total else 0.0
    return {"fk_success_pct": success_pct, "fk_orphan_pct": orphan_pct}


def dup_key_pct(df: "pl.DataFrame", key_columns: Sequence[str]) -> float:
    polars = _require_polars()
    if not key_columns or df.height == 0:
        return 0.0
    dup_df = (
        df.group_by(list(key_columns))
        .count()
        .filter(polars.col("count") > 1)
        .with_columns((polars.col("count") - 1).alias("extra"))
    )
    duplicate_rows = dup_df.select(polars.col("extra").sum()).item() or 0
    return (duplicate_rows / df.height) * 100 if df.height else 0.0


def p95_ingest_lag_min(
    df: "pl.DataFrame",
    event_col: str,
    loaded_col: str,
) -> float:
    polars = _require_polars()
    if df.height == 0:
        return 0.0
    lag_series = (
        df.with_columns(
            (
                (polars.col(loaded_col) - polars.col(event_col)).dt.total_seconds()
                / 60.0
            ).alias("_lag_min")
        )
        .select(polars.col("_lag_min"))
        .drop_nulls()
        .to_series()
    )
    if lag_series.is_empty():
        return 0.0
    return float(lag_series.quantile(0.95, interpolation="higher"))


def detect_null_spikes(
    df: "pl.DataFrame",
    event_col: str,
    target_column: str,
    window: int = 7,
    threshold: float = 0.08,
) -> List[Dict[str, float]]:
    polars = _require_polars()
    if df.height == 0:
        return []
    daily = (
        df.with_columns(polars.col(event_col).dt.truncate("1d").alias("_day"))
        .group_by("_day")
        .agg(
            [
                polars.count().alias("rows"),
                polars.col(target_column).is_null().sum().alias("nulls"),
            ]
        )
        .sort("_day")
    )
    if daily.height == 0:
        return []
    daily = daily.with_columns(
        (polars.col("nulls") / polars.col("rows")).alias("null_rate")
    )
    days = daily["_day"].to_list()
    rates = daily["null_rate"].to_list()
    spikes: List[Dict[str, float]] = []
    for idx, (day, rate) in enumerate(zip(days, rates)):
        prev_rates = rates[max(0, idx - window) : idx]
        if not prev_rates:
            continue
        median_prev = float(polars.Series(prev_rates).median())
        if rate >= median_prev + threshold:
            spikes.append({"day": day.date().isoformat(), "null_rate": rate})
    return spikes


def evaluate_slos(
    metrics: Mapping[str, float], slos: Mapping[str, float]
) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for metric, threshold in slos.items():
        value = metrics.get(metric)
        if value is None:
            continue
        if (
            metric == "fk_orphan_pct"
            or metric == "dup_keys_pct"
            or metric == "key_null_pct"
        ):
            results[metric] = value <= threshold
        elif metric == "p95_ingest_lag_min":
            results[metric] = value <= threshold
        else:
            results[metric] = value <= threshold
    results["overall_pass"] = all(results.values()) if results else True
    return results


@dataclass
class TableMetricConfig:
    required_columns: Sequence[str]
    duplicate_keys: Sequence[str]
    fk_column: Optional[str] = None
    fk_dimension_ids: Optional[Sequence[int]] = None
    event_column: Optional[str] = None
    loaded_column: Optional[str] = None
    spike_column: Optional[str] = None


def compute_table_metrics(
    df: "pl.DataFrame",
    config: TableMetricConfig,
) -> Dict[str, object]:
    _require_polars()
    metrics: Dict[str, object] = {}
    metrics["key_null_pct"] = key_null_pct(df, config.required_columns)
    metrics["dup_keys_pct"] = dup_key_pct(df, config.duplicate_keys)
    if config.fk_column and config.fk_dimension_ids is not None:
        metrics.update(fk_metrics(df, config.fk_column, config.fk_dimension_ids))
    else:
        metrics["fk_success_pct"] = 100.0
        metrics["fk_orphan_pct"] = 0.0
    if config.event_column and config.loaded_column:
        metrics["p95_ingest_lag_min"] = p95_ingest_lag_min(
            df, config.event_column, config.loaded_column
        )
    else:
        metrics["p95_ingest_lag_min"] = 0.0
    if config.spike_column and config.event_column:
        spikes = detect_null_spikes(df, config.event_column, config.spike_column)
        metrics["null_spike_days"] = spikes
    else:
        metrics["null_spike_days"] = []
    return metrics


__all__ = [
    "key_null_pct",
    "fk_metrics",
    "dup_key_pct",
    "p95_ingest_lag_min",
    "detect_null_spikes",
    "evaluate_slos",
    "TableMetricConfig",
    "compute_table_metrics",
]
