from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    threshold_pct: float = 8.0,
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
        ((polars.col("nulls") / polars.col("rows")) * 100.0).alias("null_pct")
    )
    days = daily["_day"].to_list()
    rates = daily["null_pct"].to_list()
    spikes: List[Dict[str, float]] = []
    for idx, (day, rate) in enumerate(zip(days, rates)):
        prev_rates = rates[max(0, idx - window) : idx]
        if not prev_rates:
            continue
        median_prev = float(polars.Series(prev_rates).median())
        if rate >= median_prev + threshold_pct:
            spikes.append(
                {
                    "ds": day.date().isoformat(),
                    "column": target_column,
                    "lift_pct": round(rate - median_prev, 3),
                }
            )
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
    fk_dimension_table: Optional[str] = None
    fk_dimension_key: Optional[str] = None
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


NEOBANK_TABLE_SPECS: List[Tuple[str, TableMetricConfig]] = [
    (
        "dim_customer",
        TableMetricConfig(
            required_columns=["customer_id"],
            duplicate_keys=["customer_id"],
        ),
    ),
    (
        "dim_account",
        TableMetricConfig(
            required_columns=["account_id", "customer_id"],
            duplicate_keys=["account_id"],
            fk_column="customer_id",
            fk_dimension_table="dim_customer",
        ),
    ),
    (
        "dim_card",
        TableMetricConfig(
            required_columns=["card_id", "account_id"],
            duplicate_keys=["card_id"],
            fk_column="account_id",
            fk_dimension_table="dim_account",
        ),
    ),
    (
        "dim_merchant",
        TableMetricConfig(
            required_columns=["merchant_id", "mcc"],
            duplicate_keys=["merchant_id"],
        ),
    ),
    (
        "dim_plan",
        TableMetricConfig(
            required_columns=["plan_id", "name"],
            duplicate_keys=["plan_id"],
        ),
    ),
    (
        "fact_card_transaction",
        TableMetricConfig(
            required_columns=["merchant_id"],
            duplicate_keys=["txn_id"],
            fk_column="card_id",
            fk_dimension_table="dim_card",
            fk_dimension_key="card_id",
            event_column="event_time",
            loaded_column="loaded_at",
            spike_column="merchant_id",
        ),
    ),
    (
        "fact_subscription_invoice",
        TableMetricConfig(
            required_columns=["plan_id"],
            duplicate_keys=["invoice_id"],
            fk_column="plan_id",
            fk_dimension_table="dim_plan",
            fk_dimension_key="plan_id",
            event_column="paid_at",
            loaded_column="loaded_at",
        ),
    ),
]

MARKETPLACE_TABLE_SPECS: List[Tuple[str, TableMetricConfig]] = [
    (
        "dim_buyer",
        TableMetricConfig(
            required_columns=["buyer_id"],
            duplicate_keys=["buyer_id"],
        ),
    ),
    (
        "dim_seller",
        TableMetricConfig(
            required_columns=["seller_id"],
            duplicate_keys=["seller_id"],
        ),
    ),
    (
        "dim_listing",
        TableMetricConfig(
            required_columns=["listing_id", "seller_id"],
            duplicate_keys=["listing_id"],
            fk_column="seller_id",
            fk_dimension_table="dim_seller",
            fk_dimension_key="seller_id",
        ),
    ),
    (
        "fact_order",
        TableMetricConfig(
            required_columns=["buyer_id"],
            duplicate_keys=["order_id"],
            fk_column="buyer_id",
            fk_dimension_table="dim_buyer",
            fk_dimension_key="buyer_id",
            event_column="order_time",
            loaded_column="loaded_at",
        ),
    ),
    (
        "fact_order_item",
        TableMetricConfig(
            required_columns=["listing_id"],
            duplicate_keys=["order_id", "line_id"],
            fk_column="listing_id",
            fk_dimension_table="dim_listing",
            fk_dimension_key="listing_id",
            event_column="loaded_at",
            loaded_column="loaded_at",
        ),
    ),
    (
        "fact_payment",
        TableMetricConfig(
            required_columns=["order_id"],
            duplicate_keys=["order_id"],
            fk_column="order_id",
            fk_dimension_table="fact_order",
            fk_dimension_key="order_id",
            event_column="captured_at",
            loaded_column="loaded_at",
        ),
    ),
    (
        "snapshot_listing_daily",
        TableMetricConfig(
            required_columns=["listing_id"],
            duplicate_keys=["ds", "listing_id"],
        ),
    ),
]

TABLE_METRIC_SPECS: Dict[str, List[Tuple[str, TableMetricConfig]]] = {
    "neobank": NEOBANK_TABLE_SPECS,
    "marketplace": MARKETPLACE_TABLE_SPECS,
}

DEFAULT_METRIC_ROW = {
    "key_null_pct": 0.0,
    "fk_success_pct": 100.0,
    "fk_orphan_pct": 0.0,
    "dup_keys_pct": 0.0,
    "p95_ingest_lag_min": 0.0,
    "null_spike_days": [],
}


def compute_data_health(
    archetype: str, warehouse_path: Path
) -> List[Dict[str, object]]:
    polars = _require_polars()
    archetype_key = archetype.lower()
    specs = TABLE_METRIC_SPECS.get(archetype_key, [])
    warehouse_path = Path(warehouse_path)
    if not specs:
        return []

    table_dfs: Dict[str, Optional["pl.DataFrame"]] = {}
    for table_name, _ in specs:
        table_path = warehouse_path / f"{table_name}.parquet"
        if table_path.exists():
            try:
                table_dfs[table_name] = polars.read_parquet(table_path)
            except Exception:  # pragma: no cover - unreadable file
                table_dfs[table_name] = None
        else:
            table_dfs[table_name] = None

    results: List[Dict[str, object]] = []
    for table_name, spec in specs:
        df = table_dfs.get(table_name)
        if df is None or df.height == 0 or any(
            col not in df.columns for col in spec.required_columns
        ):
            metrics = dict(DEFAULT_METRIC_ROW)
            metrics.update({"table": table_name, "row_count": int(df.height) if df else 0})
            results.append(metrics)
            continue

        working_config = spec
        if spec.fk_column and spec.fk_dimension_ids is None:
            fk_ids: Sequence[int] = []
            fk_table = spec.fk_dimension_table
            fk_key = spec.fk_dimension_key or spec.fk_column
            fk_df = table_dfs.get(fk_table) if fk_table else None
            if fk_df is not None and fk_key in fk_df.columns:
                fk_ids = fk_df.select(fk_key).to_series().to_list()  # type: ignore[arg-type]
            working_config = replace(spec, fk_dimension_ids=fk_ids)

        metrics = compute_table_metrics(df, working_config)
        metrics["table"] = table_name
        metrics["row_count"] = df.height
        results.append(metrics)
    return results


__all__ = [
    "key_null_pct",
    "fk_metrics",
    "dup_key_pct",
    "p95_ingest_lag_min",
    "detect_null_spikes",
    "evaluate_slos",
    "TableMetricConfig",
    "compute_table_metrics",
    "compute_data_health",
]
