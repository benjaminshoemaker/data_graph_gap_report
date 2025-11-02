from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from data_needs_reporter.config import AppConfig
from data_needs_reporter.generate.warehouse import (
    WAREHOUSE_SCHEMAS,
    _ensure_polars,
    _write_table,
    generate_marketplace_dims,
    generate_marketplace_facts,
    generate_neobank_dims,
    generate_neobank_facts,
)


def inject_key_nulls(df, columns: Sequence[str], rate_pct: float, rng: random.Random):
    polars = _ensure_polars()
    if df.height == 0 or rate_pct <= 0:
        return df

    target = max(1, int(df.height * rate_pct / 100))
    target = min(target, df.height)
    indices = set(rng.sample(range(df.height), target))
    mask = polars.Series("_null_mask", [idx in indices for idx in range(df.height)])
    df = df.with_columns(mask)
    updates = [
        polars.when(polars.col("_null_mask"))
        .then(polars.lit(None))
        .otherwise(polars.col(column))
        .alias(column)
        for column in columns
    ]
    df = df.with_columns(updates)
    return df.drop("_null_mask")


def inject_fk_failures(
    df,
    fk_column: str,
    valid_ids: Iterable[int],
    rate_pct: float,
    missing_ratio: float,
    rng: random.Random,
):
    polars = _ensure_polars()
    if df.height == 0 or rate_pct <= 0:
        return df

    valid_ids = list(valid_ids)
    if not valid_ids:
        return df

    total = max(1, int(df.height * rate_pct / 100))
    total = min(total, df.height)
    indices = rng.sample(range(df.height), total)
    missing_count = int(total * missing_ratio)
    missing_indices = set(indices[:missing_count])
    invalid_indices = set(indices[missing_count:])
    max_id = max(valid_ids)

    df = df.with_row_count("_row_idx")
    if missing_indices:
        df = df.with_columns(
            polars.when(polars.col("_row_idx").is_in(polars.Series(list(missing_indices))))
            .then(polars.lit(None))
            .otherwise(polars.col(fk_column))
            .alias(fk_column)
        )
    if invalid_indices:
        df = df.with_columns(
            polars.when(polars.col("_row_idx").is_in(polars.Series(list(invalid_indices))))
            .then(polars.lit(max_id + rng.randint(1000, 5000)))
            .otherwise(polars.col(fk_column))
            .alias(fk_column)
        )
    return df.drop("_row_idx")


def inject_duplicates(df, business_key: str, rate_pct: float, rng: random.Random):
    polars = _ensure_polars()
    if df.height == 0 or rate_pct <= 0:
        return df

    duplicate_count = max(1, int(df.height * rate_pct / 100))
    duplicate_count = min(duplicate_count, df.height)
    indices = rng.sample(range(df.height), duplicate_count)
    duplicate_df = (
        df.with_row_count("_row_idx")
        .filter(polars.col("_row_idx").is_in(polars.Series(indices)))
        .drop("_row_idx")
    )
    # Keep business key identical to create duplicates.
    return df.vstack(duplicate_df)


def apply_ingest_lag(df, event_col: str, loaded_at_col: str, rng: random.Random):
    polars = _ensure_polars()
    if df.height == 0:
        return df
    events = df[event_col].to_list()
    lags: list[int] = []
    for _ in events:
        if rng.random() < 0.95:
            lag = int(rng.uniform(30, 180))
        else:
            lag = int(rng.uniform(240, 720))
        lags.append(lag)
    loaded_values = [
        event + timedelta(minutes=lag) if event is not None else None
        for event, lag in zip(events, lags)
    ]
    return df.with_columns(polars.Series(loaded_at_col, loaded_values))


def inject_null_spikes(
    df,
    target_column: str,
    event_col: str,
    schedule: Sequence[Mapping[str, Any]],
    rng: random.Random,
):
    polars = _ensure_polars()
    if df.height == 0 or not schedule:
        return df

    df = df.with_columns(polars.col(event_col).dt.truncate("1d").alias("_event_day"))
    df = df.with_row_count("_row_idx")
    for entry in schedule:
        day = entry.get("day")
        rate_pct = float(entry.get("rate_pct", 0))
        if day is None or rate_pct <= 0:
            continue
        candidate_indices = (
            df.filter(polars.col("_event_day") == polars.lit(day))
            ["_row_idx"]
            .to_list()
        )
        if not candidate_indices:
            continue
        target = max(1, int(len(candidate_indices) * rate_pct / 100))
        target = min(target, len(candidate_indices))
        chosen = set(rng.sample(candidate_indices, target))
        df = df.with_columns(
            polars.when(polars.col("_row_idx").is_in(polars.Series(list(chosen))))
            .then(polars.lit(None))
            .otherwise(polars.col(target_column))
            .alias(target_column)
        )
    return df.drop("_row_idx", "_event_day")


def inject_schema_gap(
    df,
    event_col: str,
    gap_start: datetime,
    gap_end: datetime,
):
    polars = _ensure_polars()
    if df.height == 0:
        return df
    return df.filter(
        (polars.col(event_col) < polars.lit(gap_start))
        | (polars.col(event_col) >= polars.lit(gap_end))
    )


def apply_typical_neobank_defects(cfg: AppConfig, out_dir: Path, seed: int | None = None) -> Dict[str, Any]:
    polars = _ensure_polars()
    rng = random.Random((seed if seed is not None else cfg.warehouse.seed) + 333)

    out_path = Path(out_dir)
    txn_df = polars.read_parquet(out_path / "fact_card_transaction.parquet")
    invoice_df = polars.read_parquet(out_path / "fact_subscription_invoice.parquet")
    card_df = polars.read_parquet(out_path / "dim_card.parquet")

    txn_df = inject_key_nulls(txn_df, ["merchant_id"], 2.0, rng)
    txn_df = inject_fk_failures(
        txn_df,
        fk_column="card_id",
        valid_ids=card_df["card_id"],
        rate_pct=5.0,
        missing_ratio=0.7,
        rng=rng,
    )
    txn_df = inject_duplicates(txn_df, "txn_id", 0.7, rng)
    txn_df = apply_ingest_lag(txn_df, "event_time", "loaded_at", rng)

    # Null spikes twice a quarter with +10 pp absolute increase
    schedule = []
    base_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for offset in (60, 130):
        schedule.append(
            {
                "day": (base_date + timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0),
                "rate_pct": 12.0,
            }
        )
    txn_df = inject_null_spikes(txn_df, "merchant_id", "event_time", schedule, rng)

    # Schema gap: remove small window mid-year
    gap_start = datetime(2023, 7, 15, tzinfo=timezone.utc)
    gap_end = gap_start + timedelta(hours=12)
    txn_df = inject_schema_gap(txn_df, "event_time", gap_start, gap_end)

    _write_table("neobank", "fact_card_transaction", txn_df.to_dicts(), out_path, polars)

    invoice_df = inject_key_nulls(invoice_df, ["plan_id"], 1.8, rng)
    invoice_df = inject_duplicates(invoice_df, "invoice_id", 0.4, rng)
    invoice_df = apply_ingest_lag(invoice_df, "paid_at", "loaded_at", rng)
    _write_table("neobank", "fact_subscription_invoice", invoice_df.to_dicts(), out_path, polars)

    metrics = _measure_neobank_defects(txn_df, invoice_df, card_df)
    return metrics


def _measure_neobank_defects(txn_df, invoice_df, card_df) -> Dict[str, Any]:
    polars = _ensure_polars()
    total_txn = txn_df.height or 1
    key_null_rate = txn_df.filter(polars.col("merchant_id").is_null()).height / total_txn
    valid_card_ids = set(card_df["card_id"].to_list())
    fk_fail_rate = (
        txn_df.filter(
            polars.col("card_id").is_null()
            | ~polars.col("card_id").is_in(polars.Series(list(valid_card_ids)))
        ).height
        / total_txn
    )
    lag_minutes = [
        ((loaded - event).total_seconds() / 60)
        for loaded, event in zip(txn_df["loaded_at"], txn_df["event_time"])
        if loaded and event
    ]
    p95_lag = _percentile(lag_minutes, 0.95)
    spike_days = (
        txn_df.with_columns(polars.col("event_time").dt.truncate("1d").alias("event_day"))
        .groupby("event_day")
        .agg(
            [
                polars.count().alias("rows"),
                polars.col("merchant_id").is_null().sum().alias("nulls"),
            ]
        )
        .with_columns((polars.col("nulls") / polars.col("rows")).alias("null_rate"))
        .filter(polars.col("null_rate") > 0.1)
        .to_dicts()
    )
    subscriber_attach = (
        invoice_df["customer_id"].n_unique() / invoice_df.height if invoice_df.height else 0
    )
    return {
        "merchant_key_null_rate": key_null_rate,
        "card_fk_failure_rate": fk_fail_rate,
        "p95_ingest_lag_min": p95_lag,
        "spike_days": spike_days,
        "subscriber_attach_rate": subscriber_attach,
    }


def apply_typical_marketplace_defects(cfg: AppConfig, out_dir: Path, seed: int | None = None) -> Dict[str, Any]:
    polars = _ensure_polars()
    rng = random.Random((seed if seed is not None else cfg.warehouse.seed) + 555)
    out_path = Path(out_dir)

    orders_df = polars.read_parquet(out_path / "fact_order.parquet")
    payments_df = polars.read_parquet(out_path / "fact_payment.parquet")

    orders_df = inject_key_nulls(orders_df, ["buyer_id"], 1.5, rng)
    orders_df = inject_duplicates(orders_df, "order_id", 0.5, rng)
    payments_df = apply_ingest_lag(payments_df, "captured_at", "loaded_at", rng)

    _write_table("marketplace", "fact_order", orders_df.to_dicts(), out_path, polars)
    _write_table("marketplace", "fact_payment", payments_df.to_dicts(), out_path, polars)

    return {
        "order_key_null_rate": (
            orders_df.filter(polars.col("buyer_id").is_null()).height / (orders_df.height or 1)
        ),
    }


def run_typical_generation(cfg: AppConfig, archetype: str, out_dir: Path) -> Dict[str, Any]:
    archetype_key = archetype.lower()
    if archetype_key == "neobank":
        generate_neobank_dims(cfg, out_dir, seed=cfg.warehouse.seed)
        generate_neobank_facts(cfg, out_dir, out_dir, seed=cfg.warehouse.seed)
        return apply_typical_neobank_defects(cfg, out_dir, seed=cfg.warehouse.seed)
    if archetype_key == "marketplace":
        generate_marketplace_dims(cfg, out_dir, seed=cfg.warehouse.seed)
        generate_marketplace_facts(cfg, out_dir, out_dir, seed=cfg.warehouse.seed)
        return apply_typical_marketplace_defects(cfg, out_dir, seed=cfg.warehouse.seed)
    raise ValueError(f"Unsupported archetype for pipeline: {archetype}")


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * quantile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


__all__ = [
    "inject_key_nulls",
    "inject_fk_failures",
    "inject_duplicates",
    "apply_ingest_lag",
    "inject_null_spikes",
    "inject_schema_gap",
    "apply_typical_neobank_defects",
    "apply_typical_marketplace_defects",
    "run_typical_generation",
]
