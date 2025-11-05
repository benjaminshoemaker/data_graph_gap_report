from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

from data_needs_reporter.utils.hashing import compute_file_hash


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


def _detect_warehouse_archetype(warehouse_path: Path) -> str:
    path = Path(warehouse_path)
    for archetype, hints in WAREHOUSE_ARCHETYPE_HINTS.items():
        if any((path / hint).exists() for hint in hints):
            return archetype
    name = path.name.lower()
    if name in WAREHOUSE_SCHEMA_SPEC:
        return name
    return next(iter(WAREHOUSE_SCHEMA_SPEC.keys()), "neobank")


def validate_warehouse_schema(warehouse_path: Path) -> Dict[str, object]:
    path = Path(warehouse_path)
    if not path.exists():
        detail = f"Warehouse path {path} does not exist."
        return {
            "archetype": None,
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    archetype = _detect_warehouse_archetype(path)
    expected_tables = WAREHOUSE_SCHEMA_SPEC.get(archetype, {})
    issues: List[str] = []

    manifest_path = path / "schema.json"
    tables_manifest: Mapping[str, object] = {}
    if not manifest_path.exists():
        issues.append(f"Missing schema manifest: {manifest_path}")
    else:
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            tables_manifest = manifest_data.get("tables", {})  # type: ignore[assignment]
            if not isinstance(tables_manifest, Mapping):
                issues.append("Schema manifest 'tables' entry must be a mapping.")
                tables_manifest = {}
        except Exception as exc:  # pragma: no cover - invalid JSON rare
            issues.append(f"Unable to read schema manifest: {exc}")
            tables_manifest = {}

    for table_name, columns in expected_tables.items():
        table_path = path / f"{table_name}.parquet"
        if not table_path.exists():
            issues.append(f"Missing table: {table_name}.parquet")
        table_manifest_obj = tables_manifest.get(table_name) if isinstance(tables_manifest, Mapping) else None
        if not isinstance(table_manifest_obj, Mapping):
            issues.append(f"Missing schema entry for table {table_name}")
            continue
        columns_manifest = table_manifest_obj.get("columns")
        if not isinstance(columns_manifest, Mapping):
            issues.append(f"Missing column definitions for table {table_name}")
            continue
        for column_name, expected_type in columns.items():
            column_info = columns_manifest.get(column_name)
            if not isinstance(column_info, Mapping):
                issues.append(f"Missing column {table_name}.{column_name}")
                continue
            actual_type = column_info.get("type")
            if str(actual_type) != expected_type:
                issues.append(
                    f"Column {table_name}.{column_name} expected {expected_type}, found {actual_type}"
                )

    passed = not issues
    detail = (
        f"Schema valid for archetype '{archetype}'."
        if passed
        else "; ".join(issues)
    )
    return {
        "archetype": archetype,
        "passed": passed,
        "issues": issues,
        "detail": detail,
    }


def validate_volume_targets(
    warehouse_path: Path, tolerance: float = 0.10
) -> Dict[str, object]:
    path = Path(warehouse_path)
    if not path.exists():
        detail = f"Warehouse path {path} does not exist."
        return {
            "archetype": None,
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    archetype = _detect_warehouse_archetype(path)
    targets = WAREHOUSE_VOLUME_TARGETS.get(archetype, {})
    issues: List[str] = []

    manifest_path = path / "schema.json"
    tables_manifest: Mapping[str, object] = {}
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            tables_manifest = manifest_data.get("tables", {})  # type: ignore[assignment]
            if not isinstance(tables_manifest, Mapping):
                issues.append("Schema manifest 'tables' entry must be a mapping.")
                tables_manifest = {}
        except Exception as exc:  # pragma: no cover
            issues.append(f"Unable to read schema manifest: {exc}")
            tables_manifest = {}
    else:
        issues.append(f"Missing schema manifest: {manifest_path}")

    for table_name, target_count in targets.items():
        table_manifest = tables_manifest.get(table_name) if isinstance(tables_manifest, Mapping) else None
        if not isinstance(table_manifest, Mapping):
            issues.append(f"Missing manifest entry for table {table_name}")
            continue
        actual_rows = table_manifest.get("rows")
        if actual_rows is None:
            issues.append(f"Missing row count for table {table_name}")
            continue
        try:
            actual_value = int(actual_rows)
        except (TypeError, ValueError):
            issues.append(f"Invalid row count for table {table_name}: {actual_rows}")
            continue
        if actual_value <= 0:
            issues.append(f"Table {table_name} has zero rows.")
            continue
        lower = int(target_count * (1 - tolerance))
        upper = int(target_count * (1 + tolerance))
        if actual_value < lower or actual_value > upper:
            issues.append(
                f"Table {table_name} row count {actual_value} outside ±{int(tolerance * 100)}% of target {target_count}."
            )

    passed = not issues
    detail = (
        f"Volume within ±{int(tolerance * 100)}% band for archetype '{archetype}'."
        if passed
        else "; ".join(issues)
    )
    return {
        "archetype": archetype,
        "passed": passed,
        "issues": issues,
        "detail": detail,
    }


def validate_quality_targets(warehouse_path: Path) -> Dict[str, object]:
    summary_path = Path(warehouse_path) / "data_quality_summary.json"
    if not summary_path.exists():
        detail = f"Quality summary missing at {summary_path}."
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - malformed JSON rare
        detail = f"Unable to read quality summary: {exc}"
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, Mapping):
        detail = "Quality summary missing 'aggregates' section."
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    issues: List[str] = []
    details: List[str] = []
    for metric, (lower, upper) in QUALITY_TARGET_SPEC.items():
        raw_value = aggregates.get(metric)
        try:
            value = float(raw_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            issues.append(f"{metric} missing or not numeric: {raw_value!r}")
            continue

        if not (lower <= value <= upper):
            issues.append(
                f"{metric}={value:.4f} outside target range [{lower:.4f}, {upper:.4f}]."
            )
        details.append(f"{metric}={value:.4f}")

    passed = not issues
    detail = "Quality targets within range." if passed else "; ".join(issues)
    if passed and details:
        detail = f"Quality targets within range ({', '.join(details)})."

    return {
        "passed": passed,
        "issues": issues,
        "detail": detail,
    }


def validate_seasonality_targets(warehouse_path: Path) -> Dict[str, object]:
    polars = _require_polars()
    path = Path(warehouse_path)
    if not path.exists():
        detail = f"Warehouse path {path} does not exist."
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    archetype = _detect_warehouse_archetype(path)
    spec = SEASONALITY_SPECS.get(archetype)
    if spec is None:
        detail = f"No seasonality spec configured for archetype '{archetype}'."
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    table_name = str(spec["table"])
    timestamp_col = str(spec["timestamp"])
    ratio_bounds = spec["weekend_ratio"]  # type: ignore[assignment]

    table_path = path / f"{table_name}.parquet"
    if not table_path.exists():
        detail = f"Missing table for seasonality validation: {table_name}.parquet"
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    try:
        df = polars.read_parquet(table_path)
    except Exception as exc:  # pragma: no cover - unreadable file rare
        detail = f"Unable to read {table_path}: {exc}"
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    if df.height == 0:
        detail = f"Table {table_name} is empty; cannot validate seasonality."
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    if timestamp_col not in df.columns:
        detail = f"Column '{timestamp_col}' missing from {table_name}."
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    filtered = df.filter(polars.col(timestamp_col).is_not_null())
    if filtered.height == 0:
        detail = f"Column '{timestamp_col}' contains only null values."
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    enriched = filtered.with_columns(
        [
            polars.col(timestamp_col).dt.weekday().alias("_dow"),
            polars.col(timestamp_col).dt.hour().alias("_hour"),
        ]
    )

    dow_counts = (
        enriched.group_by("_dow")
        .agg(polars.count().alias("rows"))
        .sort("_dow")
    )

    issues: List[str] = []
    detail_parts: List[str] = []

    weekday_counts = dow_counts.filter(polars.col("_dow") < 5)
    weekend_counts = dow_counts.filter(polars.col("_dow") >= 5)

    weekday_mean = (
        float(weekday_counts["rows"].mean()) if weekday_counts.height else None
    )
    weekend_mean = (
        float(weekend_counts["rows"].mean()) if weekend_counts.height else None
    )

    if not weekday_mean or weekday_mean <= 0.0:
        issues.append("Insufficient weekday activity to compute seasonality factor.")
    if not weekend_mean or weekend_mean <= 0.0:
        issues.append("Insufficient weekend activity to compute seasonality factor.")

    if not issues and weekday_mean:
        weekend_ratio = weekend_mean / weekday_mean
        lower, upper = ratio_bounds  # type: ignore[misc]
        if weekend_ratio < lower or weekend_ratio > upper:
            issues.append(
                f"Weekend/weekday ratio {weekend_ratio:.3f} outside target range [{lower:.2f}, {upper:.2f}]."
            )
        else:
            detail_parts.append(f"weekend_ratio={weekend_ratio:.3f}")

    hour_counts = (
        enriched.group_by("_hour")
        .agg(polars.count().alias("rows"))
        .rename({"_hour": "hour"})
        .sort("rows", descending=True)
    )

    if hour_counts.height < 2:
        issues.append("Insufficient hour diversity to identify peak periods.")
    else:
        top_two = hour_counts.head(2)
        top_hours = [int(value) for value in top_two["hour"].to_list()]
        if not all(hour in PEAK_ALLOWED_HOURS for hour in top_hours):
            issues.append(
                f"Top hours {top_hours} must fall within 10–14 or 17–21."
            )
        else:
            has_midday = any(hour in MIDDAY_HOURS for hour in top_hours)
            has_evening = any(hour in EVENING_HOURS for hour in top_hours)
            if not has_midday or not has_evening:
                issues.append(
                    f"Top hours {top_hours} must include one in 10–14 and one in 17–21."
                )
            else:
                detail_parts.append(f"top_hours={top_hours}")

    passed = not issues
    detail = "Seasonality targets satisfied." if passed else "; ".join(issues)
    if passed and detail_parts:
        detail = f"Seasonality targets satisfied ({', '.join(detail_parts)})."

    return {
        "passed": passed,
        "issues": issues,
        "detail": detail,
    }


def validate_taxonomy_targets(warehouse_path: Path) -> Dict[str, object]:
    polars = _require_polars()
    path = Path(warehouse_path)
    if not path.exists():
        detail = f"Warehouse path {path} does not exist."
        return {
            "passed": False,
            "issues": [detail],
            "detail": detail,
        }

    archetype = _detect_warehouse_archetype(path)
    if archetype == "neobank":
        merchant_path = path / "dim_merchant.parquet"
        if not merchant_path.exists():
            detail = f"Missing merchant dimension at {merchant_path}."
            return {
                "passed": False,
                "issues": [detail],
                "detail": detail,
            }
        try:
            merchants = polars.read_parquet(merchant_path)
        except Exception as exc:  # pragma: no cover - unreadable file rare
            detail = f"Unable to read {merchant_path}: {exc}"
            return {
                "passed": False,
                "issues": [detail],
                "detail": detail,
            }
        if "mcc" not in merchants.columns:
            detail = "Column 'mcc' missing from dim_merchant."
            return {
                "passed": False,
                "issues": [detail],
                "detail": detail,
            }
        unique_mcc = int(
            merchants.select(polars.col("mcc").drop_nulls().n_unique())[0, 0]
        )
        if unique_mcc < NEOBANK_MCC_MIN:
            detail = (
                f"MCC coverage too narrow: {unique_mcc} < {NEOBANK_MCC_MIN} distinct MCCs."
            )
            return {"passed": False, "issues": [detail], "detail": detail}
        detail = f"MCC coverage satisfied ({unique_mcc} unique MCCs)."
        return {"passed": True, "issues": [], "detail": detail}

    if archetype == "marketplace":
        category_path = path / "dim_category.parquet"
        listing_path = path / "dim_listing.parquet"
        order_items_path = path / "fact_order_item.parquet"
        missing = [
            str(p.name)
            for p in [category_path, listing_path, order_items_path]
            if not p.exists()
        ]
        if missing:
            detail = f"Missing required marketplace tables: {', '.join(missing)}."
            return {"passed": False, "issues": [detail], "detail": detail}
        try:
            categories = polars.read_parquet(category_path)
            listings = polars.read_parquet(listing_path)
            order_items = polars.read_parquet(order_items_path)
        except Exception as exc:  # pragma: no cover
            detail = f"Unable to read taxonomy tables: {exc}"
            return {
                "passed": False,
                "issues": [detail],
                "detail": detail,
            }
        if categories.height == 0 or listings.height == 0 or order_items.height == 0:
            detail = "Marketplace taxonomy tables contain no data."
            return {"passed": False, "issues": [detail], "detail": detail}
        if not {"category_id", "parent_id"}.issubset(categories.columns):
            detail = "dim_category must include 'category_id' and 'parent_id'."
            return {"passed": False, "issues": [detail], "detail": detail}
        if "category_id" not in listings.columns:
            detail = "dim_listing missing 'category_id'."
            return {"passed": False, "issues": [detail], "detail": detail}
        if not {"listing_id", "qty", "item_price_cents"}.issubset(order_items.columns):
            detail = "fact_order_item missing required columns."
            return {"passed": False, "issues": [detail], "detail": detail}

        cat_top_df = categories.select(
            [
                polars.col("category_id").cast(polars.Int64),
                polars.col("parent_id").cast(polars.Int64),
            ]
        ).with_columns(
            polars.when(polars.col("parent_id").is_null())
            .then(polars.col("category_id"))
            .otherwise(polars.col("parent_id"))
            .alias("_top_category_id")
        )

        top_level_ids = {
            int(val)
            for val in cat_top_df.filter(polars.col("parent_id").is_null())
            .get_column("_top_category_id")
            .drop_nulls()
            .to_list()
        }
        issues: List[str] = []
        if len(top_level_ids) < MARKETPLACE_CATEGORY_MIN:
            issues.append(
                f"Only {len(top_level_ids)} top-level categories found; expected at least {MARKETPLACE_CATEGORY_MIN}."
            )

        listing_top_df = listings.select(
            [
                polars.col("listing_id").cast(polars.Int64),
                polars.col("category_id").cast(polars.Int64),
            ]
        ).join(
            cat_top_df.select(
                [
                    polars.col("category_id"),
                    polars.col("_top_category_id"),
                ]
            ),
            on="category_id",
            how="left",
        )

        gmv_df = order_items.select(
            [
                polars.col("listing_id").cast(polars.Int64),
                polars.col("qty").cast(polars.Int64),
                polars.col("item_price_cents").cast(polars.Int64),
            ]
        ).join(listing_top_df, on="listing_id", how="left")

        gmv_df = gmv_df.with_columns(
            (polars.col("qty") * polars.col("item_price_cents")).alias("_gmv_cents")
        )

        missing_listings = (
            gmv_df.filter(polars.col("_top_category_id").is_null())
            .select("listing_id")
            .to_series()
            .drop_nulls()
            .to_list()
        )
        if missing_listings:
            issues.append(
                "No top-level category mapping for listings: "
                + ", ".join(str(int(val)) for val in sorted(set(missing_listings)))
            )

        gmv_by_cat = (
            gmv_df.filter(polars.col("_top_category_id").is_not_null())
            .group_by("_top_category_id")
            .agg(polars.col("_gmv_cents").sum().alias("gmv_cents"))
        )

        total_gmv = int(gmv_by_cat["gmv_cents"].sum()) if gmv_by_cat.height else 0

        gmv_per_category: Dict[int, int] = {
            int(cat): int(val)
            for cat, val in zip(
                gmv_by_cat["_top_category_id"].to_list(),
                gmv_by_cat["gmv_cents"].to_list(),
            )
        }

        if total_gmv <= 0:
            issues.append("No GMV recorded across order items.")

        if not issues:
            missing_top = [
                cat_id for cat_id in top_level_ids if gmv_per_category.get(cat_id, 0) <= 0
            ]
            if missing_top:
                issues.append(
                    f"Top-level categories with zero GMV: {', '.join(map(str, missing_top))}."
                )

        if not issues and gmv_per_category:
            max_cat = max(gmv_per_category.values())
            share = max_cat / total_gmv if total_gmv else 0.0
            if share > MARKETPLACE_MAX_TOP_SHARE:
                issues.append(
                    f"Top category GMV share {share:.3f} exceeds {MARKETPLACE_MAX_TOP_SHARE:.2f}."
                )

        if issues:
            detail = "; ".join(issues)
            return {"passed": False, "issues": issues, "detail": detail}

        detail = "Marketplace taxonomy satisfied (all categories covered, balanced GMV)."
        return {"passed": True, "issues": [], "detail": detail}

    detail = f"Unsupported archetype '{archetype}' for taxonomy validation."
    return {"passed": False, "issues": [detail], "detail": detail}


def validate_monetization_targets(warehouse_path: Path) -> Dict[str, object]:
    polars = _require_polars()
    path = Path(warehouse_path)
    if not path.exists():
        detail = f"Warehouse path {path} does not exist."
        return {"passed": False, "issues": [detail], "detail": detail}

    archetype = _detect_warehouse_archetype(path)
    issues: List[str] = []
    detail_parts: List[str] = []

    if archetype == "neobank":
        txn_path = path / "fact_card_transaction.parquet"
        customer_path = path / "dim_customer.parquet"
        invoice_path = path / "fact_subscription_invoice.parquet"
    elif archetype == "marketplace":
        payment_path = path / "fact_payment.parquet"
    else:
        detail = f"Unsupported archetype '{archetype}' for monetization validation."
        return {"passed": False, "issues": [detail], "detail": detail}

    if archetype == "neobank":
        missing = [
            str(p.name)
            for p in [txn_path, customer_path, invoice_path]
            if not p.exists()
        ]
        if missing:
            detail = f"Missing required monetization tables: {', '.join(missing)}."
            return {"passed": False, "issues": [detail], "detail": detail}
        try:
            transactions = polars.read_parquet(txn_path)
            customers = polars.read_parquet(customer_path)
            invoices = polars.read_parquet(invoice_path)
        except Exception as exc:  # pragma: no cover - unreadable file rare
            detail = f"Unable to read monetization tables: {exc}"
            return {"passed": False, "issues": [detail], "detail": detail}

        if transactions.height == 0:
            issues.append("No transactions available to compute interchange rate.")
        else:
            if not {"amount_cents", "interchange_bps"}.issubset(transactions.columns):
                issues.append(
                    "fact_card_transaction must include 'amount_cents' and 'interchange_bps'."
                )
            else:
                captured = transactions
                if "auth_result" in transactions.columns:
                    captured = captured.with_columns(
                        polars.col("auth_result")
                        .cast(polars.Utf8)
                        .str.to_lowercase()
                        .alias("_auth_lower")
                    ).filter(polars.col("_auth_lower") == "captured")
                    if captured.height == 0:
                        captured = transactions
                spend_cents = float(
                    captured.select(polars.col("amount_cents").cast(polars.Float64).sum())
                    .item()
                    or 0.0
                )
                interchange_usd = float(
                    captured.select(
                        (
                            polars.col("amount_cents").cast(polars.Float64)
                            * polars.col("interchange_bps").cast(polars.Float64)
                            / 10000.0
                            / 100.0
                        ).sum()
                    ).item()
                    or 0.0
                )
                if spend_cents <= 0.0:
                    issues.append("Captured spend is zero; cannot validate interchange rate.")
                else:
                    interchange_rate = interchange_usd / (spend_cents / 100.0)
                    lower, upper = NEOBANK_INTERCHANGE_RANGE
                    if interchange_rate < lower or interchange_rate > upper:
                        issues.append(
                            f"Interchange rate {interchange_rate:.4f} outside [{lower:.3f}, {upper:.3f}]."
                        )
                    else:
                        detail_parts.append(f"interchange_rate={interchange_rate:.4f}")

        if customers.height == 0:
            issues.append("No customers found in dim_customer.")
        else:
            if "customer_id" not in customers.columns:
                issues.append("dim_customer missing 'customer_id'.")
        if invoices.height == 0:
            issues.append("No subscription invoices found; cannot compute attach rate.")
        else:
            if "customer_id" not in invoices.columns:
                issues.append("fact_subscription_invoice missing 'customer_id'.")

        if not issues and customers.height and "customer_id" in customers.columns:
            total_customers = int(
                customers.select(polars.col("customer_id").cast(polars.Int64).n_unique())[
                    0, 0
                ]
            )
            if total_customers <= 0:
                issues.append("Customer population is zero; cannot compute attach rate.")
            elif invoices.height and "customer_id" in invoices.columns:
                attached = int(
                    invoices.select(polars.col("customer_id").cast(polars.Int64).n_unique())[
                        0, 0
                    ]
                )
                attach_rate = attached / total_customers if total_customers else 0.0
                lower, upper = NEOBANK_ATTACH_RANGE
                if attach_rate < lower or attach_rate > upper:
                    issues.append(
                        f"Premium attach rate {attach_rate:.4f} outside [{lower:.2f}, {upper:.2f}]."
                    )
                else:
                    detail_parts.append(f"attach_rate={attach_rate:.4f}")

        passed = not issues
        detail = (
            "Monetization targets satisfied."
            if passed
            else "; ".join(issues)
        )
        if passed and detail_parts:
            detail = f"Monetization targets satisfied ({', '.join(detail_parts)})."
        return {"passed": passed, "issues": issues, "detail": detail}

    if archetype == "marketplace":
        payment_path = path / "fact_payment.parquet"
        if not payment_path.exists():
            detail = f"Missing payment facts at {payment_path}."
            return {"passed": False, "issues": [detail], "detail": detail}
        try:
            payments = polars.read_parquet(payment_path)
        except Exception as exc:  # pragma: no cover
            detail = f"Unable to read {payment_path}: {exc}"
            return {"passed": False, "issues": [detail], "detail": detail}
        if payments.height == 0:
            detail = "No payments available to compute take rate."
            return {"passed": False, "issues": [detail], "detail": detail}
        if not {"buyer_paid_cents", "platform_fee_cents"}.issubset(payments.columns):
            detail = "fact_payment must include 'buyer_paid_cents' and 'platform_fee_cents'."
            return {"passed": False, "issues": [detail], "detail": detail}

        sums = payments.select(
            [
                polars.col("buyer_paid_cents").cast(polars.Float64).sum().alias("paid"),
                polars.col("platform_fee_cents")
                .cast(polars.Float64)
                .sum()
                .alias("fee"),
            ]
        ).row(0)
        total_paid, total_fee = float(sums[0] or 0.0), float(sums[1] or 0.0)
        if total_paid <= 0.0:
            detail = "Total buyer paid amount is zero; cannot compute take rate."
            return {"passed": False, "issues": [detail], "detail": detail}
        take_rate = total_fee / total_paid
        lower, upper = MARKETPLACE_TAKE_RATE_RANGE
        if take_rate < lower or take_rate > upper:
            detail = (
                f"Marketplace take rate {take_rate:.4f} outside [{lower:.2f}, {upper:.2f}]."
            )
            return {"passed": False, "issues": [detail], "detail": detail}
        detail = f"Monetization targets satisfied (take_rate={take_rate:.4f})."
        return {"passed": True, "issues": [], "detail": detail}

    detail = f"Unsupported archetype '{archetype}' for monetization validation."
    return {"passed": False, "issues": [detail], "detail": detail}


def validate_trajectory_targets(warehouse_path: Path) -> Dict[str, object]:
    polars = _require_polars()
    path = Path(warehouse_path)
    if not path.exists():
        detail = f"Warehouse path {path} does not exist."
        return {"passed": False, "issues": [detail], "detail": detail}

    archetype = _detect_warehouse_archetype(path)
    if archetype != "neobank":
        detail = f"Trajectory validation not applicable for archetype '{archetype}'."
        return {"passed": True, "issues": [], "detail": detail}

    customer_path = path / "dim_customer.parquet"
    invoice_path = path / "fact_subscription_invoice.parquet"

    missing = [
        str(p.name)
        for p in [customer_path, invoice_path]
        if not p.exists()
    ]
    if missing:
        detail = f"Missing required trajectory tables: {', '.join(missing)}."
        return {"passed": False, "issues": [detail], "detail": detail}

    try:
        customers = polars.read_parquet(customer_path)
        invoices = polars.read_parquet(invoice_path)
    except Exception as exc:  # pragma: no cover - unreadable file rare
        detail = f"Unable to read trajectory tables: {exc}"
        return {"passed": False, "issues": [detail], "detail": detail}

    issues: List[str] = []
    detail_parts: List[str] = []

    if customers.height == 0:
        issues.append("Customer table empty; cannot evaluate trajectory.")
    else:
        growth = (
            customers.with_columns(
                polars.col("created_at")
                .dt.strftime("%Y-%m")
                .alias("_month")
            )
            .group_by("_month")
            .agg(polars.count().alias("new_customers"))
            .sort("_month")
        )
        months = growth.get_column("_month").to_list()
        counts = [int(val) for val in growth.get_column("new_customers").to_list()]
        if len(counts) < TRAJECTORY_MIN_MONTHS:
            issues.append(
                f"Insufficient months ({len(counts)}) to confirm growth trajectory."
            )
        else:
            for idx in range(1, len(counts)):
                prev = counts[idx - 1]
                curr = counts[idx]
                if prev <= 0:
                    continue
                if curr + 1e-6 < prev * (1 - TRAJECTORY_GROWTH_TOLERANCE):
                    issues.append(
                        f"Month {months[idx]} customer adds {curr} below allowed drop from {prev}."
                    )
                    break
            if counts[-1] < counts[0]:
                issues.append(
                    "Customer growth not increasing over timeframe."
                )
            if not issues:
                detail_parts.append("growth_monotonic")

        if "kyc_status" not in customers.columns:
            issues.append("dim_customer missing 'kyc_status'.")
        else:
            total_customers = customers.height
            if total_customers > 0:
                verified = customers.filter(polars.col("kyc_status") == "verified").height
                verified_share = verified / total_customers
                lower = KYC_VERIFIED_TARGET - KYC_TOLERANCE
                upper = KYC_VERIFIED_TARGET + KYC_TOLERANCE
                if verified_share < lower or verified_share > upper:
                    issues.append(
                        f"Verified KYC share {verified_share:.4f} outside [{lower:.4f}, {upper:.4f}]."
                    )
                else:
                    detail_parts.append(f"kyc_verified={verified_share:.4f}")

    if invoices.height == 0:
        issues.append("No subscription invoices; cannot evaluate churn.")
    else:
        unique = (
            invoices.with_columns(
                polars.col("paid_at")
                .dt.strftime("%Y-%m")
                .alias("_month")
            )
            .select(["customer_id", "_month"])
            .unique()
        )
        month_values = sorted(set(unique.get_column("_month").to_list()))
        if len(month_values) < 2:
            issues.append("Insufficient subscription months to evaluate churn.")
        else:
            month_sets: Dict[object, set[int]] = {}
            for month in month_values:
                active_ids = unique.filter(polars.col("_month") == month).get_column("customer_id")
                month_sets[month] = {int(val) for val in active_ids.to_list()}
            total_active = 0
            churn_events = 0
            for idx in range(len(month_values) - 1):
                current_month = month_values[idx]
                next_month = month_values[idx + 1]
                current_active = month_sets.get(current_month, set())
                next_active = month_sets.get(next_month, set())
                if not current_active:
                    continue
                total_active += len(current_active)
                churn_events += len(current_active - next_active)
            if total_active <= 0:
                issues.append("Unable to compute churn rate due to zero active subscribers.")
            else:
                churn_rate = churn_events / total_active
                lower = CHURN_TARGET - CHURN_TOLERANCE
                upper = CHURN_TARGET + CHURN_TOLERANCE
                if churn_rate < lower or churn_rate > upper:
                    issues.append(
                        f"Churn rate {churn_rate:.4f} outside [{lower:.4f}, {upper:.4f}]."
                    )
                else:
                    detail_parts.append(f"churn_rate={churn_rate:.4f}")

    passed = not issues
    detail = "Trajectory targets satisfied." if passed else "; ".join(issues)
    if passed and detail_parts:
        detail = f"Trajectory targets satisfied ({', '.join(detail_parts)})."
    return {"passed": passed, "issues": issues, "detail": detail}


def validate_comms_targets(comms_path: Path) -> Dict[str, object]:
    polars = _require_polars()
    path = Path(comms_path)
    if not path.exists():
        detail = f"Comms path {path} does not exist."
        return {"passed": False, "issues": [detail], "detail": detail}

    issues: List[str] = []
    detail_parts: List[str] = []

    total_threads = 0
    bucket_counts: Dict[str, int] = {}
    exec_count = 0

    nlq_token_cap_ok = False
    for filename, (lower, upper) in COMMS_VOLUME_SPEC.items():
        file_path = path / filename
        if not file_path.exists():
            issues.append(f"Missing communications file: {filename}")
            continue
        try:
            data = polars.read_parquet(file_path)
        except Exception as exc:  # pragma: no cover - unreadable file rare
            issues.append(f"Unable to read {filename}: {exc}")
            continue
        volume = data.height
        if volume < lower or volume > upper:
            issues.append(
                f"{filename} volume {volume} outside [{lower}, {upper}] target."
            )
        else:
            detail_parts.append(f"{filename}={volume}")

        if "thread_id" in data.columns:
            total_threads += data.select(polars.col("thread_id")).n_unique()

        if "bucket" in data.columns:
            counts = (
                data.group_by("bucket")
                .agg(polars.count().alias("count"))
                .to_dict(as_series=False)
            )
            for bucket, count in zip(counts["bucket"], counts["count"]):
                bucket_counts[str(bucket)] = bucket_counts.get(str(bucket), 0) + int(count)

        if filename == "slack_messages.parquet" and "is_exec" in data.columns:
            exec_count = data.filter(polars.col("is_exec") == True).height
        if filename == "nlq.parquet":
            if "tokens" not in data.columns:
                issues.append("NLQ tokens column missing; cannot evaluate cap.")
            else:
                over_cap = data.filter(polars.col("tokens") > NLQ_TOKEN_CAP)
                if over_cap.height > 0:
                    max_tokens = (
                        data.select(polars.col("tokens").max()).item() or NLQ_TOKEN_CAP
                    )
                    issues.append(
                        f"Found {over_cap.height} NLQ queries over {NLQ_TOKEN_CAP} tokens (max {int(max_tokens)})."
                    )
                else:
                    nlq_token_cap_ok = True

    total_messages = sum(bucket_counts.values())
    if total_messages <= 0:
        issues.append("No communication messages found to evaluate bucket mix.")
    else:
        for bucket, count in bucket_counts.items():
            share = count / total_messages
            if share < COMMS_BUCKET_MIN_SHARE:
                issues.append(
                    f"Bucket '{bucket}' share {share:.3f} below {COMMS_BUCKET_MIN_SHARE:.2f}."
                )
        if not issues:
            detail_parts.append("bucket_mix_ok")

    if total_threads <= 0:
        issues.append("Unable to compute executive thread share; no thread information.")
    else:
        exec_share = exec_count / total_threads
        lower, upper = EXEC_THREAD_SHARE_RANGE
        if exec_share < lower or exec_share > upper:
            issues.append(
                f"Executive thread share {exec_share:.3f} outside [{lower:.2f}, {upper:.2f}]."
            )
        else:
            detail_parts.append(f"exec_share={exec_share:.3f}")

    if nlq_token_cap_ok:
        detail_parts.append("nlq_tokens_ok")

    parse_summary_path = path / NLQ_PARSE_SUMMARY_FILE
    if parse_summary_path.exists():
        try:
            parse_summary = json.loads(parse_summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - malformed JSON rare
            issues.append(f"Unable to read {NLQ_PARSE_SUMMARY_FILE}: {exc}")
        else:
            parse_only_value = parse_summary.get("parse_only")
            parse_mode = parse_summary.get("mode")
            parse_only = False
            if isinstance(parse_only_value, str):
                parse_only = parse_only_value.strip().lower() in {"true", "1", "yes"}
            elif isinstance(parse_only_value, (int, float)):
                parse_only = bool(parse_only_value)
            elif isinstance(parse_only_value, bool):
                parse_only = parse_only_value
            if isinstance(parse_mode, str) and parse_mode.strip().lower().startswith(
                "parse"
            ):
                parse_only = True
            if parse_only:
                total = parse_summary.get("queries_total")
                parsed = parse_summary.get("queries_parsed")
                success_pct_value = parse_summary.get("parse_success_pct")
                success_rate: Optional[float] = None

                if success_pct_value is not None:
                    try:
                        success_rate = float(success_pct_value)
                    except (TypeError, ValueError):
                        success_rate = None
                    else:
                        if success_rate > 1.5:
                            success_rate = success_rate / 100.0

                if success_rate is None and total is not None and parsed is not None:
                    try:
                        total_float = float(total)
                        parsed_float = float(parsed)
                        if total_float > 0:
                            success_rate = parsed_float / total_float
                    except (TypeError, ValueError):
                        success_rate = None

                if success_rate is None:
                    issues.append(
                        "Unable to compute NLQ parse success rate with parse-only enabled."
                    )
                else:
                    if success_rate < NLQ_PARSE_SUCCESS_THRESHOLD:
                        issues.append(
                            f"NLQ parse-only success {success_rate:.3f} below {NLQ_PARSE_SUCCESS_THRESHOLD:.2f} threshold."
                        )
                    else:
                        detail_parts.append(f"nlq_parse_success={success_rate:.3f}")
    passed = not issues
    detail = "Comms targets satisfied." if passed else "; ".join(issues)
    if passed and detail_parts:
        detail = f"Comms targets satisfied ({', '.join(detail_parts)})."
    return {"passed": passed, "issues": issues, "detail": detail}


def validate_theme_mix_targets(comms_path: Path) -> Dict[str, object]:
    path = Path(comms_path)
    budget_path = path / "budget.json"
    if not budget_path.exists():
        detail = f"Missing communications budget at {budget_path}."
        return {"passed": False, "issues": [detail], "detail": detail}

    try:
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - malformed JSON rare
        detail = f"Unable to read budget: {exc}"
        return {"passed": False, "issues": [detail], "detail": detail}

    quotas = budget.get("quotas")
    coverage = budget.get("coverage")
    if not isinstance(quotas, Mapping) or not isinstance(coverage, Mapping):
        detail = "Budget file missing 'quotas' or 'coverage' sections."
        return {"passed": False, "issues": [detail], "detail": detail}

    plan_totals: Dict[str, float] = {}
    for source_info in quotas.values():
        bucket_totals = (
            source_info.get("bucket_totals")
            if isinstance(source_info, Mapping)
            else None
        )
        if not isinstance(bucket_totals, Mapping):
            continue
        for bucket, value in bucket_totals.items():
            try:
                amount = float(value)
            except (TypeError, ValueError):
                amount = 0.0
            plan_totals[str(bucket)] = plan_totals.get(str(bucket), 0.0) + amount

    actual_totals: Dict[str, float] = {}
    for source_info in coverage.values():
        per_bucket = (
            source_info.get("per_bucket")
            if isinstance(source_info, Mapping)
            else None
        )
        if not isinstance(per_bucket, Mapping):
            continue
        for bucket, entry in per_bucket.items():
            if not isinstance(entry, Mapping):
                continue
            actual = entry.get("actual")
            try:
                amount = float(actual)
            except (TypeError, ValueError):
                amount = 0.0
            actual_totals[str(bucket)] = actual_totals.get(str(bucket), 0.0) + amount

    plan_sum = sum(plan_totals.values())
    actual_sum = sum(actual_totals.values())
    issues: List[str] = []
    detail_parts: List[str] = []

    if plan_sum <= 0.0:
        issues.append("Theme plan totals are zero; cannot validate mix.")
    if actual_sum <= 0.0:
        issues.append("Theme actual totals are zero; cannot validate mix.")

    if not issues:
        buckets = set(plan_totals) | set(actual_totals)
        for bucket in buckets:
            plan_share = plan_totals.get(bucket, 0.0) / plan_sum
            actual_share = actual_totals.get(bucket, 0.0) / actual_sum
            diff = abs(actual_share - plan_share)
            if diff > THEME_SHARE_TOLERANCE:
                issues.append(
                    f"Theme '{bucket}' share diff {diff:.3f} exceeds {THEME_SHARE_TOLERANCE:.2f}."
                )
            else:
                detail_parts.append(
                    f"{bucket}={actual_share:.3f}/{plan_share:.3f}"
                )

    passed = not issues
    detail = "Theme mix targets satisfied." if passed else "; ".join(issues)
    if passed and detail_parts:
        detail = f"Theme mix targets satisfied ({', '.join(detail_parts)})."
    return {"passed": passed, "issues": issues, "detail": detail}


def validate_event_correlation(warehouse_path: Path, comms_path: Path) -> Dict[str, object]:
    polars = _require_polars()
    warehouse = Path(warehouse_path)
    comms = Path(comms_path)

    issues: List[str] = []
    detail_parts: List[str] = []

    dq_summary_path = warehouse / "data_quality_summary.json"
    summary: Optional[Mapping[str, object]] = None
    if dq_summary_path.exists():
        try:
            summary = json.loads(dq_summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - malformed JSON rare
            issues.append(f"Unable to read data_quality_summary.json: {exc}")
    spike_events = _collect_spike_events(summary)

    schema_event = _detect_schema_gap_event(
        polars, warehouse / "fact_card_transaction.parquet"
    )
    events = list(spike_events)
    if schema_event is not None:
        events.append(schema_event)

    if not events:
        detail = (
            "; ".join(issues)
            if issues
            else "Event correlation satisfied (no events detected)."
        )
        return {"passed": not issues, "issues": issues, "detail": detail}

    slack_times = _load_thread_times(
        polars, comms / "slack_messages.parquet", "thread_id", "sent_at"
    )
    email_times = _load_thread_times(
        polars, comms / "email_messages.parquet", "thread_id", "sent_at"
    )
    all_times = sorted(slack_times + email_times)
    if not all_times:
        issues.append("No Slack or Email threads available for correlation.")

    correlated_count = 0
    for event in events:
        timestamp = event["timestamp"]
        event_type = event.get("type", "event")
        if all_times and _has_thread_in_window(
            all_times, timestamp, EVENT_CORRELATION_WINDOW
        ):
            correlated_count += 1
        else:
            issues.append(
                f"No Slack/Email thread within {EVENT_CORRELATION_WINDOW_HOURS}h of {event_type} at {timestamp.isoformat()}."
            )

    if correlated_count:
        detail_parts.append(f"correlated={correlated_count}/{len(events)}")

    passed = not issues
    if passed:
        detail = (
            f"Event correlation satisfied ({', '.join(detail_parts)})"
            if detail_parts
            else "Event correlation satisfied."
        )
    else:
        detail = "; ".join(issues)
    return {"passed": passed, "issues": issues, "detail": detail}


def _coerce_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_reproducibility(warehouse_path: Path, comms_path: Path) -> Dict[str, object]:
    issues: List[str] = []
    detail_parts: List[str] = []

    warehouse_manifest_path = Path(warehouse_path) / "hashes.json"
    warehouse_hash_ok = False
    if not warehouse_manifest_path.exists():
        issues.append(f"Missing warehouse hash manifest at {warehouse_manifest_path}.")
    else:
        try:
            manifest = json.loads(warehouse_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - malformed JSON rare
            issues.append(f"Unable to read warehouse hash manifest: {exc}")
        else:
            algorithm = str(manifest.get("hash_algorithm") or "sha256").lower()
            if algorithm != "sha256":
                issues.append(f"Unsupported warehouse hash algorithm '{algorithm}'.")
            seeds = manifest.get("seeds")
            if not isinstance(seeds, Mapping) or not seeds:
                issues.append("Warehouse hash manifest missing seeds.")
            files = manifest.get("files")
            if not isinstance(files, Mapping) or not files:
                issues.append("Warehouse hash manifest missing file hashes.")
            else:
                mismatches: List[str] = []
                for name, expected_hash in files.items():
                    file_path = Path(warehouse_path) / str(name)
                    if not file_path.exists():
                        mismatches.append(f"{name} missing")
                        continue
                    actual_hash = compute_file_hash(file_path)
                    if actual_hash != str(expected_hash):
                        mismatches.append(f"{name} mismatch")
                if mismatches:
                    issues.append("Warehouse hash mismatch: " + ", ".join(mismatches))
                else:
                    warehouse_hash_ok = True

    comms_budget_path = Path(comms_path) / "budget.json"
    comms_hash_ok = False
    if not comms_budget_path.exists():
        issues.append(f"Missing communications budget at {comms_budget_path}.")
    else:
        try:
            budget = json.loads(comms_budget_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - malformed JSON rare
            issues.append(f"Unable to read communications budget: {exc}")
        else:
            hashes = budget.get("hashes")
            if not isinstance(hashes, Mapping):
                issues.append("Communications budget missing 'hashes' entry.")
            else:
                algorithm = str(hashes.get("algorithm") or "sha256").lower()
                if algorithm != "sha256":
                    issues.append(f"Unsupported communications hash algorithm '{algorithm}'.")
                files = hashes.get("files")
                if not isinstance(files, Mapping) or not files:
                    issues.append("Communications hash manifest missing file hashes.")
                else:
                    mismatches: List[str] = []
                    for name, expected_hash in files.items():
                        file_path = Path(comms_path) / str(name)
                        if not file_path.exists():
                            mismatches.append(f"{name} missing")
                            continue
                        actual_hash = compute_file_hash(file_path)
                        if actual_hash != str(expected_hash):
                            mismatches.append(f"{name} mismatch")
                    if mismatches:
                        issues.append("Communications hash mismatch: " + ", ".join(mismatches))
                    else:
                        comms_hash_ok = True
            seeds = budget.get("seeds")
            if not isinstance(seeds, Mapping) or not seeds:
                issues.append("Communications budget missing seeds.")

    if warehouse_hash_ok:
        detail_parts.append("warehouse_hashes_ok")
    if comms_hash_ok:
        detail_parts.append("comms_hashes_ok")

    passed = not issues
    if passed:
        if detail_parts:
            detail = f"Reproducibility satisfied ({', '.join(detail_parts)})."
        else:
            detail = "Reproducibility satisfied."
    else:
        detail = "; ".join(issues)
    return {"passed": passed, "issues": issues, "detail": detail}


def validate_spend_caps(comms_path: Path) -> Dict[str, object]:
    budget_path = Path(comms_path) / "budget.json"
    if not budget_path.exists():
        detail = f"Missing communications budget at {budget_path}."
        return {"passed": False, "issues": [detail], "detail": detail}

    issues: List[str] = []
    detail_parts: List[str] = []
    try:
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - malformed JSON rare
        detail = f"Unable to read communications budget: {exc}"
        return {"passed": False, "issues": [detail], "detail": detail}

    cap_usd = _coerce_float(budget.get("cap_usd"))
    cost_usd = _coerce_float(budget.get("cost_usd"))
    price_per_1k = _coerce_float(budget.get("price_per_1k_tokens"))
    tokens_used = budget.get("tokens_used")
    token_budget = budget.get("token_budget")

    if cap_usd is None:
        issues.append("Communications budget missing cap_usd.")
    if cost_usd is None:
        issues.append("Communications budget missing cost_usd.")
    if price_per_1k is None:
        issues.append("Communications budget missing price_per_1k_tokens.")

    if issues:
        return {"passed": False, "issues": issues, "detail": "; ".join(issues)}

    if cost_usd > cap_usd + 1e-9:
        issues.append(
            f"LLM spend {cost_usd:.4f} exceeds cap {cap_usd:.4f}."
        )

    if bool(budget.get("stopped_due_to_cap")):
        issues.append("Cost guard reported cap stop.")

    if isinstance(tokens_used, (int, float)) and price_per_1k is not None:
        expected_cost = round((float(tokens_used) / 1000.0) * price_per_1k, 4)
        recorded_cost = round(cost_usd, 4)
        if expected_cost != recorded_cost:
            issues.append(
                f"Recorded cost {recorded_cost:.4f} does not match token usage estimate {expected_cost:.4f}."
            )
    else:
        issues.append("Missing tokens_used for spend validation.")

    if isinstance(token_budget, (int, float)) and isinstance(tokens_used, (int, float)):
        if float(tokens_used) > float(token_budget) + 1e-9:
            issues.append("Token usage exceeded budgeted allowance.")

    if not issues:
        detail_parts.append(f"cost_usd={cost_usd:.4f}/{cap_usd:.4f}")

    passed = not issues
    if passed:
        detail = (
            f"Spend caps satisfied ({', '.join(detail_parts)})."
            if detail_parts
            else "Spend caps satisfied."
        )
    else:
        detail = "; ".join(issues)
    return {"passed": passed, "issues": issues, "detail": detail}


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

QUALITY_TARGET_SPEC: Dict[str, Tuple[float, float]] = {
    "merchant_key_null_pct": (0.01, 0.03),
    "card_fk_orphan_pct": (0.03, 0.06),
    "txn_dup_key_pct": (0.005, 0.01),
    "txn_p95_ingest_lag_min": (120.0, 240.0),
}

SEASONALITY_SPECS: Dict[str, Dict[str, object]] = {
    "neobank": {
        "table": "fact_card_transaction",
        "timestamp": "event_time",
        "weekend_ratio": (1.05, 1.30),
    },
    "marketplace": {
        "table": "fact_order",
        "timestamp": "order_time",
        "weekend_ratio": (1.15, 1.45),
    },
}

MIDDAY_HOURS = set(range(10, 15))
EVENING_HOURS = set(range(17, 22))
PEAK_ALLOWED_HOURS = MIDDAY_HOURS.union(EVENING_HOURS)
NEOBANK_MCC_MIN = 35
MARKETPLACE_CATEGORY_MIN = 10
MARKETPLACE_MAX_TOP_SHARE = 0.30
NEOBANK_INTERCHANGE_RANGE = (0.013, 0.017)
NEOBANK_ATTACH_RANGE = (0.06, 0.10)
MARKETPLACE_TAKE_RATE_RANGE = (0.11, 0.13)
TRAJECTORY_GROWTH_TOLERANCE = 0.05
TRAJECTORY_MIN_MONTHS = 4
KYC_VERIFIED_TARGET = 0.70
KYC_TOLERANCE = 0.005
CHURN_TARGET = 0.018
CHURN_TOLERANCE = 0.005
COMMS_VOLUME_SPEC = {
    "slack_messages.parquet": (2850, 3150),
    "email_messages.parquet": (760, 840),
    "nlq.parquet": (950, 1050),
}
COMMS_BUCKET_MIN_SHARE = 0.15
EXEC_THREAD_SHARE_RANGE = (0.05, 0.12)
THEME_SHARE_TOLERANCE = 0.08
NLQ_TOKEN_CAP = 30
NLQ_PARSE_SUCCESS_THRESHOLD = 0.95
NLQ_PARSE_SUMMARY_FILE = "nlq_parse_summary.json"
EVENT_CORRELATION_WINDOW_HOURS = 24
EVENT_CORRELATION_WINDOW = timedelta(hours=EVENT_CORRELATION_WINDOW_HOURS)
SCHEMA_GAP_MIN_ROWS = 500
SCHEMA_GAP_MIN_GAP_HOURS = 6


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_event_timestamp(value: object) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _ensure_timezone(value)
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    return _ensure_timezone(parsed)


def _collect_spike_events(summary: Optional[Mapping[str, object]]) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    if not isinstance(summary, Mapping):
        return events
    per_table = summary.get("per_table")
    if not isinstance(per_table, Mapping):
        return events
    table_summary = per_table.get("fact_card_transaction")
    if not isinstance(table_summary, Mapping):
        return events
    spike_days = table_summary.get("spike_days")
    if not isinstance(spike_days, Sequence):
        return events
    for entry in spike_days:
        if not isinstance(entry, Mapping):
            continue
        raw_ts = entry.get("event_day") or entry.get("day")
        timestamp = _parse_event_timestamp(raw_ts)
        if timestamp is None:
            continue
        events.append(
            {
                "type": "null_spike",
                "timestamp": timestamp,
                "detail": entry,
            }
        )
    return events


def _detect_schema_gap_event(polars, table_path: Path) -> Optional[Dict[str, object]]:
    if not table_path.exists():
        return None
    try:
        df = polars.read_parquet(table_path, columns=["event_time"])
    except Exception:  # pragma: no cover - unreadable file
        return None
    if df.height < SCHEMA_GAP_MIN_ROWS or "event_time" not in df.columns:
        return None
    event_series = (
        df.select(polars.col("event_time").drop_nulls().alias("_event_time"))
        .sort("_event_time")
        .get_column("_event_time")
    )
    if event_series.len() < 2:
        return None
    times = [_ensure_timezone(ts) for ts in event_series.to_list() if ts is not None]
    if len(times) < 2:
        return None
    max_gap = timedelta(0)
    gap_start: Optional[datetime] = None
    for prev, curr in zip(times, times[1:]):
        gap = curr - prev
        if gap > max_gap:
            max_gap = gap
            gap_start = prev
    if max_gap >= timedelta(hours=SCHEMA_GAP_MIN_GAP_HOURS) and gap_start is not None:
        midpoint = gap_start + max_gap / 2
        return {
            "type": "schema_gap",
            "timestamp": midpoint,
            "gap_hours": max_gap.total_seconds() / 3600.0,
        }
    return None


def _load_thread_times(
    polars,
    file_path: Path,
    thread_col: str,
    ts_col: str,
) -> List[datetime]:
    if not file_path.exists():
        return []
    try:
        df = polars.read_parquet(file_path, columns=[thread_col, ts_col])
    except Exception:  # pragma: no cover - unreadable file
        return []
    if df.height == 0 or thread_col not in df.columns or ts_col not in df.columns:
        return []
    grouped = (
        df.drop_nulls([thread_col, ts_col])
        .group_by(thread_col)
        .agg(polars.col(ts_col).min().alias("_thread_time"))
        .select("_thread_time")
    )
    if grouped.height == 0:
        return []
    times = [_ensure_timezone(val) for val in grouped["_thread_time"].to_list() if val is not None]
    times.sort()
    return times


def _has_thread_in_window(
    sorted_times: Sequence[datetime],
    target: datetime,
    window: timedelta,
) -> bool:
    if not sorted_times:
        return False
    target = _ensure_timezone(target)
    idx = bisect_left(sorted_times, target)
    candidates = []
    if idx < len(sorted_times):
        candidates.append(sorted_times[idx])
    if idx > 0:
        candidates.append(sorted_times[idx - 1])
    for ts in candidates:
        if abs(ts - target) <= window:
            return True
    return False

WAREHOUSE_SCHEMA_SPEC: Dict[str, Dict[str, Dict[str, str]]] = {
    "neobank": {
        "dim_customer": {
            "customer_id": "int64",
            "created_at": "timestamp[us, tz=UTC]",
            "kyc_status": "string",
        },
        "dim_account": {
            "account_id": "int64",
            "customer_id": "int64",
            "type": "string",
            "created_at": "timestamp[us, tz=UTC]",
            "status": "string",
        },
        "dim_card": {
            "card_id": "int64",
            "account_id": "int64",
            "status": "string",
            "activated_at": "timestamp[us, tz=UTC]",
        },
        "dim_merchant": {
            "merchant_id": "int64",
            "mcc": "int32",
            "name": "string",
        },
        "dim_plan": {
            "plan_id": "int64",
            "name": "string",
            "price_cents": "int64",
            "cadence": "string",
        },
        "fact_card_transaction": {
            "txn_id": "int64",
            "card_id": "int64",
            "merchant_id": "int64",
            "event_time": "timestamp[us, tz=UTC]",
            "amount_cents": "int64",
            "interchange_bps": "double",
            "channel": "string",
            "auth_result": "string",
            "loaded_at": "timestamp[us, tz=UTC]",
        },
        "fact_subscription_invoice": {
            "invoice_id": "int64",
            "customer_id": "int64",
            "plan_id": "int64",
            "period_start": "timestamp[us, tz=UTC]",
            "period_end": "timestamp[us, tz=UTC]",
            "paid_at": "timestamp[us, tz=UTC]",
            "amount_cents": "int64",
            "loaded_at": "timestamp[us, tz=UTC]",
        },
    },
    "marketplace": {
        "dim_buyer": {
            "buyer_id": "int64",
            "created_at": "timestamp[us, tz=UTC]",
            "country": "string",
        },
        "dim_seller": {
            "seller_id": "int64",
            "created_at": "timestamp[us, tz=UTC]",
            "country": "string",
            "shop_status": "string",
        },
        "dim_listing": {
            "listing_id": "int64",
            "seller_id": "int64",
            "category_id": "int64",
            "created_at": "timestamp[us, tz=UTC]",
            "status": "string",
        },
        "fact_order": {
            "order_id": "int64",
            "buyer_id": "int64",
            "order_time": "timestamp[us, tz=UTC]",
            "subtotal_cents": "int64",
            "tax_cents": "int64",
            "shipping_cents": "int64",
            "discount_cents": "int64",
            "loaded_at": "timestamp[us, tz=UTC]",
        },
        "fact_order_item": {
            "order_id": "int64",
            "line_id": "int32",
            "listing_id": "int64",
            "seller_id": "int64",
            "qty": "int32",
            "item_price_cents": "int64",
            "loaded_at": "timestamp[us, tz=UTC]",
        },
        "fact_payment": {
            "order_id": "int64",
            "captured_at": "timestamp[us, tz=UTC]",
            "buyer_paid_cents": "int64",
            "seller_earnings_cents": "int64",
            "platform_fee_cents": "int64",
            "loaded_at": "timestamp[us, tz=UTC]",
        },
        "snapshot_listing_daily": {
            "ds": "date32[day]",
            "listing_id": "int64",
            "status": "string",
        },
    },
}

WAREHOUSE_ARCHETYPE_HINTS: Dict[str, List[str]] = {
    "neobank": ["fact_card_transaction.parquet", "fact_subscription_invoice.parquet"],
    "marketplace": ["fact_order.parquet", "fact_payment.parquet"],
}

WAREHOUSE_VOLUME_TARGETS: Dict[str, Dict[str, int]] = {
    "neobank": {
        "dim_customer": 3500,
        "dim_account": 3500,
        "dim_card": 3500,
        "dim_merchant": 400,
        "dim_plan": 4,
        "fact_card_transaction": 120_000,
        "fact_subscription_invoice": 32_000,
    },
    "marketplace": {
        "dim_buyer": 4500,
        "dim_seller": 2200,
        "dim_listing": 9000,
        "fact_order": 52_000,
        "fact_order_item": 160_000,
        "fact_payment": 52_000,
        "snapshot_listing_daily": 270_000,
    },
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
            row_count = int(df.height) if df is not None else 0
            metrics.update({"table": table_name, "row_count": row_count})
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
    "validate_warehouse_schema",
    "validate_volume_targets",
    "validate_quality_targets",
    "validate_seasonality_targets",
    "validate_taxonomy_targets",
    "validate_monetization_targets",
    "validate_trajectory_targets",
    "validate_comms_targets",
    "validate_theme_mix_targets",
    "validate_event_correlation",
    "validate_reproducibility",
    "validate_spend_caps",
]
