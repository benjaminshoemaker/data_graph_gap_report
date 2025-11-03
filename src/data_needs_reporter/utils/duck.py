from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import duckdb
import polars as pl

_IDENTIFIER_CLEAN_RE = re.compile(r"[^A-Za-z0-9_]")


def open_db(db_path_or_none: Optional[Path]) -> duckdb.DuckDBPyConnection:
    if db_path_or_none is None:
        return duckdb.connect(database=":memory:")
    Path(db_path_or_none).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=str(db_path_or_none))


def attach_parquet_dir(
    conn: duckdb.DuckDBPyConnection, name: str, dir_path: Path
) -> None:
    directory = Path(dir_path)
    if not directory.exists():
        raise FileNotFoundError(f"Parquet directory not found: {dir_path}")

    parquet_files = sorted(directory.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dir_path}")

    for file_path in parquet_files:
        view_name = _make_identifier(f"{name}_{file_path.stem}")
        file_literal = str(file_path).replace("'", "''")
        conn.execute(
            f"CREATE VIEW IF NOT EXISTS {view_name} AS SELECT * FROM read_parquet('{file_literal}')"
        )


def safe_query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: Optional[Iterable] = None,
) -> pl.DataFrame:
    _guard_sql(sql)
    result = conn.execute(sql, params or ())
    return result.pl()


def _guard_sql(sql: str) -> None:
    lowered = sql.lower()
    if ";" in sql:
        raise ValueError("Semicolons are not permitted in safe_query.")
    forbidden = ("copy", "attach", "install")
    if any(keyword in lowered for keyword in forbidden):
        raise ValueError("Potentially unsafe SQL detected.")


def _make_identifier(raw: str) -> str:
    if not raw:
        raise ValueError("Identifier cannot be empty.")
    sanitized = _IDENTIFIER_CLEAN_RE.sub("_", raw)
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


_SANITY_SQL: Dict[str, Dict[str, str]] = {
    "neobank": {
        "customer_counts": "SELECT COUNT(*) AS customers FROM warehouse_dim_customer",
        "transaction_volume": (
            "SELECT COUNT(*) AS txn_count, SUM(amount_cents) AS total_amount "
            "FROM warehouse_fact_card_transaction"
        ),
        "transaction_bounds": (
            "SELECT MIN(event_time) AS min_event, MAX(event_time) AS max_event "
            "FROM warehouse_fact_card_transaction"
        ),
        "txn_join_quality": (
            "SELECT COUNT(*) AS joined "
            "FROM warehouse_fact_card_transaction t "
            "JOIN warehouse_dim_card c ON t.card_id = c.card_id"
        ),
        "daily_kpi": (
            "SELECT "
            "  date_trunc('day', event_time) AS ds, "
            "  SUM(CASE WHEN auth_result = 'captured' THEN amount_cents ELSE 0 END) / 100.0 AS captured_usd, "
            "  SUM(CASE WHEN auth_result = 'captured' THEN amount_cents * interchange_bps / 10000 ELSE 0 END) / 100.0 AS interchange_usd "
            "FROM warehouse_fact_card_transaction "
            "GROUP BY 1 "
            "ORDER BY 1"
        ),
    },
    "marketplace": {
        "order_counts": "SELECT COUNT(*) AS orders FROM warehouse_fact_order",
        "order_payment_join": (
            "SELECT COUNT(*) AS joined "
            "FROM warehouse_fact_payment p "
            "JOIN warehouse_fact_order o ON p.order_id = o.order_id"
        ),
        "order_item_totals": (
            "SELECT COUNT(*) AS items, SUM(qty) AS total_qty "
            "FROM warehouse_fact_order_item"
        ),
        "order_bounds": (
            "SELECT MIN(order_time) AS min_order, MAX(order_time) AS max_order "
            "FROM warehouse_fact_order"
        ),
        "daily_kpi": (
            "SELECT "
            "  date_trunc('day', o.order_time) AS ds, "
            "  SUM(o.subtotal_cents + o.shipping_cents + o.tax_cents - o.discount_cents) / 100.0 AS gmv_usd, "
            "  CASE "
            "    WHEN SUM(o.subtotal_cents) = 0 THEN NULL "
            "    ELSE SUM(p.platform_fee_cents) * 1.0 / SUM(o.subtotal_cents) "
            "  END AS take_rate "
            "FROM warehouse_fact_order o "
            "JOIN warehouse_fact_payment p ON o.order_id = p.order_id "
            "GROUP BY 1 "
            "ORDER BY 1"
        ),
    },
}


def run_warehouse_sanity(archetype: str, parquet_dir: Path) -> Dict[str, pl.DataFrame]:
    """Attach generated parquet tables and run archetype-specific sanity SQL."""
    arch_key = archetype.lower()
    if arch_key not in _SANITY_SQL:
        raise ValueError(f"Unsupported archetype for sanity checks: {archetype}")

    conn = open_db(None)
    attach_parquet_dir(conn, "warehouse", Path(parquet_dir))

    results: Dict[str, pl.DataFrame] = {}
    for name, sql in _SANITY_SQL[arch_key].items():
        results[name] = safe_query(conn, sql)
    return results


__all__ = [
    "attach_parquet_dir",
    "open_db",
    "run_warehouse_sanity",
    "safe_query",
]
