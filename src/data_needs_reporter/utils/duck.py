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
    records = result.fetchall()
    columns = [desc[0] for desc in result.description] if result.description else None
    if not records:
        return pl.DataFrame(schema=columns or [])
    if columns is None:
        return pl.DataFrame(records)
    return pl.DataFrame(records, schema=columns)


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
    """Compute archetype sanity metrics using in-memory Polars transforms."""
    arch_key = archetype.lower()
    base_path = Path(parquet_dir)
    if arch_key == "neobank":
        customers = pl.read_parquet(base_path / "dim_customer.parquet")
        cards = pl.read_parquet(base_path / "dim_card.parquet")
        transactions = pl.read_parquet(base_path / "fact_card_transaction.parquet")

        customer_counts = pl.DataFrame({"customers": [customers.height]})
        txn_count = transactions.height
        total_amount = float(transactions["amount_cents"].sum()) if txn_count else 0.0
        transaction_volume = pl.DataFrame(
            {"txn_count": [txn_count], "total_amount": [total_amount]}
        )

        if txn_count:
            min_event = transactions["event_time"].min()
            max_event = transactions["event_time"].max()
        else:
            min_event = None
            max_event = None
        transaction_bounds = pl.DataFrame(
            {"min_event": [min_event], "max_event": [max_event]}
        )

        joined = transactions.join(
            cards.select("card_id"), on="card_id", how="inner"
        ).height
        txn_join_quality = pl.DataFrame({"joined": [joined]})

        daily_kpi = (
            transactions.with_columns(
                [
                    pl.col("event_time").dt.truncate("1d").alias("ds"),
                    pl.col("auth_result")
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .alias("_auth"),
                    (
                        pl.col("amount_cents").cast(pl.Float64)
                        * pl.col("interchange_bps").cast(pl.Float64)
                        / 10000.0
                        / 100.0
                    ).alias("_interchange_usd"),
                    (pl.col("amount_cents").cast(pl.Float64) / 100.0).alias(
                        "_amount_usd"
                    ),
                ]
            )
            .group_by("ds")
            .agg(
                [
                    pl.when(pl.col("_auth") == "captured")
                    .then(pl.col("_amount_usd"))
                    .otherwise(0.0)
                    .sum()
                    .alias("captured_usd"),
                    pl.when(pl.col("_auth") == "captured")
                    .then(pl.col("_interchange_usd"))
                    .otherwise(0.0)
                    .sum()
                    .alias("interchange_usd"),
                ]
            )
            .select(["ds", "captured_usd", "interchange_usd"])
            .sort("ds")
        )

        return {
            "customer_counts": customer_counts,
            "transaction_volume": transaction_volume,
            "transaction_bounds": transaction_bounds,
            "txn_join_quality": txn_join_quality,
            "daily_kpi": daily_kpi,
        }

    if arch_key == "marketplace":
        orders = pl.read_parquet(base_path / "fact_order.parquet")
        payments = pl.read_parquet(base_path / "fact_payment.parquet")
        order_items = pl.read_parquet(base_path / "fact_order_item.parquet")

        order_counts = pl.DataFrame({"orders": [orders.height]})

        payment_join = orders.join(
            payments.select(["order_id"]),
            on="order_id",
            how="inner",
        ).height
        order_payment_join = pl.DataFrame({"joined": [payment_join]})

        items_count = order_items.height
        total_qty = int(order_items["qty"].sum()) if items_count else 0
        order_item_totals = pl.DataFrame(
            {"items": [items_count], "total_qty": [total_qty]}
        )

        if orders.height:
            min_order = orders["order_time"].min()
            max_order = orders["order_time"].max()
        else:
            min_order = None
            max_order = None
        order_bounds = pl.DataFrame({"min_order": [min_order], "max_order": [max_order]})

        orders_with_fee = orders.join(
            payments.select(["order_id", "platform_fee_cents"]),
            on="order_id",
            how="left",
        ).with_columns(
            [
                pl.col("order_time").dt.truncate("1d").alias("ds"),
                (
                    pl.col("subtotal_cents")
                    + pl.col("shipping_cents")
                    + pl.col("tax_cents")
                    - pl.col("discount_cents")
                ).alias("_gmv_cents"),
            ]
        )

        daily_agg = (
            orders_with_fee.group_by("ds")
            .agg(
                [
                    pl.sum("_gmv_cents").alias("_gmv_cents"),
                    pl.sum("platform_fee_cents").alias("_fee_cents"),
                    pl.sum("subtotal_cents").alias("_subtotal_cents"),
                ]
            )
            .with_columns(
                [
                    (pl.col("_gmv_cents") / 100.0).alias("gmv_usd"),
                    pl.when(pl.col("_subtotal_cents") == 0)
                    .then(None)
                    .otherwise(pl.col("_fee_cents") / pl.col("_subtotal_cents"))
                    .alias("take_rate"),
                ]
            )
            .select(["ds", "gmv_usd", "take_rate"])
            .sort("ds")
        )

        return {
            "order_counts": order_counts,
            "order_payment_join": order_payment_join,
            "order_item_totals": order_item_totals,
            "order_bounds": order_bounds,
            "daily_kpi": daily_agg,
        }

    raise ValueError(f"Unsupported archetype for sanity checks: {archetype}")


__all__ = [
    "attach_parquet_dir",
    "open_db",
    "run_warehouse_sanity",
    "safe_query",
]
