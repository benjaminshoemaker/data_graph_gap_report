from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data_needs_reporter.config import AppConfig, DEFAULT_CONFIG_PATH, load_config
from data_needs_reporter.generate.warehouse import (
    generate_marketplace_dims,
    generate_marketplace_facts,
    generate_neobank_dims,
    generate_neobank_facts,
)
from data_needs_reporter.utils.duck import run_warehouse_sanity

pl = pytest.importorskip("polars")


def _load_config() -> AppConfig:
    return load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})


def test_neobank_sanity_queries(tmp_path: Path) -> None:
    cfg = _load_config()
    generate_neobank_dims(cfg, tmp_path, seed=100)
    generate_neobank_facts(cfg, tmp_path, tmp_path, seed=200)

    results = run_warehouse_sanity("neobank", tmp_path)

    customers = results["customer_counts"]["customers"][0]
    txns = results["transaction_volume"]["txn_count"][0]
    joined = results["txn_join_quality"]["joined"][0]
    bounds = results["transaction_bounds"].to_dict(False)
    min_event = bounds["min_event"][0]
    max_event = bounds["max_event"][0]
    daily_kpi = results["daily_kpi"]

    assert customers > 0
    assert txns > 0
    assert joined == txns
    assert daily_kpi.height > 0
    assert {"ds", "captured_usd", "interchange_usd"}.issubset(daily_kpi.columns)
    assert min_event is not None
    assert max_event is not None
    assert min_event <= max_event

    base_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    window_days = cfg.warehouse.months * 30
    expected_end = base_start + timedelta(days=window_days)

    assert base_start <= min_event <= expected_end
    assert base_start <= max_event <= expected_end

    transactions = pl.read_parquet(tmp_path / "fact_card_transaction.parquet")
    assert (transactions["loaded_at"] >= transactions["event_time"]).all()


def test_marketplace_sanity_queries(tmp_path: Path) -> None:
    cfg = _load_config()
    generate_marketplace_dims(cfg, tmp_path, seed=300)
    generate_marketplace_facts(cfg, tmp_path, tmp_path, seed=400)

    results = run_warehouse_sanity("marketplace", tmp_path)

    orders = results["order_counts"]["orders"][0]
    joined = results["order_payment_join"]["joined"][0]
    items = results["order_item_totals"]["items"][0]
    total_qty = results["order_item_totals"]["total_qty"][0]
    bounds = results["order_bounds"].to_dict(False)
    min_order = bounds["min_order"][0]
    max_order = bounds["max_order"][0]
    daily_kpi = results["daily_kpi"]

    assert orders > 0
    assert joined == orders
    assert items > 0
    assert total_qty >= items
    assert daily_kpi.height > 0
    assert {"ds", "gmv_usd", "take_rate"}.issubset(daily_kpi.columns)
    assert min_order is not None
    assert max_order is not None
    assert min_order <= max_order

    base_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    window_days = cfg.warehouse.months * 30
    expected_end = base_start + timedelta(days=window_days)

    assert base_start <= min_order <= expected_end
    assert base_start <= max_order <= expected_end

    orders_df = pl.read_parquet(tmp_path / "fact_order.parquet")
    payments_df = pl.read_parquet(tmp_path / "fact_payment.parquet")

    assert (orders_df["loaded_at"] >= orders_df["order_time"]).all()
    assert (payments_df["loaded_at"] >= payments_df["captured_at"]).all()
