import importlib.util
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import polars as pl
import pytest
from typer.testing import CliRunner

from data_needs_reporter.config import DEFAULT_CONFIG_PATH, load_config
from data_needs_reporter.generate.defects import apply_typical_neobank_defects
from data_needs_reporter.generate.warehouse import (
    generate_neobank_dims,
    generate_neobank_facts,
)
from data_needs_reporter.report.metrics import (
    compute_data_health,
    validate_monetization_targets,
    validate_taxonomy_targets,
    validate_theme_mix_targets,
)
from data_needs_reporter.report.scoring import (
    compute_confidence,
    compute_marketplace_revenue_risk,
    compute_neobank_revenue_risk,
    compute_source_demand_weights,
    compute_score,
    compute_severity,
    compute_weighted_theme_shares,
    normalize_revenue,
    post_stratified_item_weights,
    select_top_actions,
    recency_decay,
    reweight_source_weights,
    trailing_monthly_revenue_median,
)
from data_needs_reporter.cli import app

_helpers_spec = importlib.util.spec_from_file_location(
    "_test_cli_helpers", Path(__file__).resolve().parent / "test_cli.py"
)
assert _helpers_spec and _helpers_spec.loader
_helpers = importlib.util.module_from_spec(_helpers_spec)
_helpers_spec.loader.exec_module(_helpers)
_write_minimal_neobank_warehouse = _helpers._write_minimal_neobank_warehouse
_write_minimal_comms = _helpers._write_minimal_comms


def _write_marketplace_taxonomy_fixture(base_path: Path, gmv_values: list[int]) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    tz = timezone.utc

    category_ids = list(range(1, len(gmv_values) + 1))
    categories = pl.DataFrame(
        {
            "category_id": category_ids,
            "parent_id": [None] * len(category_ids),
        }
    )
    categories.write_parquet(base_path / "dim_category.parquet")

    listings = pl.DataFrame(
        {
            "listing_id": category_ids,
            "seller_id": [1] * len(category_ids),
            "category_id": category_ids,
            "created_at": [datetime(2024, 1, 1, tzinfo=tz)] * len(category_ids),
            "status": ["active"] * len(category_ids),
        }
    )
    listings.write_parquet(base_path / "dim_listing.parquet")

    order_ids = list(range(1, len(gmv_values) + 1))
    loaded_times = [
        datetime(2024, 1, 2 + idx, tzinfo=tz) for idx in range(len(gmv_values))
    ]
    order_items = pl.DataFrame(
        {
            "order_id": order_ids,
            "line_id": [1] * len(order_ids),
            "listing_id": order_ids,
            "seller_id": [1] * len(order_ids),
            "qty": [1] * len(order_ids),
            "item_price_cents": gmv_values,
            "loaded_at": loaded_times,
        }
    )
    order_items.write_parquet(base_path / "fact_order_item.parquet")

    orders = pl.DataFrame(
        {
            "order_id": order_ids,
            "buyer_id": [1] * len(order_ids),
            "order_time": loaded_times,
            "subtotal_cents": gmv_values,
            "tax_cents": [0] * len(order_ids),
            "shipping_cents": [0] * len(order_ids),
            "discount_cents": [0] * len(order_ids),
            "loaded_at": loaded_times,
        }
    )
    orders.write_parquet(base_path / "fact_order.parquet")

    payments = pl.DataFrame(
        {
            "order_id": order_ids,
            "captured_at": loaded_times,
            "buyer_paid_cents": gmv_values,
            "seller_earnings_cents": gmv_values,
            "platform_fee_cents": [int(val * 0.12) for val in gmv_values],
            "loaded_at": loaded_times,
        }
    )
    payments.write_parquet(base_path / "fact_payment.parquet")


def _write_neobank_monetization_fixture(
    base_path: Path,
    *,
    interchange_bps: float = 150.0,
    attach_ratio: float = 0.08,
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    tz = timezone.utc

    customers = pl.DataFrame(
        {
            "customer_id": list(range(1, 101)),
            "created_at": [datetime(2023, 1, 1, tzinfo=tz)] * 100,
            "kyc_status": ["approved"] * 100,
        }
    )
    customers.write_parquet(base_path / "dim_customer.parquet")

    transaction_times = [datetime(2024, 1, 1 + idx, tzinfo=tz) for idx in range(30)]
    transactions = pl.DataFrame(
        {
            "txn_id": list(range(1, 31)),
            "card_id": [1] * 30,
            "merchant_id": [1] * 30,
            "event_time": transaction_times,
            "amount_cents": [20_000] * 30,
            "interchange_bps": [interchange_bps] * 30,
            "channel": ["card_present"] * 30,
            "auth_result": ["captured"] * 30,
            "loaded_at": transaction_times,
        }
    )
    transactions.write_parquet(base_path / "fact_card_transaction.parquet")

    attach_count = max(1, int(100 * attach_ratio))
    attach_customers = list(range(1, attach_count + 1))
    invoices = pl.DataFrame(
        {
            "invoice_id": list(range(1, attach_count + 1)),
            "customer_id": attach_customers,
            "plan_id": [1] * attach_count,
            "period_start": [datetime(2023, 12, 1, tzinfo=tz)] * attach_count,
            "period_end": [datetime(2023, 12, 31, tzinfo=tz)] * attach_count,
            "paid_at": [datetime(2024, 1, 5, tzinfo=tz)] * attach_count,
            "amount_cents": [999] * attach_count,
            "loaded_at": [datetime(2024, 1, 5, tzinfo=tz)] * attach_count,
        }
    )
    invoices.write_parquet(base_path / "fact_subscription_invoice.parquet")

    merchants = pl.DataFrame(
        {
            "merchant_id": [1, 2, 3, 4, 5],
            "mcc": [5001, 5002, 5003, 5004, 5005],
            "name": [
                "Merchant A",
                "Merchant B",
                "Merchant C",
                "Merchant D",
                "Merchant E",
            ],
        }
    )
    merchants.write_parquet(base_path / "dim_merchant.parquet")


def test_normalize_revenue():
    values = [1000, 1100, 900, 950]
    current = 1200
    score = normalize_revenue(values, current)
    assert 1.0 <= score <= 1.5


def test_reweight_source_weights():
    base = {"nlq": 0.5, "slack": 0.3, "email": 0.2}
    volumes = {"nlq": 50, "slack": 30, "email": 20}
    caps = {"min": 0.15, "max": 0.60}
    weights = reweight_source_weights(base, volumes, caps)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    for w in weights.values():
        assert 0.15 <= w <= 0.60


def test_reweight_source_weights_clamps_extremes():
    base = {"nlq": 0.5, "slack": 0.3, "email": 0.2}
    volumes = {"nlq": 500, "slack": 200, "email": 1}
    caps = {"min": 0.15, "max": 0.60}

    weights = reweight_source_weights(base, volumes, caps)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["nlq"] == pytest.approx(0.60, abs=1e-6)
    assert weights["slack"] == pytest.approx(0.25, abs=1e-4)
    assert weights["email"] == pytest.approx(0.15, abs=1e-4)


def test_compute_source_demand_weights_scales_with_volume():
    base = {"nlq": 0.5, "slack": 0.3, "email": 0.2}
    volumes = {"nlq": 900, "slack": 80, "email": 20}
    weights = compute_source_demand_weights(base, volumes, min_weight=0.15, max_weight=0.60)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["nlq"] == pytest.approx(0.60, abs=1e-6)
    for source_weight in weights.values():
        assert 0.15 - 1e-6 <= source_weight <= 0.60 + 1e-6


def test_compute_source_demand_weights_handles_zero_volume():
    base = {"nlq": 0.4, "slack": 0.4, "email": 0.2}
    volumes = {"nlq": 0, "slack": 0, "email": 0}
    weights = compute_source_demand_weights(base, volumes, min_weight=0.1, max_weight=0.7)
    assert weights["nlq"] == pytest.approx(0.4, abs=1e-6)
    assert weights["slack"] == pytest.approx(0.4, abs=1e-6)
    assert weights["email"] == pytest.approx(0.2, abs=1e-6)


def test_compute_severity():
    over = {"key_null_pct": 0.03, "fk_orphan_pct": 0.07}
    slos = {"key_null_pct": 0.02, "fk_orphan_pct": 0.05}
    severity = compute_severity(over, slos)
    assert severity > 0.0


def test_recency_decay():
    assert recency_decay(0) == pytest.approx(1.0)
    assert recency_decay(60) == pytest.approx(math.exp(-1))


def test_trailing_monthly_revenue_median():
    tz = timezone.utc
    monthly_totals = {
        datetime(2023, 10, 1, tzinfo=tz): 100.0,
        datetime(2023, 11, 1, tzinfo=tz): 200.0,
        datetime(2023, 12, 1, tzinfo=tz): 150.0,
        datetime(2024, 1, 1, tzinfo=tz): 400.0,
    }

    median_three = trailing_monthly_revenue_median(monthly_totals, window_months=3)
    assert median_three == pytest.approx(200.0)

    median_all = trailing_monthly_revenue_median(monthly_totals, window_months=6)
    assert median_all == pytest.approx(175.0)


def test_compute_score():
    score = compute_score(0.8, 0.6, 0.4)
    assert score == pytest.approx(0.60 * 0.8 + 0.25 * 0.6 + 0.15 * 0.4)


def test_select_top_actions_diversity():
    items = [
        {"theme": "retention", "score": 0.95, "confidence": 0.8},
        {"theme": "retention", "score": 0.93, "confidence": 0.8},
        {"theme": "growth", "score": 0.9, "confidence": 0.8},
        {"theme": "acquisition", "score": 0.88, "confidence": 0.8},
        {"theme": "quality", "score": 0.87, "confidence": 0.8},
    ]
    selected = select_top_actions(items)
    themes = [item["theme"] for item in selected]
    assert len(selected) == 3
    assert themes == ["retention", "growth", "acquisition"]
    assert len(set(themes)) == len(themes)


def test_select_top_actions_fill_when_insufficient_unique():
    items = [
        {"theme": "retention", "score": 0.9, "confidence": 0.8},
        {"theme": "retention", "score": 0.88, "confidence": 0.8},
        {"theme": "growth", "score": 0.87, "confidence": 0.8},
        {"theme": "growth", "score": 0.6, "confidence": 0.8},
    ]
    selected = select_top_actions(items)
    themes = [item["theme"] for item in selected]
    assert len(selected) == 3
    assert set(themes) == {"retention", "growth"}
    eligible = {
        str(item["theme"]) for item in items if item.get("confidence", 0.0) >= 0.55
    }
    assert len(set(themes)) == min(len(eligible), 3)


def test_compute_confidence_bounds():
    conf = compute_confidence(1.2, -0.1, 0.5, 0.5)
    assert 0.0 <= conf <= 1.0


def test_post_stratified_theme_shares() -> None:
    items = [
        {
            "source": "slack",
            "day": "2024-01-01",
            "bucket": "data_quality",
            "theme": "data_quality",
        },
        {
            "source": "slack",
            "day": "2024-01-01",
            "bucket": "data_quality",
            "theme": "governance",
        },
        {
            "source": "email",
            "day": "2024-01-01",
            "bucket": "governance",
            "theme": "governance",
        },
    ]
    strata_totals = {
        ("slack", "2024-01-01", "data_quality"): 100,
        ("email", "2024-01-01", "governance"): 50,
    }
    source_weights = {"slack": 0.6, "email": 0.4}

    weights = post_stratified_item_weights(items, strata_totals)
    assert weights == pytest.approx([50.0, 50.0, 50.0])

    shares = compute_weighted_theme_shares(
        items, strata_totals, source_weights=source_weights
    )
    assert shares["data_quality"] == pytest.approx(0.3)
    assert shares["governance"] == pytest.approx(0.7)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_neobank_revenue_risk() -> None:
    tz = timezone.utc
    transactions = [
        {
            "txn_id": 1,
            "event_time": datetime(2024, 1, 15, tzinfo=tz),
            "amount_cents": 10_000,
            "interchange_bps": 150,
        },
        {
            "txn_id": 2,
            "event_time": datetime(2024, 1, 20, tzinfo=tz),
            "amount_cents": 20_000,
            "interchange_bps": 120,
        },
        {
            "txn_id": 6,
            "event_time": datetime(2024, 1, 25, tzinfo=tz),
            "amount_cents": 5_000,
            "interchange_bps": 100,
        },
        {
            "txn_id": 3,
            "event_time": datetime(2023, 12, 10, tzinfo=tz),
            "amount_cents": 30_000,
            "interchange_bps": 140,
        },
        {
            "txn_id": 4,
            "event_time": datetime(2023, 11, 5, tzinfo=tz),
            "amount_cents": 15_000,
            "interchange_bps": 160,
        },
        {
            "txn_id": 5,
            "event_time": datetime(2023, 10, 8, tzinfo=tz),
            "amount_cents": 18_000,
            "interchange_bps": 130,
        },
    ]

    invoices = [
        {
            "invoice_id": 10,
            "period_start": datetime(2024, 1, 1, tzinfo=tz),
            "amount_cents": 1_299,
        },
        {
            "invoice_id": 14,
            "period_start": datetime(2024, 1, 15, tzinfo=tz),
            "amount_cents": 1_299,
        },
        {
            "invoice_id": 11,
            "period_start": datetime(2023, 12, 1, tzinfo=tz),
            "amount_cents": 1_299,
        },
        {
            "invoice_id": 12,
            "period_start": datetime(2023, 11, 1, tzinfo=tz),
            "amount_cents": 1_299,
        },
        {
            "invoice_id": 13,
            "period_start": datetime(2023, 10, 1, tzinfo=tz),
            "amount_cents": 1_299,
        },
    ]

    result = compute_neobank_revenue_risk(
        transactions,
        invoices,
        affected_transaction_ids=[1, 2],
        affected_invoice_ids=[10],
    )

    assert result["interchange_at_risk_usd"] == pytest.approx(3.9, rel=1e-3)
    assert result["subscription_at_risk_usd"] == pytest.approx(12.99, rel=1e-3)
    assert result["revenue_at_risk_usd"] == pytest.approx(16.89, rel=1e-3)
    assert result["median_monthly_revenue_usd"] == pytest.approx(17.19, rel=1e-3)
    assert result["revenue_risk_ratio"] == pytest.approx(16.89 / 17.19, rel=1e-3)


def test_marketplace_revenue_risk() -> None:
    tz = timezone.utc
    orders = [
        {"order_id": 1, "order_time": datetime(2023, 10, 5, tzinfo=tz)},
        {"order_id": 2, "order_time": datetime(2023, 11, 7, tzinfo=tz)},
        {"order_id": 3, "order_time": datetime(2023, 12, 12, tzinfo=tz)},
        {"order_id": 4, "order_time": datetime(2024, 1, 9, tzinfo=tz)},
    ]
    payments = [
        {
            "order_id": 1,
            "captured_at": datetime(2023, 10, 5, tzinfo=tz),
            "buyer_paid_cents": 100_000,
            "platform_fee_cents": 10_000,
        },
        {
            "order_id": 2,
            "captured_at": datetime(2023, 11, 7, tzinfo=tz),
            "buyer_paid_cents": 150_000,
            "platform_fee_cents": 15_000,
        },
        {
            "order_id": 3,
            "captured_at": datetime(2023, 12, 12, tzinfo=tz),
            "buyer_paid_cents": 120_000,
            "platform_fee_cents": 12_000,
        },
        {
            "order_id": 4,
            "captured_at": datetime(2024, 1, 9, tzinfo=tz),
            "buyer_paid_cents": 130_000,
            "platform_fee_cents": 13_000,
        },
    ]

    result = compute_marketplace_revenue_risk(
        orders, payments, affected_order_ids=[3, 4], take_rate=0.10
    )

    assert result["take_rate_used"] == pytest.approx(0.10, rel=1e-6)
    assert result["gmv_at_risk_usd"] == pytest.approx(2500.0, rel=1e-6)
    assert result["net_revenue_at_risk_usd"] == pytest.approx(250.0, rel=1e-6)
    assert result["median_monthly_revenue_usd"] == pytest.approx(130.0, rel=1e-6)
    assert result["revenue_risk_ratio"] == pytest.approx(1.0, rel=1e-6)



def test_compute_data_health_synthetic(tmp_path: Path) -> None:
    polars = pytest.importorskip("polars")
    tz = "UTC"
    day0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    day1 = datetime(2024, 1, 2, tzinfo=timezone.utc)

    polars.DataFrame(
        {
            "customer_id": [1, 2, None],
            "created_at": [day0, day1, day1],
        }
    ).write_parquet(tmp_path / "dim_customer.parquet")

    polars.DataFrame(
        {
            "merchant_id": [100, 200],
            "created_at": [day0, day0],
        }
    ).write_parquet(tmp_path / "dim_merchant.parquet")

    polars.DataFrame({"card_id": [10, 20]}).write_parquet(
        tmp_path / "dim_card.parquet"
    )

    polars.DataFrame(
        {
            "txn_id": [1, 2, None, 2],
            "card_id": [10, 20, 30, None],
            "merchant_id": [100, 200, 999, 100],
            "event_time": [day0, day1, day1, day1 + timedelta(hours=1)],
            "loaded_at": [
                day0 + timedelta(minutes=30),
                day1 + timedelta(hours=3),
                day1 + timedelta(hours=3, minutes=20),
                day1 + timedelta(hours=1, minutes=30),
            ],
        }
    ).write_parquet(tmp_path / "fact_card_transaction.parquet")

    polars.DataFrame(
        {
            "invoice_id": [1, 2],
            "customer_id": [1, 3],
            "invoice_date": [day0, day1],
            "loaded_at": [day0 + timedelta(hours=1), day1 + timedelta(hours=2)],
        }
    ).write_parquet(tmp_path / "fact_subscription_invoice.parquet")

    payload = compute_data_health(tmp_path, tz)
    assert "tables" in payload
    assert "aggregates" in payload

    tables = payload["tables"]
    assert isinstance(tables, Mapping)

    txn_metrics = tables["fact_card_transaction"]
    invoices_metrics = tables["fact_subscription_invoice"]

    assert pytest.approx(txn_metrics["key_null_pct"], rel=1e-6) == 25.0
    assert pytest.approx(txn_metrics["dup_key_pct"], rel=1e-6) == 25.0
    assert pytest.approx(txn_metrics["fk_success_pct"], rel=1e-6) == pytest.approx(
        5 / 7 * 100, rel=1e-6
    )
    assert pytest.approx(txn_metrics["orphan_pct"], rel=1e-6) == pytest.approx(
        2 / 7 * 100, rel=1e-6
    )
    assert txn_metrics["key_null_pct_daily"]
    assert txn_metrics["fk_success_pct_daily"]
    assert txn_metrics["key_null_spikes"]

    assert pytest.approx(invoices_metrics["fk_success_pct"], rel=1e-6) == 50.0
    assert pytest.approx(invoices_metrics["orphan_pct"], rel=1e-6) == 50.0

    aggregates = payload["aggregates"]
    total_rows = sum(int(m.get("row_count", 0) or 0) for m in tables.values())
    key_null_rows = sum(
        (m.get("row_count", 0) or 0) * (m.get("key_null_pct", 0.0) or 0.0) / 100.0
        for m in tables.values()
    )
    dup_rows = sum(
        (m.get("row_count", 0) or 0) * (m.get("dup_key_pct", 0.0) or 0.0) / 100.0
        for m in tables.values()
    )
    expected_key_null_pct = (
        (key_null_rows / total_rows) * 100.0 if total_rows else 0.0
    )
    expected_dup_pct = (dup_rows / total_rows) * 100.0 if total_rows else 0.0

    assert aggregates["key_null_pct"] == pytest.approx(expected_key_null_pct, rel=1e-6)
    assert aggregates["dup_key_pct"] == pytest.approx(expected_dup_pct, rel=1e-6)
    assert pytest.approx(aggregates["fk_success_pct"], rel=1e-6) == pytest.approx(
        (6 / 9) * 100, rel=1e-6
    )
    assert pytest.approx(aggregates["orphan_pct"], rel=1e-6) == pytest.approx(
        (3 / 9) * 100, rel=1e-6
    )
    assert aggregates["p95_ingest_lag_min"] == pytest.approx(200.0)


def test_validate_enforces_slos(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        comms.mkdir()
        _write_minimal_comms(comms)

        report_dir = Path("reports") / "neobank"
        payload = json.loads((report_dir / "data_health.json").read_text(encoding="utf-8"))
        aggregates = payload.setdefault("aggregates", {})
        aggregates.update(
            {
                "key_null_pct": 5.0,
                "orphan_pct": 4.0,
                "dup_key_pct": 0.8,
                "p95_ingest_lag_min": 150.0,
            }
        )
        tables = payload.setdefault("tables", {})
        if isinstance(tables, list):
            invoice_entry = next(
                (
                    entry
                    for entry in tables
                    if isinstance(entry, dict)
                    and entry.get("table") == "fact_subscription_invoice"
                ),
                None,
            )
            if invoice_entry is None:
                invoice_entry = {"table": "fact_subscription_invoice"}
                tables.append(invoice_entry)
        elif isinstance(tables, dict):
            invoice_entry = tables.setdefault("fact_subscription_invoice", {})
        else:
            invoice_entry = {}
            payload["tables"] = {"fact_subscription_invoice": invoice_entry}
        invoice_entry.update(
            {
                "row_count": 10,
                "key_null_pct": 6.0,
                "orphan_pct": 5.0,
                "dup_key_pct": 1.2,
                "p95_ingest_lag_min": 160.0,
            }
        )
        (report_dir / "data_health.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        out_dir = report_dir / "qc"
        result = runner.invoke(
            app,
            [
                "validate",
                "--warehouse",
                str(warehouse),
                "--comms",
                str(comms),
                "--out",
                str(out_dir),
                "--strict",
            ],
        )
        assert result.exit_code == 1
        summary = json.loads((out_dir / "qc_summary.json").read_text(encoding="utf-8"))
        assert summary["passed"] is False
        assert any("Aggregate key_null_pct" in issue for issue in summary["issues"])
        assert any(
            "fact_subscription_invoice key_null_pct" in issue
            for issue in summary["issues"]
        )

        csv_lines = (out_dir / "qc_checks.csv").read_text(encoding="utf-8").splitlines()
        assert any("slo.aggregate.key_null_pct" in line for line in csv_lines)
        assert any(
            "slo.fact_subscription_invoice.key_null_pct" in line for line in csv_lines
        )


def test_validate_taxonomy_targets_marketplace_pass(tmp_path: Path) -> None:
    gmv = [120] * 10
    warehouse = tmp_path / "marketplace_ok"
    _write_marketplace_taxonomy_fixture(warehouse, gmv)
    result = validate_taxonomy_targets(warehouse)
    assert result["passed"] is True


def test_validate_taxonomy_targets_marketplace_top_share(tmp_path: Path) -> None:
    gmv = [600] + [40] * 9
    warehouse = tmp_path / "marketplace_heavy"
    _write_marketplace_taxonomy_fixture(warehouse, gmv)
    result = validate_taxonomy_targets(warehouse)
    assert result["passed"] is False
    assert "Top category GMV share" in result["detail"]


def test_validate_taxonomy_targets_marketplace_missing_category(tmp_path: Path) -> None:
    gmv = [120] * 9 + [0]
    warehouse = tmp_path / "marketplace_missing"
    _write_marketplace_taxonomy_fixture(warehouse, gmv)
    result = validate_taxonomy_targets(warehouse)
    assert result["passed"] is False
    assert "Top-level categories with zero GMV" in result["detail"]


def test_validate_monetization_targets_neobank_pass(tmp_path: Path) -> None:
    warehouse = tmp_path / "neobank_ok"
    _write_neobank_monetization_fixture(warehouse)
    result = validate_monetization_targets(warehouse)
    assert result["passed"] is True


def test_validate_monetization_targets_neobank_interchange_fail(tmp_path: Path) -> None:
    warehouse = tmp_path / "neobank_bad_interchange"
    _write_neobank_monetization_fixture(warehouse, interchange_bps=50.0)
    result = validate_monetization_targets(warehouse)
    assert result["passed"] is False
    assert "Interchange rate" in result["detail"]


def test_validate_monetization_targets_neobank_attach_fail(tmp_path: Path) -> None:
    warehouse = tmp_path / "neobank_bad_attach"
    _write_neobank_monetization_fixture(warehouse, attach_ratio=0.02)
    result = validate_monetization_targets(warehouse)
    assert result["passed"] is False
    assert "attach rate" in result["detail"]


def test_validate_monetization_targets_marketplace_pass(tmp_path: Path) -> None:
    gmv = [200] * 10
    warehouse = tmp_path / "marketplace_monetization_ok"
    _write_marketplace_taxonomy_fixture(warehouse, gmv)
    result = validate_monetization_targets(warehouse)
    assert result["passed"] is True


def test_validate_monetization_targets_marketplace_fail(tmp_path: Path) -> None:
    gmv = [200] * 10
    warehouse = tmp_path / "marketplace_monetization_bad"
    _write_marketplace_taxonomy_fixture(warehouse, gmv)
    payments = pl.read_parquet(warehouse / "fact_payment.parquet")
    payments = payments.with_columns(
        (pl.col("platform_fee_cents") * 2).alias("platform_fee_cents")
    )
    payments.write_parquet(warehouse / "fact_payment.parquet")
    result = validate_monetization_targets(warehouse)
    assert result["passed"] is False
    assert "take rate" in result["detail"]


def test_validate_theme_mix_targets_pass(tmp_path: Path) -> None:
    comms = tmp_path / "comms_ok"
    comms.mkdir()
    budget = {
        "quotas": {
            "slack": {"bucket_totals": {"data_quality": 100, "governance": 100}},
            "email": {"bucket_totals": {"data_quality": 50, "governance": 50}},
        },
        "coverage": {
            "slack": {
                "per_bucket": {
                    "data_quality": {"actual": 90},
                    "governance": {"actual": 110},
                }
            },
            "email": {
                "per_bucket": {
                    "data_quality": {"actual": 45},
                    "governance": {"actual": 55},
                }
            },
        },
    }
    (comms / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    result = validate_theme_mix_targets(comms)
    assert result["passed"] is True


def test_validate_theme_mix_targets_fail(tmp_path: Path) -> None:
    comms = tmp_path / "comms_bad"
    comms.mkdir()
    budget = {
        "quotas": {
            "slack": {"bucket_totals": {"data_quality": 190, "governance": 10}},
        },
        "coverage": {
            "slack": {
                "per_bucket": {
                    "data_quality": {"actual": 100},
                    "governance": {"actual": 100},
                }
            }
        },
    }
    (comms / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    result = validate_theme_mix_targets(comms)
    assert result["passed"] is False
    assert "Theme 'data_quality'" in result["detail"]
