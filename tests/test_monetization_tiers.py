from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from data_needs_reporter.report.metrics import validate_monetization_targets

pytest.importorskip("polars")
import polars as pl  # noqa: E402


def _write_neobank_with_tiers(
    base_path: Path,
    *,
    customer_count: int = 40,
    tier_customers: dict[str, list[int]],
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    tz = timezone.utc
    base_time = datetime(2024, 1, 1, tzinfo=tz)

    customers = pl.DataFrame(
        {
            "customer_id": list(range(1, customer_count + 1)),
            "created_at": [base_time] * customer_count,
        }
    )
    customers.write_parquet(base_path / "dim_customer.parquet")

    transactions = pl.DataFrame(
        {
            "txn_id": [1, 2],
            "card_id": [1, 2],
            "merchant_id": [10, 20],
            "event_time": [base_time, base_time],
            "amount_cents": [50_000, 25_000],
            "interchange_bps": [150.0, 140.0],
            "channel": ["card_present", "card_present"],
            "auth_result": ["captured", "captured"],
            "loaded_at": [base_time, base_time],
        }
    )
    transactions.write_parquet(base_path / "fact_card_transaction.parquet")

    invoices = []
    invoice_id = 1
    for tier, customers_for_tier in tier_customers.items():
        for customer_id in customers_for_tier:
            invoices.append(
                {
                    "invoice_id": invoice_id,
                    "customer_id": customer_id,
                    "plan_id": 1,
                    "period_start": base_time,
                    "period_end": base_time,
                    "paid_at": base_time,
                    "amount_cents": 999,
                    "loaded_at": base_time,
                    "tier": tier,
                }
            )
            invoice_id += 1
    pl.DataFrame(invoices).write_parquet(
        base_path / "fact_subscription_invoice.parquet"
    )


def test_tier_attach_targets_pass(tmp_path: Path) -> None:
    warehouse = tmp_path / "tiers_pass"
    _write_neobank_with_tiers(
        warehouse,
        tier_customers={"Tier A": [1, 2], "Tier B": [3]},
    )
    result = validate_monetization_targets(
        warehouse, attach_targets={"Tier A": 0.04, "Tier B": 0.02}
    )
    assert result["passed"] is True
    assert "Tier A" in result["detail"]


def test_tier_attach_targets_fail(tmp_path: Path) -> None:
    warehouse = tmp_path / "tiers_fail"
    _write_neobank_with_tiers(
        warehouse,
        tier_customers={"Tier A": [1, 2], "Tier B": [3]},
    )
    result = validate_monetization_targets(warehouse, attach_targets={"Tier B": 0.10})
    assert result["passed"] is False
    assert any("Tier B" in issue for issue in result["issues"])
