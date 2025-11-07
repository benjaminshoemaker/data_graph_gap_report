from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("polars")
import polars as pl  # noqa: E402

from data_needs_reporter.report.metrics import (  # noqa: E402
    validate_marketplace_category_caps,
    validate_marketplace_evening_coverage,
)


def _write_marketplace_orders_for_coverage(
    base_path: Path, *, total_days: int, evening_days: set[int]
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    tz = timezone.utc
    order_rows = []
    payment_rows = []
    order_id = 1
    base_time = datetime(2024, 1, 1, tzinfo=tz)

    for day_idx in range(total_days):
        day_start = base_time + timedelta(days=day_idx)
        for hour in (10, 14):
            order_time = day_start + timedelta(hours=hour)
            order_rows.append(
                {
                    "order_id": order_id,
                    "buyer_id": 1,
                    "order_time": order_time,
                    "subtotal_cents": 10_000,
                    "tax_cents": 0,
                    "shipping_cents": 0,
                    "discount_cents": 0,
                    "loaded_at": order_time + timedelta(minutes=5),
                }
            )
            payment_rows.append(
                {
                    "order_id": order_id,
                    "captured_at": order_time + timedelta(minutes=15),
                    "buyer_paid_cents": 10_000,
                    "seller_earnings_cents": 9_000,
                    "platform_fee_cents": 1_000,
                    "loaded_at": order_time + timedelta(minutes=15),
                }
            )
            order_id += 1
        if day_idx in evening_days:
            for hour in (18, 19):
                order_time = day_start + timedelta(hours=hour)
                order_rows.append(
                    {
                        "order_id": order_id,
                        "buyer_id": 1,
                        "order_time": order_time,
                        "subtotal_cents": 10_000,
                        "tax_cents": 0,
                        "shipping_cents": 0,
                        "discount_cents": 0,
                        "loaded_at": order_time + timedelta(minutes=5),
                    }
                )
                payment_rows.append(
                    {
                        "order_id": order_id,
                        "captured_at": order_time + timedelta(minutes=15),
                        "buyer_paid_cents": 10_000,
                        "seller_earnings_cents": 9_000,
                        "platform_fee_cents": 1_000,
                        "loaded_at": order_time + timedelta(minutes=15),
                    }
                )
                order_id += 1

    pl.DataFrame(order_rows).write_parquet(base_path / "fact_order.parquet")
    pl.DataFrame(payment_rows).write_parquet(base_path / "fact_payment.parquet")


def _write_marketplace_category_fixture(
    base_path: Path,
    gmv_by_listing: dict[int, int],
    *,
    use_child_categories: bool = False,
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    tz = timezone.utc
    categories = pl.DataFrame(
        {
            "category_id": [1, 2, 3, 101, 102, 103],
            "name": [
                "Home",
                "Art",
                "Clothing",
                "Home Decor",
                "Art Prints",
                "Athleisure",
            ],
            "parent_id": [None, None, None, 1, 2, 3],
        }
    )
    categories.write_parquet(base_path / "dim_category.parquet")

    listing_categories = [101, 102, 103] if use_child_categories else [1, 2, 3]
    listings = pl.DataFrame(
        {
            "listing_id": [1, 2, 3],
            "seller_id": [1001, 1002, 1003],
            "category_id": listing_categories,
            "created_at": [
                datetime(2024, 1, 1, tzinfo=tz),
                datetime(2024, 1, 1, tzinfo=tz),
                datetime(2024, 1, 1, tzinfo=tz),
            ],
            "status": ["active", "active", "active"],
        }
    )
    listings.write_parquet(base_path / "dim_listing.parquet")

    order_item_rows = []
    for idx, (listing_id, price_cents) in enumerate(gmv_by_listing.items(), start=1):
        order_item_rows.append(
            {
                "order_id": idx,
                "line_id": 1,
                "listing_id": listing_id,
                "seller_id": listing_id + 2000,
                "qty": 1,
                "item_price_cents": price_cents,
                "loaded_at": datetime(2024, 1, 2, tzinfo=tz),
            }
        )
    pl.DataFrame(order_item_rows).write_parquet(base_path / "fact_order_item.parquet")


def test_evening_coverage_pass(tmp_path: Path) -> None:
    warehouse = tmp_path / "coverage_pass"
    _write_marketplace_orders_for_coverage(
        warehouse, total_days=10, evening_days=set(range(8))
    )
    result = validate_marketplace_evening_coverage(
        warehouse,
        tz="UTC",
        window_days=10,
        start_hour=17,
        end_hour=21,
        min_share_pct=20.0,
        min_days_pct=80.0,
    )
    assert result["passed"] is True
    assert "Evening coverage satisfied" in result["detail"]


def test_evening_coverage_fail(tmp_path: Path) -> None:
    warehouse = tmp_path / "coverage_fail"
    _write_marketplace_orders_for_coverage(
        warehouse, total_days=10, evening_days=set(range(4))
    )
    result = validate_marketplace_evening_coverage(
        warehouse,
        tz="UTC",
        window_days=10,
        start_hour=17,
        end_hour=21,
        min_share_pct=20.0,
        min_days_pct=80.0,
    )
    assert result["passed"] is False
    assert "Evening coverage shortfall" in result["detail"]


def test_category_caps_pass(tmp_path: Path) -> None:
    warehouse = tmp_path / "caps_pass"
    _write_marketplace_category_fixture(warehouse, {1: 40_000, 2: 35_000, 3: 25_000})
    result = validate_marketplace_category_caps(
        warehouse, category_caps={"Home": 0.5, "Art": 0.5}
    )
    assert result["passed"] is True
    assert "Category caps satisfied" in result["detail"]


def test_category_caps_fail(tmp_path: Path) -> None:
    warehouse = tmp_path / "caps_fail"
    _write_marketplace_category_fixture(warehouse, {1: 70_000, 2: 20_000, 3: 10_000})
    result = validate_marketplace_category_caps(warehouse, category_caps={"Home": 0.4})
    assert result["passed"] is False
    assert "exceeds cap" in result["detail"]


def test_category_caps_rollup_toggle(tmp_path: Path) -> None:
    warehouse = tmp_path / "caps_rollup"
    _write_marketplace_category_fixture(
        warehouse,
        {1: 90_000, 2: 5_000, 3: 5_000},
        use_child_categories=True,
    )
    no_rollup = validate_marketplace_category_caps(
        warehouse,
        category_caps={"Home Decor": 0.5},
        rollup_to_parent=False,
    )
    assert no_rollup["passed"] is False
    rollup = validate_marketplace_category_caps(
        warehouse,
        category_caps={"Home": 0.9},
        rollup_to_parent=True,
    )
    assert rollup["passed"] is True
