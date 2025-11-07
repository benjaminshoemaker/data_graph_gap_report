from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data_needs_reporter.report.metrics import compute_data_health

pl = pytest.importorskip("polars")


def test_compute_data_health_tracks_daily_metrics(tmp_path):
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()

    base_day = datetime(2024, 1, 1, tzinfo=timezone.utc)
    days = [base_day + timedelta(days=offset) for offset in range(3)]

    # Seed dim_customer with three days of activity and a forced PK null on day 3.
    dim_rows = [
        {"customer_id": 1, "created_at": days[0]},
        {"customer_id": 2, "created_at": days[1]},
        {"customer_id": None, "created_at": days[2]},
    ]
    pl.DataFrame(dim_rows).write_parquet(warehouse_dir / "dim_customer.parquet")

    # Fact rows include matching event/load timestamps per day to exercise lag metrics.
    lag_minutes = [5.0, 10.0, 15.0]
    fact_rows = []
    for idx, (event_time, lag) in enumerate(zip(days, lag_minutes), start=1):
        fact_rows.append(
            {
                "txn_id": idx,
                "merchant_id": 1,
                "card_id": 1,
                "event_time": event_time,
                "loaded_at": event_time + timedelta(minutes=lag),
            }
        )
    pl.DataFrame(fact_rows).write_parquet(
        warehouse_dir / "fact_card_transaction.parquet"
    )

    metrics = compute_data_health(warehouse_dir, tz="UTC")
    tables = metrics["tables"]

    dim_metrics = tables["dim_customer"]
    key_null_daily = dim_metrics["key_null_pct_daily"]
    assert len(key_null_daily) == 3
    assert all("pct" in entry and "date" in entry for entry in key_null_daily)

    spikes = dim_metrics["key_null_spikes"]
    assert len(spikes) == 1
    assert spikes[0]["date"] == days[2].date().isoformat()

    fact_metrics = tables["fact_card_transaction"]
    lag_daily = fact_metrics["p95_ingest_lag_min_daily"]
    assert len(lag_daily) == 3
    assert [entry["date"] for entry in lag_daily] == [
        day.date().isoformat() for day in days
    ]
    assert [entry["minutes"] for entry in lag_daily] == pytest.approx(lag_minutes)
