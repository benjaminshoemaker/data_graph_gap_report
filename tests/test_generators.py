from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_needs_reporter.cli import app
from data_needs_reporter.config import DEFAULT_CONFIG_PATH, load_config
from data_needs_reporter.generate.comms import COMM_USER_ROLE_MIX, write_empty_comms
from data_needs_reporter.generate.defects import apply_typical_neobank_defects
from data_needs_reporter.generate.warehouse import (
    MARKETPLACE_TABLE_SCHEMAS,
    NEOBANK_MCC_TO_SECTOR,
    NEOBANK_TABLE_SCHEMAS,
    generate_marketplace_dims,
    generate_marketplace_facts,
    generate_neobank_dims,
    generate_neobank_facts,
    write_empty_warehouse,
)

pl = pytest.importorskip("polars")

runner = CliRunner()


def test_write_empty_neobank_warehouse(tmp_path: Path) -> None:
    write_empty_warehouse("neobank", tmp_path)

    for table, schema in NEOBANK_TABLE_SCHEMAS.items():
        table_path = tmp_path / f"{table}.parquet"
        assert table_path.is_file()
        df = pl.read_parquet(table_path)
        expected_columns = [name for name, _ in schema]
        assert df.columns == expected_columns
        for column_name, dtype in schema:
            assert df.schema[column_name] == dtype


def test_write_empty_marketplace_warehouse(tmp_path: Path) -> None:
    write_empty_warehouse("marketplace", tmp_path)

    for table, schema in MARKETPLACE_TABLE_SCHEMAS.items():
        table_path = tmp_path / f"{table}.parquet"
        assert table_path.is_file()
        df = pl.read_parquet(table_path)
        expected_columns = [name for name, _ in schema]
        assert df.columns == expected_columns
        for column_name, dtype in schema:
            assert df.schema[column_name] == dtype


def test_cli_gen_warehouse_dry_run(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        out_dir = Path("warehouse_out")
        result = runner.invoke(
            app,
            [
                "gen-warehouse",
                "--archetype",
                "neobank",
                "--out",
                str(out_dir),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (out_dir / "dim_customer.parquet").exists()


def test_generate_neobank_dims_writes_consistent_tables(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})
    counts = generate_neobank_dims(cfg, tmp_path, seed=123)

    customers = pl.read_parquet(tmp_path / "dim_customer.parquet")
    accounts = pl.read_parquet(tmp_path / "dim_account.parquet")
    cards = pl.read_parquet(tmp_path / "dim_card.parquet")
    merchants = pl.read_parquet(tmp_path / "dim_merchant.parquet")

    assert customers.height == counts["dim_customer"]
    assert customers["customer_id"].is_unique().all()

    assert accounts["account_id"].is_unique().all()
    assert accounts["customer_id"].is_in(customers["customer_id"]).all()

    assert cards["account_id"].is_in(accounts["account_id"]).all()
    assert cards["card_id"].is_unique().all()

    sectors = {NEOBANK_MCC_TO_SECTOR[int(mcc)] for mcc in merchants["mcc"].unique()}
    assert len(sectors) >= 8


def test_generate_neobank_facts_baseline(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})
    generate_neobank_dims(cfg, tmp_path, seed=123)
    fact_counts = generate_neobank_facts(cfg, tmp_path, tmp_path, seed=456)

    assert fact_counts["fact_card_transaction"] > 0
    assert fact_counts["fact_subscription_invoice"] > 0

    transactions = pl.read_parquet(tmp_path / "fact_card_transaction.parquet")
    invoices = pl.read_parquet(tmp_path / "fact_subscription_invoice.parquet")
    customers = pl.read_parquet(tmp_path / "dim_customer.parquet")

    base_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    max_event = transactions["event_time"].max()
    min_event = transactions["event_time"].min()
    assert min_event >= base_start
    assert max_event <= base_start + timedelta(days=cfg.warehouse.months * 30)

    captured_ratio = (
        transactions["auth_result"] == "captured"
    ).sum() / transactions.height
    assert 0.955 <= captured_ratio <= 0.975

    subscriber_ids = invoices["customer_id"].unique()
    attach_rate = len(subscriber_ids) / customers.height
    assert 0.06 <= attach_rate <= 0.10


def test_generate_marketplace_baseline(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})
    generate_marketplace_dims(cfg, tmp_path, seed=321)
    generate_marketplace_facts(cfg, tmp_path, tmp_path, seed=654)

    orders = pl.read_parquet(tmp_path / "fact_order.parquet")
    order_items = pl.read_parquet(tmp_path / "fact_order_item.parquet")
    payments = pl.read_parquet(tmp_path / "fact_payment.parquet")

    items_per_order = (
        order_items.groupby("order_id").count().select(pl.col("count")).mean()[0]
    )
    assert 1.2 <= items_per_order <= 2.4

    order_counts = (
        orders.with_columns(pl.col("order_time").dt.weekday().alias("dow"))
        .groupby("dow")
        .count()
        .rename({"count": "orders"})
    )
    weekday_mean = order_counts.filter(pl.col("dow") < 5)["orders"].mean()
    weekend_mean = order_counts.filter(pl.col("dow") >= 5)["orders"].mean()
    weekend_factor = weekend_mean / weekday_mean
    assert 1.15 <= weekend_factor <= 1.45

    joined = payments.join(orders.select(["order_id", "subtotal_cents"]), on="order_id")
    take_rate = (
        joined.with_columns(
            (pl.col("platform_fee_cents") / pl.col("subtotal_cents")).alias("take_rate")
        )
        .filter(pl.col("subtotal_cents") > 0)
        .select(pl.col("take_rate"))
        .mean()[0]
    )
    assert 0.11 <= take_rate <= 0.13


def test_neobank_defects_targets(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})
    generate_neobank_dims(cfg, tmp_path, seed=222)
    generate_neobank_facts(cfg, tmp_path, tmp_path, seed=333)
    metrics = apply_typical_neobank_defects(cfg, tmp_path, seed=444)

    transactions = pl.read_parquet(tmp_path / "fact_card_transaction.parquet")
    dim_card = pl.read_parquet(tmp_path / "dim_card.parquet")

    key_null_rate = (
        transactions["merchant_id"].is_null().sum() / transactions.height
        if transactions.height
        else 0
    )
    assert 0.015 <= key_null_rate <= 0.03

    fk_failures = transactions.filter(
        (pl.col("card_id").is_null()) | ~pl.col("card_id").is_in(dim_card["card_id"])
    )
    fk_rate = fk_failures.height / transactions.height if transactions.height else 0
    assert 0.04 <= fk_rate <= 0.07

    lag_minutes = [
        ((loaded - event).total_seconds() / 60)
        for loaded, event in zip(
            transactions["loaded_at"].to_list(), transactions["event_time"].to_list()
        )
    ]
    lag_minutes.sort()
    if lag_minutes:
        p95_index = int(0.95 * (len(lag_minutes) - 1))
        p95 = lag_minutes[p95_index]
        assert 120 <= p95 <= 240

    if metrics.get("spike_days"):
        assert all(day["null_rate"] > 0.1 for day in metrics["spike_days"])


def test_write_empty_comms_schemas(tmp_path: Path) -> None:
    write_empty_comms(tmp_path)

    slack = pl.read_parquet(tmp_path / "slack_messages.parquet")
    email = pl.read_parquet(tmp_path / "email_messages.parquet")
    nlq = pl.read_parquet(tmp_path / "nlq.parquet")
    users = pl.read_parquet(tmp_path / "comms_users.parquet")

    assert slack.columns == [
        "message_id",
        "thread_id",
        "user_id",
        "sent_at",
        "channel",
        "bucket",
        "body",
        "tokens",
        "loaded_at",
    ]
    assert email.columns == [
        "message_id",
        "thread_id",
        "sender_id",
        "recipient_ids",
        "subject",
        "body",
        "sent_at",
        "bucket",
        "tokens",
        "loaded_at",
    ]
    assert nlq.columns == [
        "query_id",
        "user_id",
        "submitted_at",
        "text",
        "parsed_intent",
        "tokens",
        "loaded_at",
    ]
    assert users.columns == [
        "user_id",
        "role",
        "department",
        "time_zone",
        "active",
    ]
    assert slack.height == email.height == nlq.height == users.height == 0


def test_comm_user_role_mix_sums_to_one() -> None:
    assert pytest.approx(sum(COMM_USER_ROLE_MIX.values()), rel=1e-6) == 1.0
