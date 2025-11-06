import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from data_needs_reporter.cli import app
from data_needs_reporter.generate.warehouse import write_warehouse_hash_manifest
from data_needs_reporter.utils.hashing import compute_file_hash

runner = CliRunner()


def _write_minimal_neobank_warehouse(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tables = {
        "dim_customer": {
            "customer_id": "int64",
            "created_at": "timestamp[us, tz=UTC]",
            "kyc_status": "string",
            "rows": 3500,
        },
        "dim_account": {
            "account_id": "int64",
            "customer_id": "int64",
            "type": "string",
            "created_at": "timestamp[us, tz=UTC]",
            "status": "string",
            "rows": 3500,
        },
        "dim_card": {
            "card_id": "int64",
            "account_id": "int64",
            "status": "string",
            "activated_at": "timestamp[us, tz=UTC]",
            "rows": 3500,
        },
        "dim_merchant": {
            "merchant_id": "int64",
            "mcc": "int32",
            "name": "string",
            "rows": 400,
        },
        "dim_plan": {
            "plan_id": "int64",
            "name": "string",
            "price_cents": "int64",
            "cadence": "string",
            "rows": 4,
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
            "rows": 120_000,
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
            "rows": 32_000,
        },
    }
    manifest = {"tables": {}}
    data_backed = {"fact_card_transaction", "dim_customer", "fact_subscription_invoice"}
    for table_name, columns in tables.items():
        if table_name not in data_backed:
            (path / f"{table_name}.parquet").write_text("placeholder", encoding="utf-8")
        column_def = {}
        row_count = None
        for name, dtype in columns.items():
            if name == "rows":
                row_count = dtype
            else:
                column_def[name] = {"type": dtype}
        entry = {"columns": column_def}
        if row_count is not None:
            entry["rows"] = row_count
        manifest["tables"][table_name] = entry
    (path / "schema.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    base_weekday = datetime(2024, 1, 1, tzinfo=timezone.utc)  # Monday
    weekend_day = base_weekday + timedelta(days=5)  # Saturday
    records = []
    txn_id = 1
    for day, total in [(base_weekday, 10), (weekend_day, 12)]:
        per_hour = max(total // 2, 1)
        for hour in (11, 18):
            for _ in range(per_hour):
                event_time = day + timedelta(hours=hour)
                records.append(
                    {
                        "txn_id": txn_id,
                        "card_id": 1,
                        "merchant_id": 1 if hour == 11 else 2,
                        "event_time": event_time,
                        "amount_cents": 20_000,
                        "interchange_bps": 150.0,
                        "channel": "card_present",
                        "auth_result": "captured",
                        "loaded_at": event_time,
                    }
                )
                txn_id += 1
    pl.DataFrame(records).write_parquet(path / "fact_card_transaction.parquet")
    summary = {
        "aggregates": {
            "merchant_key_null_pct": 0.02,
            "card_fk_orphan_pct": 0.04,
            "txn_dup_key_pct": 0.007,
            "txn_p95_ingest_lag_min": 180.0,
        }
    }
    (path / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    merchants = [
        {"merchant_id": mid, "mcc": 5000 + mid, "name": f"Merchant {mid}"}
        for mid in range(1, 51)
    ]
    pl.DataFrame(merchants).write_parquet(path / "dim_merchant.parquet")
    month_counts = [9, 10, 11, 12, 13, 14, 15, 16]
    status_sequence = ["verified"] * 70 + ["pending"] * 20 + ["refused"] * 10

    def _month_offset(start: datetime, offset: int) -> datetime:
        month = start.month - 1 + offset
        year = start.year + month // 12
        month = month % 12 + 1
        return datetime(year, month, 1, tzinfo=start.tzinfo)

    customers = []
    cid = 1
    for idx, count in enumerate(month_counts):
        month_start = _month_offset(base_weekday - timedelta(days=30), idx)
        for i in range(count):
            created_at = month_start + timedelta(days=min(i, 27), hours=i % 12)
            customers.append(
                {
                    "customer_id": cid,
                    "created_at": created_at,
                    "kyc_status": status_sequence[cid - 1],
                }
            )
            cid += 1
    pl.DataFrame(customers).write_parquet(path / "dim_customer.parquet")

    invoice_customers = list(range(1, 9))  # 8% attach
    months = [_month_offset(base_weekday - timedelta(days=60), idx) for idx in range(8)]
    invoices = []
    invoice_id = 1
    for idx, start in enumerate(months):
        period_end = start + timedelta(days=30)
        active = invoice_customers[:-1] if idx == len(months) - 1 else invoice_customers
        for cid in active:
            paid_at = period_end + timedelta(days=2, hours=idx)
            invoices.append(
                {
                    "invoice_id": invoice_id,
                    "customer_id": cid,
                    "plan_id": 1,
                    "period_start": start,
                    "period_end": period_end,
                    "paid_at": paid_at,
                    "amount_cents": 999,
                    "loaded_at": paid_at,
                }
            )
            invoice_id += 1
    pl.DataFrame(invoices).write_parquet(path / "fact_subscription_invoice.parquet")

    write_warehouse_hash_manifest(path, seed=42)


def _write_minimal_comms(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tz = timezone.utc

    buckets = ["data_quality", "governance", "analytics", "pipeline_health"]
    bucket_count = len(buckets)

    slack_records = []
    threads = 1000
    exec_threads = int(threads * 0.08)
    message_id = 1
    for thread_id in range(1, threads + 1):
        bucket = buckets[(thread_id - 1) % bucket_count]
        is_exec_thread = thread_id <= exec_threads
        for msg_idx in range(3):
            slack_records.append(
                {
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "bucket": bucket,
                    "is_exec": bool(is_exec_thread and msg_idx == 0),
                    "ts": datetime(2024, 1, 1, tzinfo=tz)
                    + timedelta(minutes=thread_id + msg_idx),
                }
            )
            message_id += 1
    pl.DataFrame(slack_records).write_parquet(path / "slack_messages.parquet")

    email_records = []
    for idx in range(800):
        email_records.append(
            {
                "message_id": idx + 1,
                "bucket": buckets[idx % bucket_count],
                "sent_at": datetime(2024, 1, 1, tzinfo=tz)
                + timedelta(days=idx // 20, hours=idx % 6),
            }
        )
    pl.DataFrame(email_records).write_parquet(path / "email_messages.parquet")

    nlq_records = []
    for idx in range(1000):
        nlq_records.append(
            {
                "query_id": idx + 1,
                "bucket": buckets[idx % bucket_count],
                "tokens": 20,
                "created_at": datetime(2024, 1, 1, tzinfo=tz)
                + timedelta(hours=idx // 10),
            }
        )
    pl.DataFrame(nlq_records).write_parquet(path / "nlq.parquet")

    users = pl.DataFrame(
        [
            {
                "user_id": 1,
                "role": "executive",
                "department": "ops",
                "time_zone": "UTC",
                "active": True,
            }
        ]
    )
    users.write_parquet(path / "comms_users.parquet")

    plan_slack = {bucket: 750 for bucket in buckets}
    plan_email = {bucket: 200 for bucket in buckets}
    plan_nlq = {bucket: 250 for bucket in buckets}

    coverage = {
        "slack": {
            "overall": {
                "actual": 3000,
                "target": 3000,
                "coverage_pct": 1.0,
                "met_floor": True,
                "behavior": "continue",
            },
            "per_bucket": {
                bucket: {
                    "actual": plan_slack[bucket],
                    "target": plan_slack[bucket],
                    "coverage_pct": 1.0,
                }
                for bucket in plan_slack
            },
        },
        "email": {
            "overall": {
                "actual": 800,
                "target": 800,
                "coverage_pct": 1.0,
                "met_floor": True,
                "behavior": "continue",
            },
            "per_bucket": {
                bucket: {
                    "actual": plan_email[bucket],
                    "target": plan_email[bucket],
                    "coverage_pct": 1.0,
                }
                for bucket in plan_email
            },
        },
        "nlq": {
            "overall": {
                "actual": 1000,
                "target": 1000,
                "coverage_pct": 1.0,
                "met_floor": True,
                "behavior": "continue",
            },
            "per_bucket": {
                bucket: {
                    "actual": plan_nlq[bucket],
                    "target": plan_nlq[bucket],
                    "coverage_pct": 1.0,
                }
                for bucket in plan_nlq
            },
        },
    }

    quotas = {
        "slack": {
            "total": 3000,
            "day_bucket": {},
            "bucket_totals": plan_slack,
        },
        "email": {
            "total": 800,
            "day_bucket": {},
            "bucket_totals": plan_email,
        },
        "nlq": {
            "total": 1000,
            "day_bucket": {},
            "bucket_totals": plan_nlq,
        },
    }

    budget = {
        "messages": {
            "slack": 3000,
            "email": 800,
            "nlq": 1000,
        },
        "coverage": coverage,
        "quotas": quotas,
        "hashes": {
            "algorithm": "sha256",
            "files": {
                "slack_messages.parquet": compute_file_hash(
                    path / "slack_messages.parquet"
                ),
                "email_messages.parquet": compute_file_hash(
                    path / "email_messages.parquet"
                ),
                "nlq.parquet": compute_file_hash(path / "nlq.parquet"),
                "comms_users.parquet": compute_file_hash(path / "comms_users.parquet"),
            },
        },
        "seeds": {"comms": 43, "warehouse": 42},
        "cap_usd": 1.0,
        "cost_usd": 0.0,
        "price_per_1k_tokens": 0.002,
        "tokens_used": 0,
        "token_budget": 500000,
        "stopped_due_to_cap": False,
    }

    (path / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")

    parse_summary = {
        "mode": "execute",
        "parse_only": False,
        "queries_total": len(nlq_records),
        "queries_parsed": len(nlq_records),
        "parse_success_pct": 1.0,
    }
    (path / "nlq_parse_summary.json").write_text(
        json.dumps(parse_summary, indent=2), encoding="utf-8"
    )


def test_version_flag_reports_package_version() -> None:
    from data_needs_reporter import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_root_command_displays_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_run_report_generates_outputs(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        Path("warehouse").mkdir()
        Path("comms").mkdir()
        out_dir = Path("reports") / "neobank"
        result = runner.invoke(
            app,
            [
                "run-report",
                "--warehouse",
                "warehouse",
                "--comms",
                "comms",
                "--out",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.stdout

        data_health_json = out_dir / "data_health.json"
        data_health_csv = out_dir / "data_health.csv"
        themes_json = out_dir / "themes.json"
        themes_md = out_dir / "themes.md"
        exec_json = out_dir / "exec_summary.json"
        exec_md = out_dir / "exec_summary.md"
        budget_json = out_dir / "budget.json"
        figures_dir = out_dir / "figures"

        for path in [
            data_health_json,
            data_health_csv,
            themes_json,
            themes_md,
            exec_json,
            exec_md,
            budget_json,
        ]:
            assert path.exists()

        figures = [
            figures_dir / "lag_p95_daily.png",
            figures_dir / "key_null_pct_daily.png",
            figures_dir / "orphan_pct_daily.png",
            figures_dir / "dup_key_pct_bar.png",
            figures_dir / "theme_demand_monthly.png",
        ]
        for fig in figures:
            assert fig.exists()

        payload = json.loads(data_health_json.read_text(encoding="utf-8"))
        assert "tables" in payload
        assert isinstance(payload["tables"], list)
        assert payload["tables"]
        first_table = payload["tables"][0]
        for key in [
            "table",
            "key_null_pct",
            "fk_orphan_pct",
            "dup_keys_pct",
            "p95_ingest_lag_min",
        ]:
            assert key in first_table

        themes_payload = json.loads(themes_json.read_text(encoding="utf-8"))
        assert "themes" in themes_payload
        assert len(themes_payload["themes"]) >= 1

        exec_payload = json.loads(exec_json.read_text(encoding="utf-8"))
        assert "top_actions" in exec_payload
        assert exec_payload["top_actions"]

        budget_payload = json.loads(budget_json.read_text(encoding="utf-8"))
        assert "total_cost_usd" in budget_payload
        assert "warehouse_path" in budget_payload

        csv_lines = data_health_csv.read_text(encoding="utf-8").splitlines()
        assert csv_lines[0].startswith("table,")


def test_run_report_warns_on_coverage_breach() -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        warehouse.mkdir()
        comms.mkdir()
        _write_minimal_comms(comms)
        budget_path = comms / "budget.json"
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        slack_overall = (
            budget.setdefault("coverage", {})
            .setdefault("slack", {})
            .setdefault("overall", {})
        )
        slack_overall["coverage_pct"] = 0.1
        slack_overall["met_floor"] = False
        budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        out_dir = Path("reports") / "neobank"
        result = runner.invoke(
            app,
            [
                "run-report",
                "--warehouse",
                str(warehouse),
                "--comms",
                str(comms),
                "--out",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Coverage below 20%" in result.stdout


def test_run_report_strict_fails_on_coverage_breach() -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        warehouse.mkdir()
        comms.mkdir()
        _write_minimal_comms(comms)
        budget_path = comms / "budget.json"
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        slack_overall = (
            budget.setdefault("coverage", {})
            .setdefault("slack", {})
            .setdefault("overall", {})
        )
        slack_overall["coverage_pct"] = 0.05
        slack_overall["met_floor"] = False
        budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        out_dir = Path("reports") / "neobank"
        result = runner.invoke(
            app,
            [
                "run-report",
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
        assert "Coverage below 20%" in result.stdout
        assert "Report written" not in result.stdout


def test_run_report_strict_fails_on_budget_cap() -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        warehouse.mkdir()
        comms.mkdir()
        _write_minimal_comms(comms)
        budget_path = comms / "budget.json"
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        budget["stopped_due_to_cap"] = True
        coverage = budget.setdefault("coverage", {})
        for source in ("slack", "email", "nlq"):
            overall = coverage.setdefault(source, {}).setdefault("overall", {})
            overall["coverage_pct"] = 1.0
            overall["met_floor"] = True
        budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        out_dir = Path("reports") / "neobank"
        result = runner.invoke(
            app,
            [
                "run-report",
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
        assert "spend cap" in result.stdout.lower()


def test_validate_strict_fails_on_marker(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        comms.mkdir()
        _write_minimal_comms(comms)
        (warehouse / "FAIL").write_text("fail", encoding="utf-8")
        out_dir = Path("reports") / "neobank" / "qc"
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
        assert summary["overall_pass"] is False


def test_validate_passes_without_marker(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
            ],
        )
        assert result.exit_code == 0
        summary = json.loads((out_dir / "qc_summary.json").read_text(encoding="utf-8"))
        assert summary["overall_pass"] is True
        quality_check = next(
            check for check in summary["checks"] if check["name"] == "quality"
        )
        assert quality_check["passed"] is True
        seasonality_check = next(
            check for check in summary["checks"] if check["name"] == "seasonality"
        )
        assert seasonality_check["passed"] is True
        taxonomy_check = next(
            check for check in summary["checks"] if check["name"] == "taxonomy"
        )
        assert taxonomy_check["passed"] is True
        monetization_check = next(
            check for check in summary["checks"] if check["name"] == "monetization"
        )
        assert monetization_check["passed"] is True
        comms_check = next(
            check for check in summary["checks"] if check["name"] == "comms"
        )
        assert comms_check["passed"] is True
        theme_mix_check = next(
            check for check in summary["checks"] if check["name"] == "theme_mix"
        )
        assert theme_mix_check["passed"] is True


def test_validate_strict_fails_on_missing_table(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        (warehouse / "dim_customer.parquet").unlink()
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
        assert summary["overall_pass"] is False
        schema_check = next(
            check for check in summary["checks"] if check["name"] == "schema"
        )
        assert schema_check["passed"] is False


def test_validate_strict_fails_on_volume(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        manifest_path = warehouse / "schema.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tables"]["fact_card_transaction"]["rows"] = 10  # far below target
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
        assert summary["overall_pass"] is False
        volume_check = next(
            check for check in summary["checks"] if check["name"] == "volume"
        )
        assert volume_check["passed"] is False


def test_validate_strict_fails_on_quality(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        summary_path = warehouse / "data_quality_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["aggregates"]["merchant_key_null_pct"] = 0.5
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        quality_check = next(
            check for check in summary_out["checks"] if check["name"] == "quality"
        )
        assert quality_check["passed"] is False


def test_validate_strict_fails_on_seasonality(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        base_day = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bad_records = []
        txn_id = 1
        for _ in range(12):
            event_time = base_day + timedelta(hours=8)
            bad_records.append(
                {
                    "txn_id": txn_id,
                    "card_id": 1,
                    "merchant_id": 1,
                    "event_time": event_time,
                    "amount_cents": 10_000,
                    "interchange_bps": 120.0,
                    "channel": "card_present",
                    "auth_result": "captured",
                    "loaded_at": event_time,
                }
            )
            txn_id += 1
        for _ in range(2):
            event_time = base_day + timedelta(days=6, hours=22)  # Sunday late night
            bad_records.append(
                {
                    "txn_id": txn_id,
                    "card_id": 1,
                    "merchant_id": 1,
                    "event_time": event_time,
                    "amount_cents": 10_000,
                    "interchange_bps": 120.0,
                    "channel": "card_present",
                    "auth_result": "captured",
                    "loaded_at": event_time,
                }
            )
            txn_id += 1
        pl.DataFrame(bad_records).write_parquet(
            warehouse / "fact_card_transaction.parquet"
        )
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        seasonality_check = next(
            check for check in summary_out["checks"] if check["name"] == "seasonality"
        )
        assert seasonality_check["passed"] is False


def test_validate_strict_fails_on_taxonomy(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        merchants = [
            {"merchant_id": mid, "mcc": 6000 + mid, "name": f"Merchant {mid}"}
            for mid in range(1, 6)
        ]
        pl.DataFrame(merchants).write_parquet(warehouse / "dim_merchant.parquet")
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        taxonomy_check = next(
            check for check in summary_out["checks"] if check["name"] == "taxonomy"
        )
        assert taxonomy_check["passed"] is False


def test_validate_strict_fails_on_monetization(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        transactions = pl.read_parquet(warehouse / "fact_card_transaction.parquet")
        bad_transactions = transactions.with_columns(
            pl.lit(500.0).alias("interchange_bps")
        )
        bad_transactions.write_parquet(warehouse / "fact_card_transaction.parquet")
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        monetization_check = next(
            check for check in summary_out["checks"] if check["name"] == "monetization"
        )
        assert monetization_check["passed"] is False


def test_validate_strict_fails_on_trajectory(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        customers = pl.read_parquet(warehouse / "dim_customer.parquet")
        cutoff = datetime(2024, 4, 1, tzinfo=timezone.utc)
        adjusted = []
        for row in customers.to_dicts():
            created = row["created_at"]
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created >= cutoff:
                created = created - timedelta(days=180)
            row["created_at"] = created
            adjusted.append(row)
        pl.DataFrame(adjusted).write_parquet(warehouse / "dim_customer.parquet")
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        trajectory_check = next(
            check for check in summary_out["checks"] if check["name"] == "trajectory"
        )
        assert trajectory_check["passed"] is False


def test_validate_strict_fails_on_theme_mix(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        comms.mkdir()
        _write_minimal_comms(comms)
        budget_path = comms / "budget.json"
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        slack_quotas = budget.get("quotas", {}).get("slack")
        if slack_quotas and isinstance(slack_quotas, dict):
            bucket_totals = slack_quotas.get("bucket_totals", {})
            bucket_totals["data_quality"] = bucket_totals.get("data_quality", 0) + 1200
        slack_coverage = budget.get("coverage", {}).get("slack")
        if slack_coverage and isinstance(slack_coverage, dict):
            per_bucket = slack_coverage.get("per_bucket", {})
            if "data_quality" in per_bucket:
                per_bucket["data_quality"]["target"] = (
                    per_bucket["data_quality"].get("target", 0) + 1200
                )
        budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        theme_mix_check = next(
            check for check in summary_out["checks"] if check["name"] == "theme_mix"
        )
        assert theme_mix_check["passed"] is False


def test_validate_strict_fails_on_event_correlation(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        summary_path = warehouse / "data_quality_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("per_table", {})
        summary["per_table"]["fact_card_transaction"] = {
            "spike_days": [{"event_day": "2024-01-04T00:00:00+00:00"}]
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        comms.mkdir()
        _write_minimal_comms(comms)
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        event_check = next(
            check
            for check in summary_out["checks"]
            if check["name"] == "event_correlation"
        )
        assert event_check["passed"] is False
        assert "24h" in event_check["detail"]


def test_validate_strict_fails_on_reproducibility(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        comms.mkdir()
        _write_minimal_comms(comms)
        budget_path = comms / "budget.json"
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        budget_hashes = budget.get("hashes", {}).get("files", {})
        key = next(iter(budget_hashes))
        budget_hashes[key] = "deadbeef"
        budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        repro_check = next(
            check
            for check in summary_out["checks"]
            if check["name"] == "reproducibility"
        )
        assert repro_check["passed"] is False


def test_validate_strict_fails_on_spend_caps(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        comms.mkdir()
        _write_minimal_comms(comms)
        budget_path = comms / "budget.json"
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        budget["cap_usd"] = 0.0001
        budget["tokens_used"] = 5000
        budget["cost_usd"] = 0.01
        budget["price_per_1k_tokens"] = 0.002
        budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        spend_check = next(
            check for check in summary_out["checks"] if check["name"] == "spend_caps"
        )
        assert spend_check["passed"] is False


def test_validate_strict_fails_on_nlq_tokens(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        comms.mkdir()
        _write_minimal_comms(comms)
        nlq_path = comms / "nlq.parquet"
        df = pl.read_parquet(nlq_path)
        df = df.with_columns(pl.lit(40).alias("tokens"))
        df.write_parquet(nlq_path)
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        comms_check = next(
            check for check in summary_out["checks"] if check["name"] == "comms"
        )
        assert comms_check["passed"] is False
        assert "NLQ" in comms_check["detail"]


def test_validate_strict_fails_on_nlq_parse_rate(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        _write_minimal_neobank_warehouse(warehouse)
        comms.mkdir()
        _write_minimal_comms(comms)
        parse_summary_path = comms / "nlq_parse_summary.json"
        parse_summary = {
            "parse_only": True,
            "queries_total": 100,
            "queries_parsed": 93,
        }
        parse_summary_path.write_text(
            json.dumps(parse_summary, indent=2), encoding="utf-8"
        )
        out_dir = Path("reports") / "neobank" / "qc"
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
        summary_out = json.loads(
            (out_dir / "qc_summary.json").read_text(encoding="utf-8")
        )
        assert summary_out["overall_pass"] is False
        comms_check = next(
            check for check in summary_out["checks"] if check["name"] == "comms"
        )
        assert comms_check["passed"] is False
        assert "parse" in comms_check["detail"].lower()


def test_quickstart_generates_reports(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["quickstart"])
        assert result.exit_code == 0, result.stdout
        index = Path("reports") / "index.md"
        assert index.exists()
        content = index.read_text(encoding="utf-8")
        assert "Neobank" in content
        assert "Marketplace" in content

        for archetype in ("neobank", "marketplace"):
            assert (Path("data") / archetype / "dim_customer.parquet").exists()
            assert (Path("comms") / archetype / "slack_messages.parquet").exists()
            report_dir = Path("reports") / archetype
            assert (report_dir / "exec_summary.json").exists()
            themes = json.loads(
                (report_dir / "themes.json").read_text(encoding="utf-8")
            )
            assert themes["themes"]


def test_quickstart_no_llm_outputs_minimal(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["quickstart", "--no-llm"])
        assert result.exit_code == 0
        report_dir = Path("reports") / "neobank"
        themes = json.loads((report_dir / "themes.json").read_text(encoding="utf-8"))
        assert themes["themes"] == []
        assert (report_dir / "exec_summary.md").exists()


def test_quickstart_fast_scales_volumes(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["quickstart", "--fast"])
        assert result.exit_code == 0
        slack = pl.read_parquet(Path("comms") / "neobank" / "slack_messages.parquet")
        email = pl.read_parquet(Path("comms") / "neobank" / "email_messages.parquet")
        nlq = pl.read_parquet(Path("comms") / "neobank" / "nlq.parquet")
        assert slack.height <= 400
        assert email.height <= 120
        assert nlq.height <= 150


def test_console_entrypoint_version(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "data_needs_reporter.cli", "--version"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
