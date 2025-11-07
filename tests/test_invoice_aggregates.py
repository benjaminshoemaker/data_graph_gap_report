from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import importlib.util

import pytest
from typer.testing import CliRunner

from data_needs_reporter.cli import app

runner = CliRunner()

_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "_test_cli_helpers", Path(__file__).resolve().parent / "test_cli.py"
)
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
assert _HELPERS_SPEC and _HELPERS_SPEC.loader
_HELPERS_SPEC.loader.exec_module(_HELPERS)
_write_minimal_comms = _HELPERS._write_minimal_comms
_write_minimal_neobank_warehouse = _HELPERS._write_minimal_neobank_warehouse


def _write_invoice_config(path: Path) -> None:
    config = {
        "paths": {
            "data": "warehouse",
            "comms": "comms",
            "reports": "reports",
            "meta": "meta",
        },
        "warehouse": {
            "archetypes": ["neobank"],
            "scale": "evaluation",
            "quality": "typical",
            "trajectory": "T1",
            "tz": "UTC",
            "months": 1,
            "seed": 42,
        },
        "comms": {
            "slack_threads": 5,
            "email_threads": 3,
            "nlq": 2,
            "sample_policy": "event_aware",
            "seed": 43,
        },
        "classification": {
            "engine": "llm",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_output_tokens": 64,
            "concurrency": 2,
            "api_cap_usd": 0.25,
            "prefilter_threshold": 0.35,
            "env_key_var": "OPENAI_API_KEY",
        },
        "entities": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_cap_usd": 0.1,
        },
        "report": {
            "scoring_weights": {"revenue": 0.6, "demand": 0.25, "severity": 0.15},
            "slos": {
                "key_null_pct": 1.0,
                "fk_orphan_pct": 2.0,
                "dup_keys_pct": 0.2,
                "p95_ingest_lag_min": 120.0,
            },
            "window_days": 30,
            "demand_base_weights": {"nlq": 0.5, "slack": 0.3, "email": 0.2},
            "demand_weight_caps": {"min": 0.15, "max": 0.60},
            "revenue_norm": "median_3m",
            "invoice_aggregates": {
                "enabled": True,
                "slos": {
                    "key_null_pct": 1.0,
                    "dup_key_pct": 0.5,
                    "p95_ingest_lag_min": 150.0,
                },
            },
        },
        "budget": {
            "mode": "sample_to_fit",
            "safety_margin": 0.25,
            "coverage_floor_pct": 20,
        },
        "cache": {"enabled": False, "dir": ".cache/llm"},
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _write_invoice_warehouse(path: Path) -> None:
    polars = pytest.importorskip("polars")
    _write_minimal_neobank_warehouse(path)
    tz = timezone.utc
    base = datetime(2024, 1, 1, tzinfo=tz)

    polars.DataFrame(
        {
            "account_id": [10, 20, 30],
            "customer_id": [1, 2, 3],
            "type": ["checking", "savings", "credit"],
            "created_at": [base] * 3,
            "status": ["active", "active", "active"],
        }
    ).write_parquet(path / "dim_account.parquet")

    polars.DataFrame(
        {
            "card_id": [100, 200],
            "account_id": [10, 20],
            "status": ["active", "active"],
            "activated_at": [base + timedelta(days=1), base + timedelta(days=2)],
        }
    ).write_parquet(path / "dim_card.parquet")

    polars.DataFrame(
        {
            "merchant_id": [500, 600],
            "mcc": [5411, 5734],
            "name": ["Fresh Market", "Data Tools"],
            "created_at": [base, base + timedelta(days=1)],
        }
    ).write_parquet(path / "dim_merchant.parquet")

    polars.DataFrame(
        {
            "plan_id": [1, 2],
            "name": ["Basic", "Pro"],
            "price_cents": [0, 999],
            "cadence": ["monthly", "monthly"],
        }
    ).write_parquet(path / "dim_plan.parquet")


def test_invoice_aggregates_checks(tmp_path: Path) -> None:
    pytest.importorskip("polars")

    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        reports_dir = Path("reports") / "neobank"
        (Path("meta")).mkdir(parents=True, exist_ok=True)

        _write_invoice_warehouse(warehouse)
        _write_minimal_comms(comms)

        config_path = Path("test_config.json")
        _write_invoice_config(config_path)

        run_report = runner.invoke(
            app,
            [
                "--config",
                str(config_path),
                "run-report",
                "--warehouse",
                str(warehouse),
                "--comms",
                str(comms),
                "--out",
                str(reports_dir),
            ],
        )
        assert run_report.exit_code == 0, run_report.stdout

        payload = json.loads(
            (reports_dir / "data_health.json").read_text(encoding="utf-8")
        )
        agg_tables = payload.get("aggregates_by_table", {})
        assert isinstance(agg_tables, dict)
        assert "fact_subscription_invoice" in agg_tables

        qc_dir = reports_dir / "qc"
        validate_run = runner.invoke(
            app,
            [
                "--config",
                str(config_path),
                "validate",
                "--warehouse",
                str(warehouse),
                "--comms",
                str(comms),
                "--out",
                str(qc_dir),
            ],
        )
        assert validate_run.exit_code == 0, validate_run.stdout

        summary = json.loads((qc_dir / "qc_summary.json").read_text(encoding="utf-8"))
        check_names = [
            check.get("name")
            for check in summary.get("checks", [])
            if isinstance(check, Mapping)
        ]
        invoice_checks = [
            name
            for name in check_names
            if isinstance(name, str) and name.startswith("slo.invoice_aggregates.")
        ]
        assert invoice_checks, "Expected invoice aggregate SLO checks to be present."
