from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_needs_reporter.cli import app

runner = CliRunner()


def _write_config(path: Path) -> None:
    config = {
        "paths": {
            "data": "data",
            "comms": "comms",
            "reports": "reports",
            "meta": "meta",
        },
        "warehouse": {
            "archetypes": ["neobank", "marketplace"],
            "scale": "evaluation",
            "quality": "typical",
            "trajectory": "T1",
            "tz": "America/Los_Angeles",
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
                "p95_ingest_lag_min": 90,
            },
            "window_days": 30,
            "demand_base_weights": {"nlq": 0.5, "slack": 0.3, "email": 0.2},
            "demand_weight_caps": {"min": 0.15, "max": 0.60},
            "revenue_norm": "median_3m",
        },
        "budget": {
            "mode": "sample_to_fit",
            "safety_margin": 0.25,
            "coverage_floor_pct": 20,
        },
        "cache": {"enabled": True, "dir": ".cache/llm"},
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("polars")

    with runner.isolated_filesystem():
        config_path = Path("test_config.json")
        _write_config(config_path)

        def invoke(args: list[str]):
            result = runner.invoke(app, ["--config", str(config_path), *args])
            assert result.exit_code == 0, result.stdout

        invoke(["init"])
        invoke(
            [
                "gen-warehouse",
                "--archetype",
                "neobank",
                "--out",
                "data/neobank",
                "--dry-run",
            ]
        )
        invoke(["gen-comms", "--archetype", "neobank", "--out", "comms/neobank"])
        invoke(
            [
                "run-report",
                "--warehouse",
                "data/neobank",
                "--comms",
                "comms/neobank",
                "--out",
                "reports/neobank",
            ]
        )
        invoke(
            [
                "validate",
                "--warehouse",
                "data/neobank",
                "--comms",
                "comms/neobank",
                "--out",
                "reports/neobank/qc",
                "--strict",
            ]
        )

        assert (Path("reports") / "neobank" / "exec_summary.json").exists()
