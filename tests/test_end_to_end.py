from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import pytest
from typer.testing import CliRunner

from data_needs_reporter.cli import app

runner = CliRunner()


def _normalize_data_health_tables(
    tables: object,
) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    items: list[tuple[object, object]] = []
    if isinstance(tables, Mapping):
        items = list(tables.items())
    elif isinstance(tables, list):
        for entry in tables:
            if isinstance(entry, Mapping) and "table" in entry:
                items.append((entry["table"], entry))
    for name, metrics in items:
        if not isinstance(name, str) or not isinstance(metrics, Mapping):
            continue
        normalized[name] = {
            "key_null_pct": float(metrics.get("key_null_pct", 0.0) or 0.0),
            "fk_success_pct": float(metrics.get("fk_success_pct", 0.0) or 0.0),
            "orphan_pct": float(
                metrics.get("orphan_pct", metrics.get("fk_orphan_pct", 0.0) or 0.0)
            ),
            "dup_key_pct": float(
                metrics.get("dup_key_pct", metrics.get("dup_keys_pct", 0.0) or 0.0)
            ),
            "p95_ingest_lag_min": float(
                metrics.get("p95_ingest_lag_min", 0.0) or 0.0
            ),
            "row_count": int(metrics.get("row_count", 0) or 0),
            "key_null_spikes": metrics.get(
                "key_null_spikes", metrics.get("null_spike_days", [])
            ),
        }
    return normalized


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
        golden_dir = Path(__file__).resolve().parent / "goldens"
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

        report_dir = Path("reports") / "neobank"
        data_health = json.loads((report_dir / "data_health.json").read_text())
        themes = json.loads((report_dir / "themes.json").read_text())
        exec_summary = json.loads((report_dir / "exec_summary.json").read_text())
        figure_hashes = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (report_dir / "figures").glob("*.png")
        }

        if os.getenv("UPDATE_GOLDENS") == "1":
            golden_dir.mkdir(parents=True, exist_ok=True)
            (golden_dir / "data_health.json").write_text(
                json.dumps(data_health, indent=2), encoding="utf-8"
            )
            (golden_dir / "themes.json").write_text(
                json.dumps(themes, indent=2), encoding="utf-8"
            )
            (golden_dir / "figures.json").write_text(
                json.dumps(figure_hashes, indent=2), encoding="utf-8"
            )
        else:
            expected_data_health = json.loads(
                (golden_dir / "data_health.json").read_text(encoding="utf-8")
            )
            expected_themes = json.loads(
                (golden_dir / "themes.json").read_text(encoding="utf-8")
            )
            expected_figures = json.loads(
                (golden_dir / "figures.json").read_text(encoding="utf-8")
            )

            assert _normalize_data_health_tables(
                data_health.get("tables")
            ) == _normalize_data_health_tables(expected_data_health.get("tables"))
            if "aggregates" in expected_data_health:
                assert data_health.get("aggregates") == expected_data_health["aggregates"]
            else:
                assert "aggregates" in data_health
            assert themes == expected_themes
            assert figure_hashes == expected_figures

        top_actions = exec_summary.get("top_actions", [])
        assert len(top_actions) == 3
        required_keys = {
            "theme",
            "demand",
            "revenue",
            "severity",
            "recency",
            "score",
            "confidence",
            "examples",
        }
        for action in top_actions:
            assert required_keys.issubset(action.keys())
