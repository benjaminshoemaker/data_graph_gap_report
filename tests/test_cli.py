import os
import subprocess
import sys
import json
from pathlib import Path

import typer
from typer.testing import CliRunner

from data_needs_reporter.cli import app


runner = CliRunner()


def test_version_flag_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


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


def test_validate_strict_fails_on_marker(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        warehouse = Path("warehouse")
        comms = Path("comms")
        warehouse.mkdir()
        comms.mkdir()
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
        warehouse.mkdir()
        comms.mkdir()
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


def test_quickstart_creates_index(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["quickstart", "--no-llm"])
        assert result.exit_code == 0, result.stdout
        index = Path("reports") / "index.md"
        assert index.exists()
        content = index.read_text(encoding="utf-8")
        assert "Neobank" in content
        assert "Marketplace" in content


def test_quickstart_no_llm_outputs_minimal(tmp_path: Path) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["quickstart", "--no-llm"])
        assert result.exit_code == 0
        report_dir = Path("reports") / "neobank"
        assert (report_dir / "themes.json").exists()
        themes = json.loads((report_dir / "themes.json").read_text(encoding="utf-8"))
        assert themes["themes"] == []


def test_console_entrypoint_version(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    proc = subprocess.run([sys.executable, "-m", "data_needs_reporter.cli", "--version"], capture_output=True, text=True, env=env)
    assert proc.returncode == 0
