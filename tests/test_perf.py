from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH_SPEC = importlib.util.spec_from_file_location(
    "bench_module", Path(__file__).resolve().parents[1] / "scripts" / "bench.py"
)
assert _BENCH_SPEC and _BENCH_SPEC.loader
bench_module = importlib.util.module_from_spec(_BENCH_SPEC)
sys.modules["bench_module"] = bench_module
_BENCH_SPEC.loader.exec_module(bench_module)
render_markdown_summary = bench_module.render_markdown_summary


@pytest.mark.performance
@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="Performance budgets are informational in CI.",
)
def test_perf_benchmarks_respect_budgets(tmp_path: Path) -> None:
    """Run the bench script locally and ensure all commands meet their targets."""

    cmd = [
        sys.executable,
        "scripts/bench.py",
        "--workspace",
        str(tmp_path),
        "--report-format",
        "json",
        "--report-file",
        str(tmp_path / "bench.json"),
        "--fail-on-budget",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    detail = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, f"bench.py failed\n{detail}"

    payload_path = tmp_path / "bench.json"
    assert payload_path.exists(), "bench.json missing in workspace"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload.get("all_within_budget"), f"Benchmarks exceeded budgets\n{detail}"


def test_render_markdown_summary_headers() -> None:
    summary = {
        "commands": [
            {
                "name": "run-report",
                "samples": 5,
                "p50_ms": 120.0,
                "p95_ms": 150.0,
                "max_ms": 175.0,
            }
        ]
    }
    table = render_markdown_summary(summary)
    assert "| Step | Samples | P50_ms | P95_ms | Max_ms |" in table
    assert "run-report" in table


def test_render_markdown_summary_defaults() -> None:
    summary = {"commands": [{"name": "validate"}]}
    table = render_markdown_summary(summary)
    assert "validate" in table
    assert "| validate | 1 | 0.0 | 0.0 | 0.0 |" in table
