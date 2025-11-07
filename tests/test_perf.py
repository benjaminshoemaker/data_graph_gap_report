from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.performance
@pytest.mark.skipif(
    os.environ.get("CI"),
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
        "--fail-on-budget",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    detail = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, f"bench.py failed\n{detail}"

    payload = json.loads(result.stdout)
    assert payload.get("all_within_budget"), f"Benchmarks exceeded budgets\n{detail}"
