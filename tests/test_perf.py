import os
import time

import pytest
from typer.testing import CliRunner

from data_needs_reporter.cli import app

if os.getenv("CI"):  # pragma: no cover - exercised in CI
    pytest.skip("Performance timings are skipped in CI.", allow_module_level=True)

pytest.importorskip("polars")

runner = CliRunner()

COMMAND_BUDGETS: tuple[tuple[str, list[str], float], ...] = (
    (
        "gen-warehouse",
        ["gen-warehouse", "--archetype", "neobank", "--out", "data/neobank", "--dry-run"],
        3.0,
    ),
    (
        "gen-comms",
        ["gen-comms", "--archetype", "neobank", "--out", "comms/neobank"],
        3.5,
    ),
    (
        "run-report",
        [
            "run-report",
            "--warehouse",
            "data/neobank",
            "--comms",
            "comms/neobank",
            "--out",
            "reports/neobank",
        ],
        2.5,
    ),
    (
        "validate",
        [
            "validate",
            "--warehouse",
            "data/neobank",
            "--comms",
            "comms/neobank",
            "--out",
            "reports/neobank/qc",
            "--strict",
        ],
        2.0,
    ),
)


def test_cli_performance_budgets() -> None:
    timings: list[tuple[str, float, float]] = []
    with runner.isolated_filesystem():
        for name, args, budget in COMMAND_BUDGETS:
            start = time.perf_counter()
            result = runner.invoke(app, args)
            stdout = getattr(result, "stdout", result.output)
            try:
                stderr = result.stderr  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                stderr = ""
            assert (
                result.exit_code == 0
            ), f"{name} failed with output:\n{stdout}\n{stderr}"
            elapsed = time.perf_counter() - start
            timings.append((name, elapsed, budget))
            assert (
                elapsed <= budget
            ), f"{name} exceeded budget: {elapsed:.2f}s > {budget:.2f}s"

    for name, elapsed, budget in timings:
        print(f"{name}: {elapsed:.2f}s (budget ≤ {budget:.2f}s)")
