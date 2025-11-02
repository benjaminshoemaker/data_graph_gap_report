import time

from typer.testing import CliRunner

from data_needs_reporter.cli import app

runner = CliRunner()


def test_quickstart_performance():
    with runner.isolated_filesystem():
        start = time.time()
        result = runner.invoke(app, ["quickstart", "--no-llm"])
        assert result.exit_code == 0
        elapsed = time.time() - start
        assert elapsed < 5
