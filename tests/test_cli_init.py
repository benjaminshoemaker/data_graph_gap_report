from pathlib import Path

from typer.testing import CliRunner

from data_needs_reporter.cli import app

runner = CliRunner()


def test_init_creates_expected_directories_and_config_file() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0

        for path in ("data", "comms", "reports", "meta", ".cache/llm"):
            assert Path(path).is_dir()

        config_path = Path("configs/default.yaml")
        assert config_path.is_file()
        assert config_path.read_text(encoding="utf-8")


def test_init_is_idempotent() -> None:
    with runner.isolated_filesystem():
        first = runner.invoke(app, ["init"])
        assert first.exit_code == 0

        config_path = Path("configs/default.yaml")
        assert config_path.is_file()
        initial_content = config_path.read_text(encoding="utf-8")

        second = runner.invoke(app, ["init"])
        assert second.exit_code == 0
        assert "Created" not in second.stdout
        assert config_path.read_text(encoding="utf-8") == initial_content
