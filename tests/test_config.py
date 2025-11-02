import pytest
from pydantic import ValidationError

from data_needs_reporter.config import DEFAULT_CONFIG_PATH, AppConfig, load_config


def test_default_config_matches_yaml_defaults() -> None:
    config = load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})

    assert isinstance(config, AppConfig)
    assert config.classification.api_cap_usd == 1.0
    assert config.report.window_days == 30
    assert config.cache.enabled is True


def test_env_override_wins_over_yaml() -> None:
    env = {"LLM_API_CAP_USD": "0.8"}
    config = load_config(DEFAULT_CONFIG_PATH, None, env=env, cli_overrides={})

    assert config.classification.api_cap_usd == 0.8


def test_cli_override_beats_env() -> None:
    env = {"LLM_API_CAP_USD": "0.8"}
    cli_overrides = {"classification.api_cap_usd": 0.5}
    config = load_config(
        DEFAULT_CONFIG_PATH, None, env=env, cli_overrides=cli_overrides
    )

    assert config.classification.api_cap_usd == 0.5


def test_invalid_override_type_raises_validation_error() -> None:
    cli_overrides = {"classification.api_cap_usd": "not-a-number"}

    with pytest.raises(ValidationError) as exc:
        load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides=cli_overrides)

    assert "api_cap_usd" in str(exc.value)
