from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from pydantic import BaseModel

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


class PathsConfig(BaseModel):
    data: str
    comms: str
    reports: str
    meta: str


class WarehouseConfig(BaseModel):
    archetypes: list[str]
    scale: str
    quality: str
    trajectory: str
    tz: str
    months: int
    seed: int


class CommsConfig(BaseModel):
    slack_threads: int
    email_threads: int
    nlq: int
    sample_policy: str
    seed: int


class ClassificationConfig(BaseModel):
    engine: str
    provider: str
    model: str
    temperature: float
    max_output_tokens: int
    concurrency: int
    api_cap_usd: float
    prefilter_threshold: float
    env_key_var: str


class EntitiesConfig(BaseModel):
    provider: str
    model: str
    api_cap_usd: float


class ReportSLOs(BaseModel):
    key_null_pct: float
    fk_orphan_pct: float
    dup_keys_pct: float
    p95_ingest_lag_min: float


class ReportConfig(BaseModel):
    scoring_weights: Dict[str, float]
    slos: ReportSLOs
    window_days: int
    demand_base_weights: Dict[str, float]
    demand_weight_caps: Dict[str, float]
    revenue_norm: str


class BudgetConfig(BaseModel):
    mode: str
    safety_margin: float
    coverage_floor_pct: int


class CacheConfig(BaseModel):
    enabled: bool
    dir: str


class AppConfig(BaseModel):
    paths: PathsConfig
    warehouse: WarehouseConfig
    comms: CommsConfig
    classification: ClassificationConfig
    entities: EntitiesConfig
    report: ReportConfig
    budget: BudgetConfig
    cache: CacheConfig


def _resolve_default_config_path() -> Path:
    repo_candidate = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    package_candidate = Path(__file__).resolve().parent / "configs" / "default.yaml"
    for candidate in (repo_candidate, package_candidate):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate default configuration; expected it under "
        f"{repo_candidate} or {package_candidate}."
    )


DEFAULT_CONFIG_PATH = _resolve_default_config_path()
ENV_TO_PATH: Dict[str, Tuple[str, ...]] = {
    "LLM_API_CAP_USD": ("classification", "api_cap_usd"),
    "REPORT_WINDOW_DAYS": ("report", "window_days"),
    "DNR_CACHE_ENABLED": ("cache", "enabled"),
}


def load_config(
    default_path: Path,
    override_yaml_path_or_none: Optional[Path],
    env: Mapping[str, str],
    cli_overrides: Mapping[str, Any],
) -> AppConfig:
    data = _load_yaml(default_path)
    if override_yaml_path_or_none:
        data = _deep_merge(data, _load_yaml(override_yaml_path_or_none))

    data = _apply_env_overrides(data, env)
    data = _apply_cli_overrides(data, cli_overrides)

    return AppConfig.model_validate(data)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return {}

    if yaml is not None:
        loaded = yaml.safe_load(content)
    else:
        loaded = json.loads(content)

    return loaded or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(
    data: Dict[str, Any], env: Mapping[str, str]
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(data))
    for var, path in ENV_TO_PATH.items():
        if var in env:
            _assign_path(updated, path, env[var])
    return updated


def _apply_cli_overrides(
    data: Dict[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(data))
    for key, value in overrides.items():
        path = tuple(key.split(".")) if isinstance(key, str) else tuple(key)
        if not path:
            continue
        _assign_path(updated, path, value)
    return updated


def _assign_path(
    target: MutableMapping[str, Any], path: Tuple[str, ...], value: Any
) -> None:
    cursor: MutableMapping[str, Any] = target
    for part in path[:-1]:
        if part not in cursor or not isinstance(cursor[part], MutableMapping):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[path[-1]] = value


__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG_PATH",
    "load_config",
]
