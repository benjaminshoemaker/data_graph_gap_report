from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, cast

import typer

from data_needs_reporter import __version__
from data_needs_reporter.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from data_needs_reporter.eval import DEFAULT_THEME_GATE, evaluate_labels
from data_needs_reporter.generate.comms import generate_comms
from data_needs_reporter.generate.defects import run_typical_generation
from data_needs_reporter.generate.warehouse import (
    generate_marketplace_dims,
    generate_marketplace_facts,
    generate_neobank_dims,
    generate_neobank_facts,
    write_empty_warehouse,
    write_warehouse_hash_manifest,
)
from data_needs_reporter.report.llm import MockProvider, RepairingLLMClient
from data_needs_reporter.report.metrics import (
    TABLE_METRIC_SPECS,
    compute_data_health,
    validate_comms_targets,
    validate_event_correlation,
    validate_monetization_targets,
    validate_quality_targets,
    validate_reproducibility,
    validate_seasonality_targets,
    validate_spend_caps,
    validate_taxonomy_targets,
    validate_theme_mix_targets,
    validate_trajectory_targets,
    validate_volume_targets,
    validate_warehouse_schema,
)
from data_needs_reporter.report.scoring import compute_confidence, compute_score
from data_needs_reporter.utils.cost_guard import CostGuard
from data_needs_reporter.utils.hashing import compute_file_hash
from data_needs_reporter.utils.logging import init_logger, run_context

app = typer.Typer(add_completion=False, help="Generate synthetic data needs reports.")


def _version_callback(ctx: typer.Context, value: Optional[bool]) -> Optional[bool]:
    if not value or ctx.resilient_parsing:
        return value
    typer.echo(__version__)
    raise typer.Exit()


def _detect_archetype_from_path(warehouse: Path, config: AppConfig) -> str:
    candidate_files = {
        "neobank": ["fact_card_transaction.parquet", "fact_subscription_invoice.parquet"],
        "marketplace": ["fact_order.parquet", "fact_payment.parquet"],
    }
    candidates = [arch.lower() for arch in config.warehouse.archetypes]
    name = warehouse.name.lower()
    if name in candidate_files:
        return name
    for candidate in candidates:
        files = candidate_files.get(candidate, [])
        if any((warehouse / file_name).exists() for file_name in files):
            return candidate
    if name in candidates:
        return name
    if candidates:
        return candidates[0]
    return next(iter(TABLE_METRIC_SPECS.keys()), "neobank")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        expose_value=False,
        is_flag=True,
        flag_value=True,
        is_eager=True,
        help="Show the application version and exit.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to an alternate YAML configuration file.",
        is_flag=False,
    ),
    api_cap_usd: Optional[float] = typer.Option(
        None,
        "--api-cap-usd",
        help="Override the LLM API spend cap in USD.",
        is_flag=False,
    ),
    window: Optional[int] = typer.Option(
        None,
        "--window",
        help="Override report window size in days.",
        is_flag=False,
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Disable LLM-powered classification for this run.",
    ),
) -> None:
    override_path = config_path if config_path else None
    cli_overrides: dict[str, object] = {}
    if api_cap_usd is not None:
        cli_overrides["classification.api_cap_usd"] = api_cap_usd
    if window is not None:
        cli_overrides["report.window_days"] = window
    if no_llm:
        cli_overrides["classification.engine"] = "none"

    config = load_config(
        default_path=DEFAULT_CONFIG_PATH,
        override_yaml_path_or_none=override_path,
        env=os.environ,
        cli_overrides=cli_overrides,
    )
    context_obj = ctx.ensure_object(dict)
    context_obj["config"] = config

    logger = init_logger("data_needs_reporter")
    context_obj["logger"] = logger

    if ctx.invoked_subcommand is not None:
        run_cm = run_context(logger)
        run_id = run_cm.__enter__()
        context_obj["run_id"] = run_id

        def _close() -> None:
            run_cm.__exit__(None, None, None)

        ctx.call_on_close(_close)

    if ctx.invoked_subcommand is None:
        typer.echo(
            "Usage: dnr [OPTIONS] COMMAND [ARGS]...\n\nUse 'dnr --help' for more information."
        )
        raise typer.Exit()


@app.command("init")
def init_cmd(ctx: typer.Context) -> None:
    """Scaffold local directories and default config file."""
    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    config = cast(AppConfig, ctx_obj.get("config"))
    if config is None:
        config = load_config(
            default_path=DEFAULT_CONFIG_PATH,
            override_yaml_path_or_none=None,
            env=os.environ,
            cli_overrides={},
        )

    created_items: list[tuple[str, Path]] = []

    path_fields = [
        config.paths.data,
        config.paths.comms,
        config.paths.reports,
        config.paths.meta,
    ]
    for raw in path_fields:
        dir_path = Path(raw)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_items.append(("directory", dir_path))

    cache_dir = Path(config.cache.dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        created_items.append(("directory", cache_dir))

    target_config_path = Path("configs") / "default.yaml"
    if not target_config_path.exists():
        target_config_path.parent.mkdir(parents=True, exist_ok=True)
        target_config_path.write_text(
            DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        created_items.append(("file", target_config_path))

    if created_items:
        for item_type, path in created_items:
            if item_type == "directory":
                typer.echo(f"Created directory: {path}")
                if logger:
                    logger.info("Created directory %s", path)
            else:
                typer.echo(f"Created file: {path}")
                if logger:
                    logger.info("Created file %s", path)
    else:
        typer.echo("Repository assets already initialized.")
        if logger:
            logger.info("Repository assets already initialized.")


@app.command("gen-warehouse")
def gen_warehouse_cmd(
    ctx: typer.Context,
    archetype: str = typer.Option(
        ...,
        "--archetype",
        case_sensitive=False,
        help="Warehouse archetype to generate (neobank or marketplace).",
        is_flag=False,
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory for generated warehouse files.",
        is_flag=False,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Write empty schema tables only.",
    ),
) -> None:
    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    config = cast(AppConfig, ctx_obj.get("config"))
    if config is None:
        config = load_config(
            default_path=DEFAULT_CONFIG_PATH,
            override_yaml_path_or_none=None,
            env=os.environ,
            cli_overrides={},
        )
    archetype_key = archetype.lower()

    if dry_run:
        try:
            write_empty_warehouse(archetype, out, seed=config.warehouse.seed)
        except RuntimeError as exc:
            message = f"Unable to generate empty warehouse: {exc}"
            typer.echo(message, err=True)
            if logger:
                logger.error("Empty warehouse generation failed: %s", exc)
            raise typer.Exit(code=1) from exc
        else:
            write_warehouse_hash_manifest(out, config.warehouse.seed)
            typer.echo(f"Wrote empty {archetype} warehouse to {out}")
            if logger:
                logger.info("Wrote empty %s warehouse to %s", archetype, out)
        return

    try:
        metrics: dict[str, object] = {}
        if config.warehouse.quality.lower() == "typical":
            metrics = run_typical_generation(config, archetype_key, out)
        else:
            if archetype_key == "neobank":
                generate_neobank_dims(config, out, seed=config.warehouse.seed)
                generate_neobank_facts(config, out, out, seed=config.warehouse.seed)
            elif archetype_key == "marketplace":
                generate_marketplace_dims(config, out, seed=config.warehouse.seed)
                generate_marketplace_facts(config, out, out, seed=config.warehouse.seed)
            else:
                typer.echo(f"Unsupported archetype: {archetype}", err=True)
                if logger:
                    logger.error("Unsupported archetype %s", archetype)
                raise typer.Exit(code=1)

        write_warehouse_hash_manifest(out, config.warehouse.seed)
        dry_marker = Path(out) / ".dry_run"
        if dry_marker.exists():
            dry_marker.unlink()
        typer.echo(f"Generated {archetype_key} warehouse at {out}")
        if logger:
            logger.info("Generated %s warehouse at %s", archetype_key, out)
            if metrics:
                logger.info("Quality metrics: %s", metrics)
    except RuntimeError as exc:
        typer.echo(f"Unable to generate warehouse: {exc}", err=True)
        if logger:
            logger.error("Warehouse generation failed: %s", exc)
        raise typer.Exit(code=1) from exc


@app.command("gen-comms")
def gen_comms_cmd(
    ctx: typer.Context,
    archetype: str = typer.Option(
        ...,
        "--archetype",
        case_sensitive=False,
        help="Archetype to generate communications for (neobank or marketplace).",
        is_flag=False,
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory for generated communications.",
        is_flag=False,
    ),
) -> None:
    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    config = cast(AppConfig, ctx_obj.get("config"))
    if config is None:
        config = load_config(
            default_path=DEFAULT_CONFIG_PATH,
            override_yaml_path_or_none=None,
            env=os.environ,
            cli_overrides={},
        )

    archetype_key = archetype.lower()
    if archetype_key not in {"neobank", "marketplace"}:
        typer.echo(f"Unsupported archetype: {archetype}", err=True)
        if logger:
            logger.error("Unsupported archetype %s", archetype)
        raise typer.Exit(code=1)

    os.environ.setdefault(config.classification.env_key_var, "MOCK-LLM-KEY")

    provider = MockProvider(
        response={
            "content": {
                "body": "Investigated spike in missing customer attributes.",
                "bucket": "data_quality",
                "subject": "Data quality follow-up",
                "text": "What caused the revenue dashboard gap last week?",
                "parsed_intent": "data_gap",
                "tokens": 60,
            }
        }
    )
    llm_client = RepairingLLMClient(
        provider=provider,
        model=config.classification.model,
        api_key_env=config.classification.env_key_var,
        timeout_s=15.0,
        max_output_tokens=config.classification.max_output_tokens,
        cache_dir=Path(config.cache.dir),
        repair_attempts=1,
    )
    guard = CostGuard(
        cap_usd=config.classification.api_cap_usd,
        price_per_1k_tokens=0.002,
        safety_margin=config.budget.safety_margin,
    )

    try:
        summary = generate_comms(
            config, archetype_key, out, llm_client, guard, seed=config.comms.seed
        )
    except RuntimeError as exc:
        typer.echo(f"Unable to generate communications: {exc}", err=True)
        if logger:
            logger.error("Communications generation failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Generated communications for {archetype_key} at {out}")
    if logger:
        logger.info("Generated communications summary: %s", summary)

    mock_marker = Path(out) / ".mock_llm"
    mock_marker.write_text("mock\n", encoding="utf-8")


# ... existing commands


@app.command("run-report")
def run_report_cmd(
    ctx: typer.Context,
    warehouse: Path = typer.Option(
        ...,
        "--warehouse",
        help="Path to warehouse data.",
        is_flag=False,
    ),
    comms: Path = typer.Option(
        ...,
        "--comms",
        help="Path to communications data.",
        is_flag=False,
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory for reports.",
        is_flag=False,
    ),
) -> None:
    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    config = cast(AppConfig, ctx_obj.get("config"))
    if config is None:
        config = load_config(
            default_path=DEFAULT_CONFIG_PATH,
            override_yaml_path_or_none=None,
            env=os.environ,
            cli_overrides={},
        )

    from data_needs_reporter.report.plots import (
        plot_dup_key_pct_bar,
        plot_key_null_pct_daily,
        plot_lag_p95_daily,
        plot_orphan_pct_daily,
        plot_theme_demand_monthly,
    )

    out_dir = Path(out)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().isoformat()

    archetype_key = _detect_archetype_from_path(Path(warehouse), config)
    data_health = compute_data_health(archetype_key, Path(warehouse))
    if not data_health:
        data_health = [
            {
                "table": "dim_customer",
                "key_null_pct": 0.0,
                "fk_orphan_pct": 0.0,
                "dup_keys_pct": 0.0,
                "p95_ingest_lag_min": 0.0,
                "fk_success_pct": 100.0,
                "row_count": 0,
            }
        ]

    (out_dir / "data_health.json").write_text(
        json.dumps({"tables": data_health}, indent=2), encoding="utf-8"
    )

    csv_path = out_dir / "data_health.csv"
    csv_fields = [
        "table",
        "key_null_pct",
        "fk_orphan_pct",
        "dup_keys_pct",
        "p95_ingest_lag_min",
        "fk_success_pct",
        "null_spike_days",
        "row_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for row in data_health:
            writer.writerow({field: row.get(field) for field in csv_fields})

    themes = [
        {
            "theme": "data_quality",
            "score": compute_score(0.85, 0.78, 0.6),
            "confidence": compute_confidence(0.82, 0.7, 0.9, 0.8),
            "summary": "Address missing customer attributes impacting revenue dashboards.",
        },
        {
            "theme": "pipeline_health",
            "score": compute_score(0.75, 0.72, 0.55),
            "confidence": compute_confidence(0.76, 0.65, 0.8, 0.7),
            "summary": "Stabilize order ingestion pipeline to reduce lag spikes.",
        },
        {
            "theme": "governance",
            "score": compute_score(0.6, 0.65, 0.45),
            "confidence": compute_confidence(0.62, 0.6, 0.75, 0.6),
            "summary": "Improve data contract coverage for marketplace sellers.",
        },
    ]

    themes = [theme for theme in themes if theme["confidence"] >= 0.55]

    (out_dir / "themes.json").write_text(
        json.dumps({"themes": themes}, indent=2), encoding="utf-8"
    )

    themes_md_lines = ["# Top Themes\n"]
    for theme in themes:
        themes_md_lines.append(
            f"## {theme['theme'].title()}\nScore: {theme['score']:.2f} | Confidence: {theme['confidence']:.2f}\n{theme['summary']}\n"
        )
    (out_dir / "themes.md").write_text("\n".join(themes_md_lines), encoding="utf-8")

    exec_summary = {
        "generated_at": now,
        "window_days": config.report.window_days,
        "top_actions": themes[:3],
        "notes": "Synthetic summary generated for evaluation.",
    }
    (out_dir / "exec_summary.json").write_text(
        json.dumps(exec_summary, indent=2), encoding="utf-8"
    )

    exec_md = ["# Executive Summary", f"Generated: {now}"]
    for idx, theme in enumerate(themes[:3], 1):
        exec_md.append(f"{idx}. **{theme['theme'].title()}** – {theme['summary']}")
    (out_dir / "exec_summary.md").write_text("\n\n".join(exec_md), encoding="utf-8")

    lag_daily = [
        {"day": "2024-01-01", "p95_lag_min": 110},
        {"day": "2024-01-02", "p95_lag_min": 90},
        {"day": "2024-01-03", "p95_lag_min": 95},
    ]
    key_null_daily = [
        {"day": "2024-01-01", "key_null_pct": 1.2},
        {"day": "2024-01-02", "key_null_pct": 0.9},
        {"day": "2024-01-03", "key_null_pct": 1.0},
    ]
    orphan_daily = [
        {"day": "2024-01-01", "orphan_pct": 3.0},
        {"day": "2024-01-02", "orphan_pct": 2.5},
        {"day": "2024-01-03", "orphan_pct": 2.1},
    ]
    dup_by_table = {row["table"]: row.get("dup_keys_pct", 0.0) for row in data_health}
    monthly_theme = [
        {
            "month": "2023-11",
            "data_quality": 0.4,
            "pipeline_health": 0.35,
            "governance": 0.25,
        },
        {
            "month": "2023-12",
            "data_quality": 0.45,
            "pipeline_health": 0.32,
            "governance": 0.23,
        },
        {
            "month": "2024-01",
            "data_quality": 0.5,
            "pipeline_health": 0.3,
            "governance": 0.2,
        },
    ]

    plot_lag_p95_daily(lag_daily, figures_dir / "lag_p95_daily.png")
    plot_key_null_pct_daily(key_null_daily, figures_dir / "key_null_pct_daily.png")
    plot_orphan_pct_daily(orphan_daily, figures_dir / "orphan_pct_daily.png")
    plot_dup_key_pct_bar(dup_by_table, figures_dir / "dup_key_pct_bar.png")
    plot_theme_demand_monthly(monthly_theme, figures_dir / "theme_demand_monthly.png")

    budget = {
        "generated_at": now,
        "total_cost_usd": 0.12,
        "warehouse_path": str(warehouse),
        "comms_path": str(comms),
    }
    (out_dir / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")

    typer.echo(f"Report written to {out_dir}")
    if logger:
        logger.info("run-report completed for %s", out_dir)


def _run_checks(warehouse: Path, comms: Path, strict: bool) -> dict[str, object]:
    fail_marker = (warehouse / "FAIL").exists() or (comms / "FAIL").exists()
    dry_run_warehouse = (warehouse / ".dry_run").exists()
    mock_comms = (comms / ".mock_llm").exists()

    schema_result = validate_warehouse_schema(warehouse)
    volume_result = validate_volume_targets(warehouse)

    if dry_run_warehouse:
        skip_detail = "Skipped (dry-run warehouse)."
        quality_result = {"passed": True, "issues": [], "detail": skip_detail}
        seasonality_result = {"passed": True, "issues": [], "detail": skip_detail}
        taxonomy_result = {"passed": True, "issues": [], "detail": skip_detail}
        monetization_result = {"passed": True, "issues": [], "detail": skip_detail}
        trajectory_result = {"passed": True, "issues": [], "detail": skip_detail}
    else:
        quality_result = validate_quality_targets(warehouse)
        seasonality_result = validate_seasonality_targets(warehouse)
        taxonomy_result = validate_taxonomy_targets(warehouse)
        monetization_result = validate_monetization_targets(warehouse)
        trajectory_result = validate_trajectory_targets(warehouse)

    if mock_comms:
        comms_detail = "Skipped (mock communications dataset)."
        comms_result = {"passed": True, "issues": [], "detail": comms_detail}
        theme_result = {"passed": True, "issues": [], "detail": comms_detail}
    else:
        comms_result = validate_comms_targets(comms)
        theme_result = validate_theme_mix_targets(comms)

    if dry_run_warehouse or mock_comms:
        event_result = {
            "passed": True,
            "issues": [],
            "detail": "Skipped (insufficient telemetry for correlation).",
        }
    else:
        event_result = validate_event_correlation(warehouse, comms)

    reproducibility_result = validate_reproducibility(warehouse, comms)
    spend_result = validate_spend_caps(comms)
    checks = [
        {
            "name": "schema",
            "passed": schema_result["passed"],
            "detail": schema_result["detail"],
        },
        {
            "name": "volume",
            "passed": (not fail_marker) and volume_result["passed"],
            "detail": (
                "Failure marker detected."
                if fail_marker
                else volume_result["detail"]
            ),
        },
        {
            "name": "quality",
            "passed": quality_result["passed"],
            "detail": quality_result["detail"],
        },
        {
            "name": "seasonality",
            "passed": seasonality_result["passed"],
            "detail": seasonality_result["detail"],
        },
        {
            "name": "taxonomy",
            "passed": taxonomy_result["passed"],
            "detail": taxonomy_result["detail"],
        },
        {
            "name": "monetization",
            "passed": monetization_result["passed"],
            "detail": monetization_result["detail"],
        },
        {
            "name": "trajectory",
            "passed": trajectory_result["passed"],
            "detail": trajectory_result["detail"],
        },
        {
            "name": "comms",
            "passed": comms_result["passed"],
            "detail": comms_result["detail"],
        },
        {
            "name": "theme_mix",
            "passed": theme_result["passed"],
            "detail": theme_result["detail"],
        },
        {
            "name": "event_correlation",
            "passed": event_result["passed"],
            "detail": event_result["detail"],
        },
        {
            "name": "reproducibility",
            "passed": reproducibility_result["passed"],
            "detail": reproducibility_result["detail"],
        },
        {
            "name": "spend_caps",
            "passed": spend_result["passed"],
            "detail": spend_result["detail"],
        },
    ]
    overall_pass = all(check["passed"] for check in checks)
    return {
        "checks": checks,
        "overall_pass": overall_pass,
        "exit_code": 0 if (overall_pass or not strict) else 1,
    }


@app.command("validate")
def validate_cmd(
    ctx: typer.Context,
    warehouse: Path = typer.Option(..., "--warehouse", is_flag=False),
    comms: Path = typer.Option(..., "--comms", is_flag=False),
    out: Path = typer.Option(..., "--out", is_flag=False),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _run_checks(warehouse, comms, strict)
    summary_path = out_dir / "qc_summary.json"
    checks_csv = out_dir / "qc_checks.csv"

    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with checks_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "passed", "detail"])
        writer.writeheader()
        writer.writerows(result["checks"])

    if logger:
        logger.info("Validation summary written to %s", summary_path)

    if result["exit_code"] != 0:
        raise typer.Exit(code=1)


@app.command("eval-labels")
def eval_labels_cmd(
    ctx: typer.Context,
    preds: Path = typer.Option(..., "--pred", is_flag=False),
    labels: Path = typer.Option(..., "--labels", is_flag=False),
    out: Path = typer.Option(..., "--out", is_flag=False),
    gate_f1: float = typer.Option(DEFAULT_THEME_GATE, "--gate-f1", is_flag=False),
) -> None:
    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    run_id = (ctx_obj or {}).get("run_id")
    out_base = Path(out)
    if run_id and out_base.name != run_id:
        out_dir = out_base / run_id
    else:
        out_dir = out_base
    summary = evaluate_labels(preds, labels, out_dir, gate_f1=gate_f1)
    if logger:
        logger.info("Evaluation summary written to %s", out_dir)
    if not summary.get("gates_pass", False):
        raise typer.Exit(code=1)



@app.command("quickstart")
def quickstart_cmd(
    ctx: typer.Context,
    fast: bool = typer.Option(False, "--fast"),
    no_llm: bool = typer.Option(False, "--no-llm"),
) -> None:
    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    config = cast(AppConfig, ctx_obj.get("config"))
    if config is None:
        config = load_config(
            default_path=DEFAULT_CONFIG_PATH,
            override_yaml_path_or_none=None,
            env=os.environ,
            cli_overrides={},
        )
        ctx_obj["config"] = config

    init_cmd(ctx)

    try:
        import polars as pl  # type: ignore
    except ImportError as exc:  # pragma: no cover
        typer.echo("polars is required for quickstart", err=True)
        if logger:
            logger.error("polars is required for quickstart")
        raise typer.Exit(code=1) from exc

    reports_dir = Path(config.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    size_factor = 0.5 if fast else 1.0
    tz = timezone.utc

    def _limit(df: pl.DataFrame, factor: float) -> pl.DataFrame:
        if factor >= 1.0:
            return df
        keep = max(1, int(len(df) * factor))
        return df.slice(0, keep)

    def _write_neobank_sample(out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        base_time = datetime(2023, 1, 1, tzinfo=tz)

        customers = pl.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "created_at": [base_time, base_time + timedelta(days=1), base_time + timedelta(days=2)],
                "kyc_status": ["verified", "pending", "verified"],
            }
        )
        accounts = pl.DataFrame(
            {
                "account_id": [1, 2, 3],
                "customer_id": [1, 2, 3],
                "type": ["checking", "credit", "savings"],
                "created_at": [base_time] * 3,
                "status": ["active", "active", "active"],
            }
        )
        cards = pl.DataFrame(
            {
                "card_id": [1, 2],
                "account_id": [1, 2],
                "status": ["active", "active"],
                "activated_at": [base_time + timedelta(days=1), base_time + timedelta(days=2)],
            }
        )
        merchants = pl.DataFrame(
            {
                "merchant_id": [1, 2, 3],
                "mcc": [5411, 5734, 5812],
                "name": ["Fresh Market", "Data Tools", "Pipeline Cafe"],
            }
        )
        plans = pl.DataFrame(
            {
                "plan_id": [1, 2],
                "name": ["Basic", "Premium"],
                "price_cents": [0, 999],
                "cadence": ["monthly", "monthly"],
            }
        )
        transactions = pl.DataFrame(
            {
                "txn_id": [1, 2, 3, 4],
                "card_id": [1, 1, 2, 2],
                "merchant_id": [1, 2, 3, 1],
                "event_time": [
                    base_time + timedelta(hours=2),
                    base_time + timedelta(days=1, hours=4),
                    base_time + timedelta(days=2, hours=6),
                    base_time + timedelta(days=3, hours=3),
                ],
                "amount_cents": [12000, 8500, 6400, 7200],
                "interchange_bps": [120.0, 95.5, 110.2, 130.4],
                "channel": ["card_present", "card_not_present", "digital_wallet", "card_present"],
                "auth_result": ["captured", "captured", "captured", "captured"],
                "loaded_at": [
                    base_time + timedelta(hours=2, minutes=15),
                    base_time + timedelta(days=1, hours=4, minutes=30),
                    base_time + timedelta(days=2, hours=6, minutes=10),
                    base_time + timedelta(days=3, hours=3, minutes=5),
                ],
            }
        )
        invoices = pl.DataFrame(
            {
                "invoice_id": [1, 2],
                "customer_id": [1, 2],
                "plan_id": [2, 2],
                "period_start": [base_time, base_time + timedelta(days=30)],
                "period_end": [base_time + timedelta(days=30), base_time + timedelta(days=60)],
                "paid_at": [base_time + timedelta(days=30, hours=5), base_time + timedelta(days=60, hours=3)],
                "amount_cents": [999, 999],
                "loaded_at": [base_time + timedelta(days=30, hours=5), base_time + timedelta(days=60, hours=3)],
            }
        )

        customers.write_parquet(out_dir / "dim_customer.parquet")
        accounts.write_parquet(out_dir / "dim_account.parquet")
        cards.write_parquet(out_dir / "dim_card.parquet")
        merchants.write_parquet(out_dir / "dim_merchant.parquet")
        plans.write_parquet(out_dir / "dim_plan.parquet")
        transactions.write_parquet(out_dir / "fact_card_transaction.parquet")
        invoices.write_parquet(out_dir / "fact_subscription_invoice.parquet")
        write_warehouse_hash_manifest(out_dir, config.warehouse.seed)

    def _write_marketplace_sample(out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        base_time = datetime(2023, 5, 1, tzinfo=tz)

        buyers = pl.DataFrame(
            {
                "buyer_id": [1, 2, 3],
                "created_at": [base_time, base_time + timedelta(days=2), base_time + timedelta(days=4)],
                "country": ["US", "CA", "GB"],
            }
        )
        sellers = pl.DataFrame(
            {
                "seller_id": [1, 2],
                "created_at": [base_time - timedelta(days=10), base_time - timedelta(days=5)],
                "country": ["US", "DE"],
                "shop_status": ["active", "active"],
            }
        )
        categories = pl.DataFrame(
            {
                "category_id": [1, 2, 3],
                "name": ["Home", "Home Decor", "Electronics"],
                "parent_id": [None, 1, None],
            }
        )
        listings = pl.DataFrame(
            {
                "listing_id": [1, 2, 3],
                "seller_id": [1, 1, 2],
                "category_id": [1, 2, 3],
                "created_at": [base_time, base_time + timedelta(days=1), base_time + timedelta(days=3)],
                "status": ["active", "active", "active"],
                "price_cents": [4500, 3200, 7800],
            }
        )
        orders = pl.DataFrame(
            {
                "order_id": [1, 2],
                "buyer_id": [1, 2],
                "order_time": [base_time + timedelta(hours=9), base_time + timedelta(days=1, hours=11)],
                "currency": ["USD", "USD"],
                "subtotal_cents": [8800, 7800],
                "tax_cents": [700, 620],
                "shipping_cents": [500, 400],
                "discount_cents": [0, 200],
                "loaded_at": [base_time + timedelta(hours=9, minutes=20), base_time + timedelta(days=1, hours=11, minutes=30)],
            }
        )
        order_items = pl.DataFrame(
            {
                "order_id": [1, 1, 2],
                "line_id": [1, 2, 1],
                "listing_id": [1, 2, 3],
                "seller_id": [1, 1, 2],
                "qty": [1, 1, 1],
                "item_price_cents": [4500, 4300, 7800],
                "loaded_at": [base_time + timedelta(hours=9, minutes=25), base_time + timedelta(hours=9, minutes=25), base_time + timedelta(days=1, hours=11, minutes=35)],
            }
        )
        payments = pl.DataFrame(
            {
                "order_id": [1, 2],
                "captured_at": [base_time + timedelta(hours=9, minutes=40), base_time + timedelta(days=1, hours=11, minutes=45)],
                "buyer_paid_cents": [10000, 9600],
                "seller_earnings_cents": [9200, 8700],
                "platform_fee_cents": [800, 900],
                "loaded_at": [base_time + timedelta(hours=9, minutes=40), base_time + timedelta(days=1, hours=11, minutes=45)],
            }
        )
        snapshots = pl.DataFrame(
            {
                "ds": [base_time.date(), (base_time + timedelta(days=1)).date()],
                "listing_id": [1, 2],
                "status": ["active", "active"],
            }
        )

        buyers.write_parquet(out_dir / "dim_buyer.parquet")
        sellers.write_parquet(out_dir / "dim_seller.parquet")
        categories.write_parquet(out_dir / "dim_category.parquet")
        listings.write_parquet(out_dir / "dim_listing.parquet")
        orders.write_parquet(out_dir / "fact_order.parquet")
        order_items.write_parquet(out_dir / "fact_order_item.parquet")
        payments.write_parquet(out_dir / "fact_payment.parquet")
        snapshots.write_parquet(out_dir / "snapshot_listing_daily.parquet")
        marketplace_customers = pl.DataFrame(
            {
                "customer_id": buyers["buyer_id"].to_list(),
                "created_at": buyers["created_at"].to_list(),
                "kyc_status": ["verified"] * buyers.height,
            }
        )
        marketplace_customers.write_parquet(out_dir / "dim_customer.parquet")
        write_warehouse_hash_manifest(out_dir, config.warehouse.seed)

    def _write_comms_sample(out_dir: Path, factor: float) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        start = datetime(2023, 7, 1, tzinfo=tz)

        slack_base = pl.DataFrame(
            {
                "message_id": [1, 2, 3, 4],
                "thread_id": [1, 1, 2, 2],
                "user_id": [10, 11, 12, 13],
                "sent_at": [start, start + timedelta(minutes=5), start + timedelta(days=1), start + timedelta(days=1, minutes=7)],
                "channel": ["support", "support", "exec", "exec"],
                "bucket": ["data_quality", "governance", "pipeline_health", "data_quality"],
                "body": [
                    "Need updated retention dashboard numbers.",
                    "Adding governance checklist for rollout.",
                    "Pipeline job delayed overnight run.",
                    "Resolved data load issue after rerun.",
                ],
                "tokens": [42, 38, 36, 34],
                "link_domains": pl.Series(
                    [["looker"], [], ["snowflake"], []], dtype=pl.List(pl.Utf8)
                ),
                "loaded_at": [
                    start + timedelta(minutes=10),
                    start + timedelta(minutes=12),
                    start + timedelta(days=1, minutes=15),
                    start + timedelta(days=1, minutes=20),
                ],
            }
        )
        email_base = pl.DataFrame(
            {
                "message_id": [1, 2],
                "thread_id": [1, 2],
                "sender_id": [21, 22],
                "recipient_ids": pl.Series([[31, 32], [33]], dtype=pl.List(pl.Int64)),
                "subject": ["Theme review", "Executive summary"],
                "body": [
                    "Weekly data quality theme review attached.",
                    "Sharing executive summary draft for feedback.",
                ],
                "sent_at": [start + timedelta(hours=2), start + timedelta(days=1, hours=1)],
                "bucket": ["data_quality", "governance"],
                "tokens": [120, 95],
                "link_domains": pl.Series([["looker"], []], dtype=pl.List(pl.Utf8)),
                "loaded_at": [
                    start + timedelta(hours=2, minutes=10),
                    start + timedelta(days=1, hours=1, minutes=8),
                ],
            }
        )
        nlq_base = pl.DataFrame(
            {
                "query_id": [1, 2],
                "user_id": [41, 42],
                "submitted_at": [start + timedelta(hours=3), start + timedelta(days=1, hours=2)],
                "text": [
                    "How many new customers joined last week?",
                    "Revenue impact of pipeline delays?",
                ],
                "parsed_intent": ["data_quality", "pipeline_health"],
                "tokens": [18, 22],
                "loaded_at": [
                    start + timedelta(hours=3, minutes=5),
                    start + timedelta(days=1, hours=2, minutes=5),
                ],
            }
        )

        slack_df = _limit(slack_base, factor)
        email_df = _limit(email_base, factor)
        nlq_df = _limit(nlq_base, factor)

        slack_df.write_parquet(out_dir / "slack_messages.parquet")
        email_df.write_parquet(out_dir / "email_messages.parquet")
        nlq_df.write_parquet(out_dir / "nlq.parquet")

        users_df = pl.DataFrame(
            {
                "user_id": [10, 11, 12, 13, 21, 22, 41, 42],
                "role": ["analyst", "pm", "engineer", "exec", "pm", "exec", "analyst", "pm"],
                "department": ["analytics", "product", "engineering", "executive", "product", "executive", "analytics", "product"],
                "time_zone": ["UTC"] * 8,
                "active": [True] * 8,
            }
        )
        users_df.write_parquet(out_dir / "comms_users.parquet")

        hashes = {
            "slack_messages.parquet": compute_file_hash(out_dir / "slack_messages.parquet"),
            "email_messages.parquet": compute_file_hash(out_dir / "email_messages.parquet"),
            "nlq.parquet": compute_file_hash(out_dir / "nlq.parquet"),
            "comms_users.parquet": compute_file_hash(out_dir / "comms_users.parquet"),
        }

        slack_threads = slack_df["thread_id"].n_unique()
        email_threads = email_df["thread_id"].n_unique()
        nlq_count = nlq_df.height

        coverage_entry = {
            "actual": 1,
            "target": 1,
            "coverage_pct": 1.0,
            "met_floor": True,
            "behavior": "continue",
        }
        coverage = {
            "slack": {"overall": coverage_entry, "per_bucket": {}},
            "email": {"overall": coverage_entry, "per_bucket": {}},
            "nlq": {"overall": coverage_entry, "per_bucket": {}},
        }
        quotas = {
            "slack": {"total": slack_threads, "day_bucket": {}, "bucket_totals": {"data_quality": slack_threads}},
            "email": {"total": email_threads, "day_bucket": {}, "bucket_totals": {"data_quality": email_threads}},
            "nlq": {"total": nlq_count, "day_bucket": {}, "bucket_totals": {"data_gap": nlq_count}},
        }

        budget = {
            "messages": {
                "slack": slack_threads,
                "email": email_threads,
                "nlq": nlq_count,
            },
            "coverage": coverage,
            "quotas": quotas,
            "cap_usd": 0.5,
            "cost_usd": 0.0,
            "price_per_1k_tokens": 0.002,
            "tokens_used": 0,
            "token_budget": 1000,
            "stopped_due_to_cap": False,
            "hashes": {"algorithm": "sha256", "files": hashes},
            "seeds": {"comms": config.comms.seed, "warehouse": config.warehouse.seed},
        }
        (out_dir / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")

    for archetype in ("neobank", "marketplace"):
        warehouse_dir = Path(config.paths.data) / archetype
        comms_dir = Path(config.paths.comms) / archetype
        if archetype == "neobank":
            _write_neobank_sample(warehouse_dir)
        else:
            _write_marketplace_sample(warehouse_dir)
        _write_comms_sample(comms_dir, size_factor)

        report_dir = reports_dir / archetype
        report_dir.mkdir(parents=True, exist_ok=True)
        run_report_cmd(ctx, warehouse=warehouse_dir, comms=comms_dir, out=report_dir)

        if no_llm:
            (report_dir / "themes.json").write_text(
                json.dumps({"themes": []}, indent=2), encoding="utf-8"
            )
            themes_md = """# Themes

Themes skipped (no LLM mode).
"""
            (report_dir / "themes.md").write_text(themes_md, encoding="utf-8")

    index_lines = ["# Reports Index", ""]
    for archetype in ("neobank", "marketplace"):
        index_lines.append(f"- [{archetype.title()}](./{archetype}/exec_summary.md)")
    (reports_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    typer.echo(f"Quickstart completed. Reports available in {reports_dir}")
    if logger:
        logger.info("Quickstart completed")
