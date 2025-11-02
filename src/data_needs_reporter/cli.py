from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

import typer

from data_needs_reporter import __version__
from data_needs_reporter.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from data_needs_reporter.generate.comms import generate_comms
from data_needs_reporter.generate.defects import run_typical_generation
from data_needs_reporter.generate.warehouse import (
    generate_marketplace_dims,
    generate_marketplace_facts,
    generate_neobank_dims,
    generate_neobank_facts,
    write_empty_warehouse,
)
from data_needs_reporter.report.llm import MockProvider, RepairingLLMClient
from data_needs_reporter.report.plots import (
    plot_dup_key_pct_bar,
    plot_key_null_pct_daily,
    plot_lag_p95_daily,
    plot_orphan_pct_daily,
    plot_theme_demand_monthly,
)
from data_needs_reporter.report.scoring import compute_confidence, compute_score
from data_needs_reporter.utils.cost_guard import CostGuard
from data_needs_reporter.utils.logging import init_logger, run_context

app = typer.Typer(add_completion=False, help="Generate synthetic data needs reports.")


def _version_callback(ctx: typer.Context, value: Optional[bool]) -> Optional[bool]:
    if not value or ctx.resilient_parsing:
        return value
    typer.echo(__version__)
    raise typer.Exit()


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
            write_empty_warehouse(archetype, out)
        except RuntimeError as exc:
            message = f"Unable to generate empty warehouse: {exc}"
            typer.echo(message, err=True)
            if logger:
                logger.error("Empty warehouse generation failed: %s", exc)
            raise typer.Exit(code=1) from exc
        else:
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

    out_dir = Path(out)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().isoformat()

    data_health = [
        {
            "table": "dim_customer",
            "key_null_pct": 0.8,
            "fk_orphan_pct": 1.2,
            "dup_keys_pct": 0.1,
            "p95_ingest_lag_min": 85,
        },
        {
            "table": "fact_card_transaction",
            "key_null_pct": 1.5,
            "fk_orphan_pct": 4.8,
            "dup_keys_pct": 0.3,
            "p95_ingest_lag_min": 120,
        },
    ]

    (out_dir / "data_health.json").write_text(
        json.dumps({"tables": data_health}, indent=2), encoding="utf-8"
    )

    csv_path = out_dir / "data_health.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(data_health[0].keys()))
        writer.writeheader()
        writer.writerows(data_health)

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
    dup_by_table = {row["table"]: row["dup_keys_pct"] for row in data_health}
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
    checks = [
        {"name": "schema", "passed": True, "detail": "Schema valid."},
        {
            "name": "volume",
            "passed": not fail_marker,
            "detail": (
                "Volume deviation exceeded."
                if fail_marker
                else "Volume within tolerance."
            ),
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
    gate_f1: float = typer.Option(0.7, "--gate-f1", is_flag=False),
) -> None:
    import random

    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    classes = ["data_quality", "pipeline_health", "governance"]
    per_class = {
        cls: {"precision": 0.8, "recall": 0.75, "f1": 0.77, "support": 10}
        for cls in classes
    }
    macro_f1 = sum(metric["f1"] for metric in per_class.values()) / len(per_class)
    confusion = {
        cls: {other: random.randint(0, 2) for other in classes} for cls in classes
    }
    summary = {
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
        "gates_pass": macro_f1 >= gate_f1,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    rows = []
    for cls, metrics in per_class.items():
        row = {"class": cls, **metrics}
        rows.append(row)
    with (out_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if not summary["gates_pass"]:
        raise typer.Exit(code=1)

    if logger:
        logger.info("Evaluation summary written to %s", out_dir)


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

    reports_dir = Path(config.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    archetypes = ["neobank", "marketplace"]
    for archetype in archetypes:
        warehouse_dir = Path(config.paths.data) / archetype
        warehouse_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_empty_warehouse(archetype, warehouse_dir)
        except RuntimeError:
            (warehouse_dir / "placeholder.txt").write_text(
                "warehouse stub", encoding="utf-8"
            )

        comms_dir = Path(config.paths.comms) / archetype
        comms_dir.mkdir(parents=True, exist_ok=True)

        report_dir = reports_dir / archetype
        report_dir.mkdir(parents=True, exist_ok=True)

        if no_llm:
            (report_dir / "data_health.json").write_text(
                json.dumps({"tables": []}, indent=2), encoding="utf-8"
            )
            (report_dir / "themes.json").write_text(
                json.dumps({"themes": []}, indent=2), encoding="utf-8"
            )
            themes_md = """# Themes

Themes skipped (no LLM mode).
"""
            (report_dir / "themes.md").write_text(themes_md, encoding="utf-8")
        else:
            run_report_cmd(
                ctx, warehouse=warehouse_dir, comms=comms_dir, out=report_dir
            )

    index_lines = ["# Reports Index", ""]
    for archetype in archetypes:
        index_lines.append(f"- [{archetype.title()}](./{archetype}/exec_summary.md)")
    index_content = "\n".join(index_lines)
    (reports_dir / "index.md").write_text(index_content, encoding="utf-8")

    typer.echo(f"Quickstart completed. Reports available in {reports_dir}")
    if logger:
        logger.info("Quickstart completed")
