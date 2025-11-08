from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast

import typer

from data_needs_reporter import __version__
from data_needs_reporter.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from data_needs_reporter.eval import (
    COVERAGE_GATE,
    DEFAULT_THEME_GATE,
    RELEVANCE_GATE,
    THEME_SOURCES,
    collect_eval_rows,
    load_predictions,
    summarize_pairs,
)
from data_needs_reporter.generate.comms import generate_comms
from data_needs_reporter.generate.defects import run_typical_generation
from data_needs_reporter.generate.warehouse import (
    generate_marketplace_dims,
    generate_marketplace_facts,
    generate_neobank_dims,
    generate_neobank_facts,
    write_empty_warehouse,
    write_schema_manifest,
    write_warehouse_hash_manifest,
)
from data_needs_reporter.report.llm import MockProvider, RepairingLLMClient
from data_needs_reporter.report.metrics import (
    TABLE_METRIC_SPECS,
    lookup_invoice_metric_value,
    validate_comms_targets,
    validate_event_correlation,
    validate_marketplace_category_caps,
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
from data_needs_reporter.report.run import (
    assess_budget_health,
    write_data_health_report,
    write_exec_summary,
)
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
        "neobank": [
            "fact_card_transaction.parquet",
            "fact_subscription_invoice.parquet",
        ],
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


def _evaluate_budget_health(comms_path: Path) -> tuple[list[str], bool]:
    budget_path = Path(comms_path) / "budget.json"
    if not budget_path.exists():
        raise FileNotFoundError(f"Budget file not found: {budget_path}")

    try:
        payload = json.loads(budget_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in budget file {budget_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Budget file {budget_path} must be a JSON object.")

    warnings, strict_failure = assess_budget_health(payload)
    return warnings, strict_failure


def _run_labels_evaluation(
    predictions: Any,
    labels_dir: Path,
    pl_module: Any,
) -> tuple[
    dict[str, object], list[dict[str, object]], dict[str, dict[str, dict[str, int]]]
]:
    pl = pl_module
    summary_sources: dict[str, object] = {}
    per_class_rows: list[dict[str, object]] = []
    confusion_outputs: dict[str, dict[str, dict[str, int]]] = {}
    gates: list[dict[str, object]] = []
    overall_true: list[str] = []
    overall_pred: list[str] = []
    overall_conf: list[float] = []
    total_labels = 0
    total_matched = 0

    for source in THEME_SOURCES:
        source_frame = predictions.filter(pl.col("source") == source)
        (
            theme_true,
            theme_pred,
            theme_conf,
            coverage,
            label_count,
        ) = collect_eval_rows(source_frame, labels_dir, source=source, task="theme")
        total_labels += label_count
        total_matched += len(theme_true)
        theme_metrics = summarize_pairs(theme_true, theme_pred, theme_conf)
        theme_metrics.setdefault("overall", {})["coverage"] = coverage

        summary_sources[source] = {
            "coverage": coverage,
            "label_count": label_count,
            "theme": theme_metrics,
        }

        for class_name, cls_metrics in theme_metrics["per_class"].items():
            per_class_rows.append(
                {
                    "source": source,
                    "task": "theme",
                    "class": class_name,
                    "precision": cls_metrics["precision"],
                    "recall": cls_metrics["recall"],
                    "f1": cls_metrics["f1"],
                    "support": cls_metrics["support"],
                }
            )

        confusion_outputs[f"confusion_{source}_theme.json"] = theme_metrics["confusion"]

        gates.append(
            {
                "name": f"{source}_theme_macro_f1",
                "value": theme_metrics["overall"].get("macro_f1", 0.0),
                "threshold": DEFAULT_THEME_GATE,
                "passed": theme_metrics["overall"].get("macro_f1", 0.0)
                >= DEFAULT_THEME_GATE,
            }
        )
        gates.append(
            {
                "name": f"{source}_coverage",
                "value": coverage,
                "threshold": COVERAGE_GATE,
                "passed": coverage >= COVERAGE_GATE,
            }
        )

        overall_true.extend(theme_true)
        overall_pred.extend(theme_pred)
        overall_conf.extend(theme_conf)

        if source in {"slack", "email"}:
            rel_true, rel_pred, rel_conf, _, _ = collect_eval_rows(
                source_frame, labels_dir, source=source, task="relevance"
            )
            rel_metrics = summarize_pairs(rel_true, rel_pred, rel_conf)
            rel_metrics.setdefault("overall", {})["coverage"] = coverage
            summary_sources[source]["relevance"] = rel_metrics
            confusion_outputs[f"confusion_{source}_relevance.json"] = rel_metrics[
                "confusion"
            ]
            for class_name, cls_metrics in rel_metrics["per_class"].items():
                per_class_rows.append(
                    {
                        "source": source,
                        "task": "relevance",
                        "class": class_name,
                        "precision": cls_metrics["precision"],
                        "recall": cls_metrics["recall"],
                        "f1": cls_metrics["f1"],
                        "support": cls_metrics["support"],
                    }
                )
            gates.append(
                {
                    "name": f"{source}_relevance_macro_f1",
                    "value": rel_metrics["overall"].get("macro_f1", 0.0),
                    "threshold": RELEVANCE_GATE,
                    "passed": rel_metrics["overall"].get("macro_f1", 0.0)
                    >= RELEVANCE_GATE,
                }
            )

    overall_metrics = summarize_pairs(overall_true, overall_pred, overall_conf)
    overall_coverage = total_matched / total_labels if total_labels else 0.0
    overall_metrics.setdefault("overall", {})["coverage"] = overall_coverage

    summary = {
        "overall": overall_metrics,
        "sources": summary_sources,
    }
    summary["gates"] = {gate["name"]: gate for gate in gates}
    summary["gates_pass"] = all(gate.get("passed", False) for gate in gates)

    return summary, per_class_rows, confusion_outputs


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

        write_schema_manifest(archetype, out)
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
        cache_enabled=config.cache.enabled,
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
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with non-zero status if coverage or budget issues are detected.",
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

    budget_warnings: list[str] = []
    strict_failure = False
    try:
        budget_warnings, strict_failure = _evaluate_budget_health(comms)
    except FileNotFoundError:
        budget_warnings = []
        strict_failure = False
    except Exception as exc:  # pragma: no cover - unexpected corruption
        message = f"Failed to inspect communications budget: {exc}"
        typer.echo(message, err=True)
        if logger:
            logger.error("Budget inspection failed: %s", exc, exc_info=True)
        raise typer.Exit(code=1) from exc

    now = datetime.utcnow().isoformat()

    warehouse_path = Path(warehouse)
    tz_config = getattr(getattr(config, "warehouse", object()), "tz", "UTC")
    data_health_payload = write_data_health_report(config, warehouse_path, out_dir)
    tables_section = data_health_payload.get("tables", {})
    table_rows: list[dict[str, object]] = []
    if isinstance(tables_section, Mapping):
        for table_name, metrics in sorted(tables_section.items()):
            row: dict[str, object] = {"table": table_name}
            if isinstance(metrics, Mapping):
                row.update(metrics)
            table_rows.append(row)
    if not table_rows:
        table_rows = [
            {
                "table": "dim_customer",
                "row_count": 0,
                "key_null_pct": 0.0,
                "fk_success_pct": 100.0,
                "orphan_pct": 0.0,
                "dup_key_pct": 0.0,
                "p95_ingest_lag_min": 0.0,
                "key_null_spikes": [],
            }
        ]

    csv_path = out_dir / "data_health.csv"
    csv_fields = [
        "table",
        "row_count",
        "key_null_pct",
        "fk_success_pct",
        "orphan_pct",
        "dup_key_pct",
        "p95_ingest_lag_min",
        "key_null_spikes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for row in table_rows:
            serialized: dict[str, object] = {}
            for field in csv_fields:
                value = row.get(field)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                serialized[field] = value
            writer.writerow(serialized)

    write_exec_summary(
        config=config,
        warehouse_path=warehouse_path,
        comms_path=Path(comms),
        out_dir=out_dir,
        data_health=data_health_payload,
        generated_at=now,
    )

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

    def _aggregate_daily(
        rows: Sequence[Mapping[str, object]],
        metric_key: str,
        value_field: str,
    ) -> list[dict[str, float]]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            entries = row.get(metric_key)
            if not isinstance(entries, Sequence):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                day = entry.get("date") or entry.get("day")
                value = entry.get(value_field)
                if day is None or value is None:
                    continue
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    continue
                totals[day] = totals.get(day, 0.0) + value_float
                counts[day] = counts.get(day, 0) + 1
        aggregated = []
        for day in sorted(totals.keys()):
            count = counts.get(day, 1)
            aggregated.append({"day": day, value_field: totals[day] / count})
        return aggregated

    lag_daily = _aggregate_daily(table_rows, "p95_ingest_lag_min_daily", "minutes")
    key_null_daily = _aggregate_daily(table_rows, "key_null_pct_daily", "pct")
    orphan_daily = _aggregate_daily(table_rows, "orphan_pct_daily", "pct")
    dup_by_table = {
        str(row.get("table")): float(row.get("dup_key_pct", 0.0) or 0.0)
        for row in table_rows
    }
    plot_lag_p95_daily(lag_daily, figures_dir / "lag_p95_daily.png")
    plot_key_null_pct_daily(key_null_daily, figures_dir / "key_null_pct_daily.png")
    plot_orphan_pct_daily(orphan_daily, figures_dir / "orphan_pct_daily.png")
    plot_dup_key_pct_bar(dup_by_table, figures_dir / "dup_key_pct_bar.png")
    plot_theme_demand_monthly(
        [
            {
                "month": "2023-11",
                "data_quality": 0.40,
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
                "data_quality": 0.50,
                "pipeline_health": 0.30,
                "governance": 0.20,
            },
        ],
        figures_dir / "theme_demand_monthly.png",
    )

    budget = {
        "generated_at": now,
        "total_cost_usd": 0.12,
        "warehouse_path": str(warehouse),
        "comms_path": str(comms),
    }
    (out_dir / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")

    for warning in budget_warnings:
        typer.echo(f"WARNING: {warning}")
        if logger:
            logger.warning(warning)

    if strict and strict_failure:
        if logger:
            logger.error("Strict run-report failure due to coverage or budget issues.")
        raise typer.Exit(code=1)

    typer.echo(f"Report written to {out_dir}")
    if logger:
        logger.info("run-report completed for %s", out_dir)


def _run_checks(
    warehouse: Path, comms: Path, strict: bool, config: AppConfig
) -> dict[str, object]:
    fail_marker = (warehouse / "FAIL").exists() or (comms / "FAIL").exists()
    dry_run_warehouse = (warehouse / ".dry_run").exists()
    mock_comms = (comms / ".mock_llm").exists()
    archetype_key = _detect_archetype_from_path(warehouse, config)

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
        seasonality_result = validate_seasonality_targets(
            warehouse, tz=getattr(config.warehouse, "tz", None)
        )
        taxonomy_result = validate_taxonomy_targets(warehouse)
        monetization_cfg = getattr(config.report, "monetization", None)
        attach_targets = None
        if monetization_cfg is not None:
            if hasattr(monetization_cfg, "attach_targets"):
                attach_targets = getattr(monetization_cfg, "attach_targets", None)
            elif isinstance(monetization_cfg, Mapping):
                attach_targets = monetization_cfg.get("attach_targets")
        monetization_result = validate_monetization_targets(
            warehouse, attach_targets=attach_targets
        )
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
                "Failure marker detected." if fail_marker else volume_result["detail"]
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
    if not dry_run_warehouse and archetype_key == "marketplace":
        marketplace_cfg = getattr(config.report, "marketplace", None)
        category_caps_cfg = getattr(marketplace_cfg, "category_caps", None)
        caps_values: Optional[Mapping[str, float]] = None
        rollup_caps = False
        if category_caps_cfg is not None:
            if hasattr(category_caps_cfg, "values"):
                caps_values = dict(getattr(category_caps_cfg, "values") or {})
                rollup_caps = bool(
                    getattr(category_caps_cfg, "rollup_to_parent", False)
                )
            elif isinstance(category_caps_cfg, Mapping):
                raw_values = category_caps_cfg.get("values")
                if isinstance(raw_values, Mapping):
                    caps_values = dict(raw_values)
                else:
                    caps_values = {
                        key: float(value)
                        for key, value in category_caps_cfg.items()
                        if isinstance(value, (int, float))
                    }
                rollup_caps = bool(category_caps_cfg.get("rollup_to_parent", False))
        if caps_values:
            category_result = validate_marketplace_category_caps(
                warehouse,
                category_caps=caps_values,
                rollup_to_parent=rollup_caps,
            )
            checks.append(
                {
                    "name": "marketplace_category_caps",
                    "passed": category_result["passed"],
                    "detail": category_result["detail"],
                }
            )
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
    config = cast(AppConfig, ctx_obj.get("config"))
    if config is None:
        config = load_config(
            default_path=DEFAULT_CONFIG_PATH,
            override_yaml_path_or_none=None,
            env=os.environ,
            cli_overrides={},
        )
        ctx_obj["config"] = config
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _run_checks(warehouse, comms, strict, config)
    archetype_key = _detect_archetype_from_path(warehouse, config)
    checks: list[dict[str, object]] = list(result.get("checks", []))
    issues: list[str] = [
        f"{check.get('name')} failed: {check.get('detail')}"
        for check in checks
        if not check.get("passed", False)
    ]

    slo_config = getattr(config.report, "slos", {})
    if hasattr(slo_config, "model_dump"):
        slo_thresholds: Mapping[str, float] = slo_config.model_dump()
    elif isinstance(slo_config, Mapping):
        slo_thresholds = dict(slo_config)
    else:
        slo_thresholds = {}
    metric_aliases: Mapping[str, Sequence[str]] = {
        "key_null_pct": ("key_null_pct",),
        "fk_orphan_pct": ("fk_orphan_pct", "orphan_pct"),
        "dup_keys_pct": ("dup_keys_pct", "dup_key_pct"),
        "p95_ingest_lag_min": ("p95_ingest_lag_min",),
    }

    def _extract_metric(source: object, metric_name: str) -> Optional[float]:
        if not isinstance(source, Mapping):
            return None
        for alias in metric_aliases.get(metric_name, (metric_name,)):
            value = source.get(alias)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _format_value(metric_name: str, value: float) -> str:
        if metric_name.endswith("_min") or metric_name.endswith("_minutes"):
            return f"{value:.2f} min"
        if metric_name.endswith("_count") or metric_name == "row_count":
            return f"{value:.0f}"
        return f"{value:.2f}%"

    data_health_path = out_dir.parent / "data_health.json"
    data_health_payload: Optional[Mapping[str, object]] = None
    if data_health_path.exists():
        try:
            data_health_payload = json.loads(
                data_health_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            detail = f"Unable to parse {data_health_path}: {exc}"
            checks.append({"name": "data_health", "passed": False, "detail": detail})
            issues.append(detail)
            data_health_payload = None
    else:
        detail = f"Missing data health report at {data_health_path}"
        checks.append({"name": "data_health", "passed": False, "detail": detail})
        issues.append(detail)

    aggregates: Mapping[str, object] = {}
    aggregates_by_table: Mapping[str, object] = {}
    tables_section: object = {}
    if isinstance(data_health_payload, Mapping):
        aggregates_raw = data_health_payload.get("aggregates", {})
        if isinstance(aggregates_raw, Mapping):
            aggregates = aggregates_raw  # type: ignore[assignment]
        aggregates_by_table_raw = data_health_payload.get("aggregates_by_table", {})
        if isinstance(aggregates_by_table_raw, Mapping):
            aggregates_by_table = aggregates_by_table_raw  # type: ignore[assignment]
        tables_section = data_health_payload.get("tables", {})

    def _lookup_table(name: str) -> Optional[Mapping[str, object]]:
        if isinstance(tables_section, Mapping):
            table_entry = tables_section.get(name)
            return table_entry if isinstance(table_entry, Mapping) else None
        if isinstance(tables_section, list):
            for entry in tables_section:
                if isinstance(entry, Mapping) and str(entry.get("table")) == name:
                    return entry
        return None

    invoice_metrics = _lookup_table("fact_subscription_invoice")
    invoice_aggregate_metrics = None
    if isinstance(aggregates_by_table, Mapping):
        entry = aggregates_by_table.get("fact_subscription_invoice")
        if isinstance(entry, Mapping):
            invoice_aggregate_metrics = entry
    evening_metrics = None
    if isinstance(data_health_payload, Mapping):
        raw_evening = data_health_payload.get("marketplace_evening_window")
        if isinstance(raw_evening, Mapping):
            evening_metrics = raw_evening

    for metric_name, threshold_raw in slo_thresholds.items():
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            continue
        agg_value = _extract_metric(aggregates, metric_name)
        agg_passed = agg_value is not None and agg_value <= threshold
        if agg_value is None:
            detail = f"Aggregate {metric_name} missing in {data_health_path.name}"
            agg_passed = False
        else:
            detail = (
                f"Aggregate {metric_name} {_format_value(metric_name, agg_value)} "
                f"(limit {_format_value(metric_name, threshold)})"
            )
        checks.append(
            {
                "name": f"slo.aggregate.{metric_name}",
                "passed": agg_passed,
                "detail": detail,
            }
        )
        if not agg_passed:
            issues.append(detail)

        table_value = _extract_metric(invoice_metrics, metric_name)
        table_passed = table_value is not None and table_value <= threshold
        if invoice_metrics is None:
            table_detail = "fact_subscription_invoice metrics missing"
            table_passed = False
        elif table_value is None:
            table_detail = f"fact_subscription_invoice missing {metric_name} metric"
            table_passed = False
        else:
            table_detail = (
                f"fact_subscription_invoice {metric_name} "
                f"{_format_value(metric_name, table_value)} "
                f"(limit {_format_value(metric_name, threshold)})"
            )
        checks.append(
            {
                "name": f"slo.fact_subscription_invoice.{metric_name}",
                "passed": table_passed,
                "detail": table_detail,
            }
        )
        if not table_passed:
            issues.append(table_detail)

    invoice_cfg_raw = getattr(config.report, "invoice_aggregates", None)
    invoice_aggs_enabled = False
    invoice_slos_cfg: Optional[object] = None
    if invoice_cfg_raw is not None:
        if hasattr(invoice_cfg_raw, "enabled"):
            invoice_aggs_enabled = bool(getattr(invoice_cfg_raw, "enabled", False))
            invoice_slos_cfg = getattr(invoice_cfg_raw, "slos", None)
        elif isinstance(invoice_cfg_raw, Mapping):
            invoice_aggs_enabled = bool(invoice_cfg_raw.get("enabled"))
            invoice_slos_cfg = invoice_cfg_raw.get("slos")

    invoice_aggregate_thresholds: Dict[str, float] = {}
    if invoice_slos_cfg is not None:
        if hasattr(invoice_slos_cfg, "model_dump"):
            raw_slos = invoice_slos_cfg.model_dump(exclude_none=True)
        elif isinstance(invoice_slos_cfg, Mapping):
            raw_slos = {k: v for k, v in invoice_slos_cfg.items() if v is not None}
        else:
            raw_slos = {}
        for key, value in raw_slos.items():
            try:
                invoice_aggregate_thresholds[key] = float(value)
            except (TypeError, ValueError):
                continue
    elif invoice_aggs_enabled:
        for key, value in slo_thresholds.items():
            try:
                invoice_aggregate_thresholds[key] = float(value)
            except (TypeError, ValueError):
                continue

    if (
        invoice_aggs_enabled
        and invoice_aggregate_metrics is not None
        and invoice_aggregate_thresholds
    ):
        for metric_name, threshold_raw in invoice_aggregate_thresholds.items():
            try:
                threshold = float(threshold_raw)
            except (TypeError, ValueError):
                continue
            comparator, display_metric, agg_value = lookup_invoice_metric_value(
                invoice_aggregate_metrics, metric_name
            )
            if comparator == "min":
                agg_passed = agg_value is not None and agg_value >= threshold
            else:
                agg_passed = agg_value is not None and agg_value <= threshold
            if agg_value is None:
                detail = (
                    "fact_subscription_invoice aggregate "
                    f"{display_metric or metric_name} missing in "
                    f"{data_health_path.name}"
                )
            else:
                limit_text = _format_value(display_metric, threshold)
                comparison_symbol = "≥" if comparator == "min" else "≤"
                detail = (
                    f"fact_subscription_invoice aggregate {display_metric} "
                    f"{_format_value(display_metric, agg_value)} "
                    f"(limit {limit_text}, {comparison_symbol})"
                )
            checks.append(
                {
                    "name": f"slo.invoice_aggregates.{metric_name}",
                    "passed": agg_passed,
                    "detail": detail,
                }
            )
            if not agg_passed:
                issues.append(detail)
    elif invoice_aggs_enabled and invoice_aggregate_thresholds:
        detail = (
            "fact_subscription_invoice aggregates missing in "
            f"{data_health_path.name}"
        )
        checks.append(
            {"name": "slo.invoice_aggregates", "passed": False, "detail": detail}
        )
        issues.append(detail)

    if archetype_key == "marketplace":
        marketplace_cfg = getattr(config.report, "marketplace", None)
        evening_cfg = (
            getattr(marketplace_cfg, "evening_window", None)
            if marketplace_cfg
            else None
        )
        start_hour = getattr(evening_cfg, "start_hour", 17)
        end_hour = getattr(evening_cfg, "end_hour", 21)
        min_share_pct = getattr(evening_cfg, "min_share_pct", 20.0)
        min_days_pct = getattr(evening_cfg, "min_days_pct", 80.0)
        if evening_metrics is None:
            detail = (
                "marketplace evening coverage metrics missing in "
                f"{data_health_path.name}"
            )
            for suffix in ("overall_share_pct", "days_pct"):
                checks.append(
                    {
                        "name": f"slo.marketplace_evening.{suffix}",
                        "passed": False,
                        "detail": detail,
                    }
                )
            issues.append(detail)
        else:
            overall_value = float(evening_metrics.get("overall_share_pct") or 0.0)
            overall_detail = (
                f"Marketplace evening overall share "
                f"{_format_value('overall_share_pct', overall_value)} "
                f"(limit {_format_value('overall_share_pct', min_share_pct)}, ≥)"
            )
            overall_pass = overall_value >= float(min_share_pct)
            checks.append(
                {
                    "name": "slo.marketplace_evening.overall_share_pct",
                    "passed": overall_pass,
                    "detail": overall_detail,
                }
            )
            if not overall_pass:
                issues.append(overall_detail)

            days_value = float(evening_metrics.get("days_pct") or 0.0)
            days_detail = (
                f"Marketplace evening qualifying days "
                f"{_format_value('days_pct', days_value)} "
                f"(limit {_format_value('days_pct', min_days_pct)}, ≥)"
            )
            days_pass = days_value >= float(min_days_pct)
            checks.append(
                {
                    "name": "slo.marketplace_evening.days_pct",
                    "passed": days_pass,
                    "detail": days_detail,
                }
            )
            if not days_pass:
                issues.append(days_detail)

    overall_pass = all(check.get("passed", False) for check in checks)
    exit_code = 0 if (overall_pass or not strict) else 1

    detail_message = (
        "All checks passed."
        if overall_pass
        else ("; ".join(issues) if issues else "Check qc_checks.csv for details.")
    )

    summary = {
        "passed": overall_pass,
        "overall_pass": overall_pass,
        "issues": issues,
        "detail": detail_message,
        "checks": checks,
        "exit_code": exit_code,
    }

    summary_path = out_dir / "qc_summary.json"
    checks_csv = out_dir / "qc_checks.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with checks_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "passed", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    if logger:
        logger.info("Validation summary written to %s", summary_path)

    if exit_code != 0:
        raise typer.Exit(code=1)


@app.command("eval-labels")
def eval_labels_cmd(
    ctx: typer.Context,
    preds: Path = typer.Option(
        ..., "--pred", help="Directory with prediction parquet files."
    ),
    labels: Path = typer.Option(
        ..., "--labels", help="Directory containing oracle label parquet files."
    ),
    out: Path = typer.Option(..., "--out", help="Evaluation report output directory."),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with non-zero code when any evaluation gate fails.",
    ),
) -> None:
    ctx_obj = ctx.obj or {}
    logger = ctx_obj.get("logger")
    try:
        import polars as pl  # type: ignore
    except ImportError as exc:  # pragma: no cover
        typer.echo("polars is required for eval-labels", err=True)
        if logger:
            logger.error("polars is required for eval-labels")
        raise typer.Exit(code=1) from exc

    predictions = load_predictions(preds)
    labels_dir = Path(labels)
    summary, per_class_rows, confusion_outputs = _run_labels_evaluation(
        predictions, labels_dir, pl
    )

    run_id = (ctx_obj or {}).get("run_id")
    out_base = Path(out)
    if run_id and out_base.name != run_id:
        out_dir = out_base / run_id
    else:
        out_dir = out_base
    out_dir.mkdir(parents=True, exist_ok=True)

    summary["run_id"] = run_id

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if per_class_rows:
        pl.DataFrame(per_class_rows).write_csv(out_dir / "per_class.csv")

    for filename, matrix in confusion_outputs.items():
        (out_dir / filename).write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    if logger:
        logger.info("Evaluation summary written to %s", out_dir)

    if strict and not summary.get("gates_pass", False):
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
                "created_at": [
                    base_time,
                    base_time + timedelta(days=1),
                    base_time + timedelta(days=2),
                ],
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
                "activated_at": [
                    base_time + timedelta(days=1),
                    base_time + timedelta(days=2),
                ],
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
                "channel": [
                    "card_present",
                    "card_not_present",
                    "digital_wallet",
                    "card_present",
                ],
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
                "period_end": [
                    base_time + timedelta(days=30),
                    base_time + timedelta(days=60),
                ],
                "paid_at": [
                    base_time + timedelta(days=30, hours=5),
                    base_time + timedelta(days=60, hours=3),
                ],
                "amount_cents": [999, 999],
                "loaded_at": [
                    base_time + timedelta(days=30, hours=5),
                    base_time + timedelta(days=60, hours=3),
                ],
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
                "created_at": [
                    base_time,
                    base_time + timedelta(days=2),
                    base_time + timedelta(days=4),
                ],
                "country": ["US", "CA", "GB"],
            }
        )
        sellers = pl.DataFrame(
            {
                "seller_id": [1, 2],
                "created_at": [
                    base_time - timedelta(days=10),
                    base_time - timedelta(days=5),
                ],
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
                "created_at": [
                    base_time,
                    base_time + timedelta(days=1),
                    base_time + timedelta(days=3),
                ],
                "status": ["active", "active", "active"],
                "price_cents": [4500, 3200, 7800],
            }
        )
        orders = pl.DataFrame(
            {
                "order_id": [1, 2],
                "buyer_id": [1, 2],
                "order_time": [
                    base_time + timedelta(hours=9),
                    base_time + timedelta(days=1, hours=11),
                ],
                "currency": ["USD", "USD"],
                "subtotal_cents": [8800, 7800],
                "tax_cents": [700, 620],
                "shipping_cents": [500, 400],
                "discount_cents": [0, 200],
                "loaded_at": [
                    base_time + timedelta(hours=9, minutes=20),
                    base_time + timedelta(days=1, hours=11, minutes=30),
                ],
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
                "loaded_at": [
                    base_time + timedelta(hours=9, minutes=25),
                    base_time + timedelta(hours=9, minutes=25),
                    base_time + timedelta(days=1, hours=11, minutes=35),
                ],
            }
        )
        payments = pl.DataFrame(
            {
                "order_id": [1, 2],
                "captured_at": [
                    base_time + timedelta(hours=9, minutes=40),
                    base_time + timedelta(days=1, hours=11, minutes=45),
                ],
                "buyer_paid_cents": [10000, 9600],
                "seller_earnings_cents": [9200, 8700],
                "platform_fee_cents": [800, 900],
                "loaded_at": [
                    base_time + timedelta(hours=9, minutes=40),
                    base_time + timedelta(days=1, hours=11, minutes=45),
                ],
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
                "sent_at": [
                    start,
                    start + timedelta(minutes=5),
                    start + timedelta(days=1),
                    start + timedelta(days=1, minutes=7),
                ],
                "channel": ["support", "support", "exec", "exec"],
                "bucket": [
                    "data_quality",
                    "governance",
                    "pipeline_health",
                    "data_quality",
                ],
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
                "sent_at": [
                    start + timedelta(hours=2),
                    start + timedelta(days=1, hours=1),
                ],
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
                "submitted_at": [
                    start + timedelta(hours=3),
                    start + timedelta(days=1, hours=2),
                ],
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
                "role": [
                    "analyst",
                    "pm",
                    "engineer",
                    "exec",
                    "pm",
                    "exec",
                    "analyst",
                    "pm",
                ],
                "department": [
                    "analytics",
                    "product",
                    "engineering",
                    "executive",
                    "product",
                    "executive",
                    "analytics",
                    "product",
                ],
                "time_zone": ["UTC"] * 8,
                "active": [True] * 8,
            }
        )
        users_df.write_parquet(out_dir / "comms_users.parquet")

        hashes = {
            "slack_messages.parquet": compute_file_hash(
                out_dir / "slack_messages.parquet"
            ),
            "email_messages.parquet": compute_file_hash(
                out_dir / "email_messages.parquet"
            ),
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
            "slack": {
                "total": slack_threads,
                "day_bucket": {},
                "bucket_totals": {"data_quality": slack_threads},
            },
            "email": {
                "total": email_threads,
                "day_bucket": {},
                "bucket_totals": {"data_quality": email_threads},
            },
            "nlq": {
                "total": nlq_count,
                "day_bucket": {},
                "bucket_totals": {"data_gap": nlq_count},
            },
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
        (out_dir / "budget.json").write_text(
            json.dumps(budget, indent=2), encoding="utf-8"
        )

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
