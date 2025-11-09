from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from data_needs_reporter.config import AppConfig
from data_needs_reporter.report.entities import (
    EntityExtractionConfig,
    build_entity_dictionary,
    extract_entities,
)
from data_needs_reporter.report.llm import LLMClient
from data_needs_reporter.report.metrics import (
    compute_data_health,
    compute_evening_window_share,
    lookup_invoice_metric_value,
)
from data_needs_reporter.report.scoring import (
    ScoringWeights,
    compute_confidence,
    compute_score,
    compute_severity,
    compute_source_demand_weights,
    normalize_revenue,
    select_top_actions,
)

try:  # pragma: no cover - optional dependency
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


THEME_TABLE_HINTS: Dict[str, List[str]] = {
    "data_quality": ["dim_customer", "fact_card_transaction"],
    "governance": ["dim_customer", "dim_plan"],
    "analytics": ["fact_card_transaction"],
    "pipeline_health": ["fact_card_transaction", "fact_subscription_invoice"],
    "data_gap": ["fact_card_transaction", "fact_subscription_invoice"],
}
DEFAULT_DEMAND_WEIGHTS: Dict[str, float] = {
    "nlq": 0.50,
    "slack": 0.30,
    "email": 0.20,
}
SUMMARY_MAX_LEN = 160
DEFAULT_EVENING_WINDOW = (17, 21)
INVOICE_DISPLAY_METRICS = (
    "missing_pct",
    "on_time_pct",
    "dup_key_pct",
    "p95_payment_lag_min",
)


def _require_polars() -> "pl":
    if pl is None:  # pragma: no cover
        raise RuntimeError("polars is required for entity extraction pipeline.")
    return pl


def _collect_positive_ids(
    predictions_df: Optional["pl.DataFrame"],
    source_name: str,
    threshold: float,
) -> set:
    if predictions_df is None or predictions_df.height == 0:
        return set()
    positive_ids: set = set()
    for row in predictions_df.to_dicts():
        if row.get("source") != source_name:
            continue
        relevance = row.get("relevance")
        try:
            relevance_val = float(relevance)
        except (TypeError, ValueError):
            relevance_val = 0.0
        if relevance_val < threshold:
            continue
        message_ids = row.get("message_ids") or []
        for message_id in message_ids:
            try:
                positive_ids.add(int(message_id))
            except (TypeError, ValueError):
                continue
    return positive_ids


def _records_from_frame(
    frame: Optional["pl.DataFrame"],
    id_field: str,
    text_field: str,
    source: str,
    include_ids: Optional[Iterable[int]] = None,
) -> Sequence[Mapping[str, object]]:
    if frame is None or frame.height == 0:
        return []
    polars = _require_polars()
    subset = frame
    if include_ids is not None:
        ids = list({int(mid) for mid in include_ids})
        if not ids:
            return []
        subset = frame.filter(polars.col(id_field).is_in(ids))
    records: list[dict[str, object]] = []
    for row in subset.to_dicts():
        record_id = row.get(id_field)
        if record_id is None:
            continue
        body = row.get(text_field) or ""
        if source == "email":
            subject = row.get("subject") or ""
            body = f"{subject} {body}".strip()
        records.append(
            {
                "source": source,
                "record_id": record_id,
                "message_id": record_id if source != "nlq" else None,
                "query_id": record_id if source == "nlq" else None,
                "body": body,
                "text": body if source == "nlq" else None,
            }
        )
    return records


def run_entity_extraction_for_archetype(
    archetype: str,
    llm_client: LLMClient,
    slack_messages: Optional["pl.DataFrame"],
    email_messages: Optional["pl.DataFrame"],
    nlq_messages: Optional["pl.DataFrame"],
    predictions: Optional["pl.DataFrame"],
    out_path: Path,
    *,
    dictionary: Optional[Mapping[str, Sequence[str]]] = None,
    config: Optional[EntityExtractionConfig] = None,
    relevance_threshold: float = 0.5,
) -> Tuple[Sequence[Mapping[str, object]], Dict[str, Dict[str, float]]]:
    dictionary_map = dictionary or build_entity_dictionary(archetype)

    slack_ids = _collect_positive_ids(predictions, "slack", relevance_threshold)
    email_ids = _collect_positive_ids(predictions, "email", relevance_threshold)

    records: list[Mapping[str, object]] = []
    records.extend(
        _records_from_frame(slack_messages, "message_id", "body", "slack", slack_ids)
    )
    records.extend(
        _records_from_frame(email_messages, "message_id", "body", "email", email_ids)
    )
    records.extend(_records_from_frame(nlq_messages, "query_id", "text", "nlq"))

    extraction_config = config or EntityExtractionConfig()
    results, coverage = extract_entities(
        records,
        llm_client,
        dictionary_map,
        config=extraction_config,
        out_path=out_path,
    )
    return results, coverage


def assess_budget_health(
    budget: Mapping[str, Any],
    *,
    coverage_floor: float = 0.2,
) -> Tuple[List[str], bool]:
    """Identify budget warnings and whether they should fail strict mode."""

    warnings: List[str] = []
    strict_failure = False

    coverage = budget.get("coverage")
    if isinstance(coverage, Mapping):
        for source, entry in coverage.items():
            overall = entry.get("overall", {})
            coverage_pct_raw = overall.get("coverage_pct")
            try:
                coverage_pct = float(coverage_pct_raw)
            except (TypeError, ValueError):
                coverage_pct = None

            met_floor = overall.get("met_floor")
            below_floor = (
                coverage_pct is not None and coverage_pct < coverage_floor
            ) or met_floor is False
            if below_floor:
                pct_display = (
                    f"{coverage_pct:.2f}"
                    if coverage_pct is not None
                    else str(coverage_pct_raw)
                )
                warnings.append(
                    f"Coverage below {coverage_floor:.0%} for {source} (coverage={pct_display})."
                )
                strict_failure = True

    if budget.get("stopped_due_to_cap"):
        warnings.append("Generation halted due to spend cap.")
        strict_failure = True

    return warnings, strict_failure


def write_data_health_report(
    config: AppConfig, warehouse_dir: Path, out_dir: Path
) -> Dict[str, object]:
    tz = getattr(getattr(config, "warehouse", object()), "tz", "UTC")
    payload = compute_data_health(Path(warehouse_dir), tz)
    out_path = Path(out_dir)
    tables = payload.get("tables")
    if not isinstance(tables, dict) or not tables:
        payload["tables"] = {
            "dim_customer": {
                "row_count": 0,
                "key_null_pct": 0.0,
                "key_null_pct_daily": [],
                "key_null_spikes": [],
                "fk_success_pct": 100.0,
                "fk_success_pct_daily": [],
                "orphan_pct": 0.0,
                "orphan_pct_daily": [],
                "dup_key_pct": 0.0,
                "dup_key_pct_daily": [],
                "p95_ingest_lag_min": 0.0,
                "p95_ingest_lag_min_daily": [],
            }
        }
    aggregates = payload.get("aggregates")
    if not isinstance(aggregates, dict):
        payload["aggregates"] = {
            "key_null_pct": 0.0,
            "fk_success_pct": 100.0,
            "orphan_pct": 0.0,
            "dup_key_pct": 0.0,
            "p95_ingest_lag_min": 0.0,
        }
    aggregates_by_table = payload.get("aggregates_by_table")
    if not isinstance(aggregates_by_table, dict):
        payload["aggregates_by_table"] = {}
    marketplace_cfg = getattr(getattr(config, "report", object()), "marketplace", None)
    if marketplace_cfg is not None:
        payments_path = Path(warehouse_dir) / "fact_payment.parquet"
        payments_df = _read_parquet_frame(payments_path)
        if payments_df is not None and payments_df.height:
            window_cfg = getattr(marketplace_cfg, "evening_window", None)
            start_hour = getattr(window_cfg, "start_hour", DEFAULT_EVENING_WINDOW[0])
            end_hour = getattr(window_cfg, "end_hour", DEFAULT_EVENING_WINDOW[1])
            min_share_pct = getattr(window_cfg, "min_share_pct", 20.0)
            min_days_pct = getattr(window_cfg, "min_days_pct", 80.0)
            window_days = getattr(
                getattr(config, "report", object()), "window_days", 30
            )
            metrics = compute_evening_window_share(
                payments_df,
                tz=tz,
                window_days=int(window_days or 30),
                start_hour=int(start_hour),
                end_hour=int(end_hour),
                min_share_pct=float(min_share_pct),
                min_days_pct=float(min_days_pct),
            )
            payload["marketplace_evening_window"] = metrics

    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "data_health.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _read_parquet_frame(path: Path) -> Optional["pl.DataFrame"]:
    if not path.exists():
        return None
    polars = _require_polars()
    try:
        return polars.scan_parquet(str(path)).collect(streaming=True)
    except FileNotFoundError:
        return None


def _ensure_datetime(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _theme_tables(theme: str) -> List[str]:
    canonical = theme.strip().lower()
    return THEME_TABLE_HINTS.get(canonical, THEME_TABLE_HINTS.get("data_quality", []))


def _format_example_text(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    return shorten(cleaned, SUMMARY_MAX_LEN, placeholder="…") if cleaned else cleaned


def _get_row_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        if field in row:
            value = row[field]
            if value not in (None, ""):
                return value
    return None


def _estimate_relevance(tokens: Optional[int], text: str) -> float:
    length = max(tokens or 0, max(len(text) // 4, 1))
    base = 0.35 + min(length, 240) / 480.0
    return min(max(base, 0.35), 1.0)


def _estimate_class_conf(tokens: Optional[int]) -> float:
    if tokens is None or tokens <= 0:
        return 0.6
    return min(0.95, max(0.55, 0.45 + min(tokens, 320) / 400.0))


def _build_predictions(
    slack_df: Optional["pl.DataFrame"],
    email_df: Optional["pl.DataFrame"],
    nlq_df: Optional["pl.DataFrame"],
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []

    def _append(
        *,
        theme: str,
        source: str,
        timestamp: Optional[datetime],
        text: str,
        tokens: Optional[int],
    ) -> None:
        predictions.append(
            {
                "theme": (theme or "other").strip().lower(),
                "source": source,
                "timestamp": timestamp,
                "text": text.strip(),
                "relevance": _estimate_relevance(tokens, text),
                "class_conf": _estimate_class_conf(tokens),
            }
        )

    if slack_df is not None and slack_df.height > 0:
        for row in slack_df.to_dicts():
            theme = str(
                _get_row_value(row, "bucket", "theme", "intent") or "data_quality"
            )
            timestamp = _ensure_datetime(
                _get_row_value(row, "sent_at", "ts", "loaded_at", "created_at")
            )
            body = _get_row_value(row, "body", "text")
            if not body:
                thread = row.get("thread_id") or row.get("message_id")
                body = f"Slack thread {thread} discussing {theme}."
            tokens = row.get("tokens")
            _append(
                theme=theme,
                source="slack",
                timestamp=timestamp,
                text=str(body),
                tokens=tokens,
            )

    if email_df is not None and email_df.height > 0:
        for row in email_df.to_dicts():
            theme = str(_get_row_value(row, "bucket", "theme") or "governance")
            timestamp = _ensure_datetime(
                _get_row_value(row, "sent_at", "loaded_at", "ts")
            )
            subject = str(_get_row_value(row, "subject") or "")
            body = str(_get_row_value(row, "body") or "")
            text = (subject + " " + body).strip() or f"Email thread about {theme}."
            tokens = row.get("tokens")
            _append(
                theme=theme,
                source="email",
                timestamp=timestamp,
                text=text,
                tokens=tokens,
            )

    if nlq_df is not None and nlq_df.height > 0:
        for row in nlq_df.to_dicts():
            theme = str(
                _get_row_value(row, "parsed_intent", "bucket", "theme", "intent")
                or "data_gap"
            )
            timestamp = _ensure_datetime(
                _get_row_value(row, "submitted_at", "created_at", "loaded_at")
            )
            text = str(_get_row_value(row, "text") or f"NLQ about {theme}.")
            tokens = row.get("tokens")
            _append(
                theme=theme,
                source="nlq",
                timestamp=timestamp,
                text=text,
                tokens=tokens,
            )

    return [entry for entry in predictions if entry["theme"] and entry["text"]]


def _aggregate_theme_stats(predictions: Sequence[Mapping[str, Any]]) -> Tuple[
    Dict[str, Dict[str, Any]],
    Counter,
]:
    theme_stats: Dict[str, Dict[str, Any]] = {}
    source_counts: Counter = Counter()
    for entry in predictions:
        theme = entry["theme"]
        source = entry["source"]
        source_counts[source] += 1
        stats = theme_stats.setdefault(
            theme,
            {
                "counts": Counter(),
                "relevances": [],
                "timestamps": [],
                "examples": [],
                "sources": set(),
                "class_confs": [],
            },
        )
        stats["counts"][source] += 1
        stats["relevances"].append(float(entry.get("relevance", 0.0) or 0.0))
        stats["class_confs"].append(float(entry.get("class_conf", 0.0) or 0.0))
        timestamp = entry.get("timestamp")
        if isinstance(timestamp, datetime):
            stats["timestamps"].append(timestamp)
        text = entry.get("text")
        if text and len(stats["examples"]) < 3:
            stats["examples"].append({"source": source, "text": text})
        stats["sources"].add(source)
    return theme_stats, source_counts


def _compute_demand_scores(
    theme_stats: Mapping[str, Dict[str, Any]],
    source_counts: Mapping[str, int],
    config: AppConfig,
) -> Dict[str, float]:
    base_weights = getattr(config.report, "demand_base_weights", DEFAULT_DEMAND_WEIGHTS)
    caps = getattr(config.report, "demand_weight_caps", {"min": 0.15, "max": 0.60})
    min_weight = float(caps.get("min", 0.15))
    max_weight = float(caps.get("max", 0.60))
    volumes = {source: int(source_counts.get(source, 0)) for source in base_weights}
    for source, count in source_counts.items():
        volumes.setdefault(source, int(count))
    source_weights = compute_source_demand_weights(
        base_weights, volumes, min_weight, max_weight
    )
    demand_scores: Dict[str, float] = {}
    for theme, stats in theme_stats.items():
        score = 0.0
        for source, total in volumes.items():
            if total <= 0:
                continue
            source_weight = source_weights.get(source, 0.0)
            if source_weight <= 0.0:
                continue
            share = stats["counts"].get(source, 0) / total
            if share <= 0:
                continue
            score += source_weight * share
        demand_scores[theme] = min(max(score, 0.0), 1.0)
    return demand_scores


def _parse_generated_at(value: str) -> datetime:
    if not value:
        return datetime.utcnow().replace(tzinfo=timezone.utc)
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = datetime.utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _compute_recency_scores(
    theme_stats: Mapping[str, Dict[str, Any]],
    reference_dt: datetime,
) -> Dict[str, float]:
    recency: Dict[str, float] = {}
    for theme, stats in theme_stats.items():
        timestamps = stats.get("timestamps") or []
        if not timestamps:
            recency[theme] = 0.0
            continue
        latest = max(timestamps)
        delta_days = max((reference_dt - latest).total_seconds() / 86400.0, 0.0)
        recency[theme] = math.exp(-delta_days / 60.0)
    return recency


def _normalize_tables_payload(tables: object) -> Dict[str, Mapping[str, Any]]:
    normalized: Dict[str, Mapping[str, Any]] = {}
    if isinstance(tables, Mapping):
        iterator = tables.items()
    elif isinstance(tables, list):
        iterator = [
            (entry.get("table"), entry)
            for entry in tables
            if isinstance(entry, Mapping)
        ]
    else:
        iterator = []
    for name, metrics in iterator:
        if not isinstance(name, str) or not isinstance(metrics, Mapping):
            continue
        normalized[name.lower()] = metrics
    return normalized


def _compute_table_severity_map(
    tables: Mapping[str, Mapping[str, Any]],
    slos: Mapping[str, float],
) -> Dict[str, float]:
    severity: Dict[str, float] = {}
    for table, metrics in tables.items():
        overages = {
            "key_null_pct": float(metrics.get("key_null_pct", 0.0) or 0.0),
            "fk_orphan_pct": float(
                metrics.get("fk_orphan_pct", metrics.get("orphan_pct", 0.0)) or 0.0
            ),
            "dup_keys_pct": float(
                metrics.get("dup_keys_pct", metrics.get("dup_key_pct", 0.0)) or 0.0
            ),
            "p95_ingest_lag_min": float(metrics.get("p95_ingest_lag_min", 0.0) or 0.0),
        }
        severity[table] = compute_severity(overages, slos)
    return severity


def _load_rows(path: Path, columns: Sequence[str]) -> List[Mapping[str, Any]]:
    if not path.exists():
        return []
    polars = _require_polars()
    lazy_frame = polars.scan_parquet(str(path))
    if columns:
        available = set(lazy_frame.columns)
        selected = [polars.col(col) for col in columns if col in available]
        if selected:
            lazy_frame = lazy_frame.select(selected)
    df = lazy_frame.collect(streaming=True)
    records = df.to_dicts()
    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing and records:
            for row in records:
                for col in missing:
                    row.setdefault(col, None)
    return records


def _normalized_monthly_score(
    rows: Sequence[Mapping[str, Any]],
    time_key: str,
    value_func,
) -> float:
    totals: Dict[datetime, float] = {}
    for row in rows:
        timestamp = _ensure_datetime(row.get(time_key))
        if timestamp is None:
            continue
        month = datetime(timestamp.year, timestamp.month, 1, tzinfo=timezone.utc)
        totals[month] = totals.get(month, 0.0) + float(value_func(row) or 0.0)
    if not totals:
        return 0.0
    months = sorted(totals.keys())
    current_month = months[-1]
    current_value = totals[current_month]
    series = [totals[month] for month in months]
    normalized = normalize_revenue(series, current_value)
    return min(max(normalized / 2.0, 0.0), 1.0)


def _compute_table_revenue_scores(
    warehouse_path: Path,
    attach_targets: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    txn_rows = _load_rows(
        warehouse_path / "fact_card_transaction.parquet",
        ["txn_id", "event_time", "amount_cents", "interchange_bps"],
    )

    def _txn_value(row: Mapping[str, Any]) -> float:
        amount_cents = float(row.get("amount_cents") or 0.0)
        interchange_bps = float(row.get("interchange_bps") or 0.0)
        return (amount_cents / 100.0) * (interchange_bps / 10000.0)

    scores["fact_card_transaction"] = _normalized_monthly_score(
        txn_rows, "event_time", _txn_value
    )

    invoice_rows = _load_rows(
        warehouse_path / "fact_subscription_invoice.parquet",
        ["invoice_id", "paid_at", "amount_cents", "tier"],
    )

    target_map: Dict[str, float] = {}
    if attach_targets:
        for key, value in attach_targets.items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric <= 0.0:
                continue
            if numeric > 1.0:
                numeric = numeric / 100.0
            if numeric <= 0.0:
                continue
            target_map[key.strip().lower()] = numeric

    def _invoice_value(row: Mapping[str, Any]) -> float:
        amount_cents = float(row.get("amount_cents") or 0.0)
        tier = str(row.get("tier") or "").strip().lower()
        if tier and tier in target_map:
            target = max(target_map[tier], 1e-6)
            return (amount_cents / 100.0) / target
        return amount_cents / 100.0

    scores["fact_subscription_invoice"] = _normalized_monthly_score(
        invoice_rows, "paid_at", _invoice_value
    )
    return scores


def _assemble_actions(
    theme_stats: Mapping[str, Dict[str, Any]],
    demand_scores: Mapping[str, float],
    recency_scores: Mapping[str, float],
    table_severity: Mapping[str, float],
    table_revenue: Mapping[str, float],
    source_counts: Mapping[str, int],
    config: AppConfig,
) -> List[Dict[str, Any]]:
    scoring_weights = getattr(config.report, "scoring_weights", {})
    weights = ScoringWeights(
        revenue=float(scoring_weights.get("revenue", 0.6)),
        demand=float(scoring_weights.get("demand", 0.25)),
        severity=float(scoring_weights.get("severity", 0.15)),
    )
    total_sources = max(len(source_counts) or len(DEFAULT_DEMAND_WEIGHTS), 1)
    global_severity = (
        sum(table_severity.values()) / len(table_severity) if table_severity else 0.0
    )
    global_revenue = (
        sum(table_revenue.values()) / len(table_revenue) if table_revenue else 0.0
    )
    actions: List[Dict[str, Any]] = []
    for theme, stats in theme_stats.items():
        theme_tables = _theme_tables(theme) or ["fact_card_transaction"]
        severity_values = [
            table_severity.get(table, global_severity) for table in theme_tables
        ]
        severity_score = max(severity_values) if severity_values else global_severity
        revenue_values = [
            table_revenue.get(table, global_revenue) for table in theme_tables
        ]
        revenue_score = (
            sum(revenue_values) / len(revenue_values)
            if revenue_values
            else global_revenue
        )
        demand_score = demand_scores.get(theme, 0.0)
        recency = recency_scores.get(theme, 0.0)
        score = compute_score(revenue_score, demand_score, severity_score, weights)

        class_conf = (
            sum(stats["class_confs"]) / len(stats["class_confs"])
            if stats["class_confs"]
            else 0.0
        )
        entity_conf = min(1.0, len(theme_tables) / 3.0)
        coverage = demand_score
        agreement = len(stats["sources"]) / total_sources
        confidence = compute_confidence(class_conf, entity_conf, coverage, agreement)

        examples = [
            {
                "source": example["source"],
                "text": _format_example_text(example["text"]),
            }
            for example in stats["examples"][:2]
        ]
        summary = (
            examples[0]["text"] if examples else "No recent demand examples available."
        )

        actions.append(
            {
                "theme": theme,
                "demand": round(demand_score, 4),
                "revenue": round(revenue_score, 4),
                "severity": round(severity_score, 4),
                "recency": round(recency, 3),
                "score": round(min(max(score, 0.0), 1.0), 4),
                "confidence": round(min(max(confidence, 0.0), 1.0), 4),
                "examples": examples,
                "summary": summary,
            }
        )
    return sorted(actions, key=lambda item: item["score"], reverse=True)


def _format_invoice_value(metric: str, value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if metric == "row_count":
        return f"{int(value)}"
    if metric.endswith("_min") or metric.endswith("_minutes"):
        return f"{value:.2f} min"
    return f"{value:.2f}%"


def _summarize_invoice_aggregates(
    config: AppConfig, data_health: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    invoice_cfg = getattr(config.report, "invoice_aggregates", None)
    aggregates_section = data_health.get("aggregates_by_table")
    if not isinstance(aggregates_section, Mapping):
        return None
    invoice_entry = aggregates_section.get("fact_subscription_invoice")
    if not isinstance(invoice_entry, Mapping) or not invoice_entry:
        return None

    summary: Dict[str, Any] = {"metrics": dict(invoice_entry)}
    summary["enabled"] = (
        bool(getattr(invoice_cfg, "enabled", False)) if invoice_cfg else False
    )

    thresholds: Dict[str, float] = {}
    invoice_slos_cfg = getattr(invoice_cfg, "slos", None) if invoice_cfg else None
    if invoice_slos_cfg is not None:
        if hasattr(invoice_slos_cfg, "model_dump"):
            raw_slos = invoice_slos_cfg.model_dump(exclude_none=True)
        elif isinstance(invoice_slos_cfg, Mapping):
            raw_slos = {k: v for k, v in invoice_slos_cfg.items() if v is not None}
        else:
            raw_slos = {}
        for key, value in raw_slos.items():
            try:
                thresholds[key] = float(value)
            except (TypeError, ValueError):
                continue

    if not thresholds:
        return summary

    checks: List[Dict[str, Any]] = []
    passed = True
    for key, threshold in thresholds.items():
        comparator, metric_name, value = lookup_invoice_metric_value(
            summary["metrics"], key
        )
        if comparator == "min":
            metric_passed = value is not None and value >= threshold
        else:
            metric_passed = value is not None and value <= threshold
        checks.append(
            {
                "name": key,
                "metric": metric_name,
                "comparator": comparator,
                "threshold": threshold,
                "value": value,
                "passed": metric_passed,
            }
        )
        if not metric_passed:
            passed = False

    summary["slos"] = thresholds
    summary["checks"] = checks
    summary["passed"] = passed
    return summary


def _summarize_evening_window(
    config: AppConfig, data_health: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    marketplace_cfg = getattr(getattr(config, "report", object()), "marketplace", None)
    metrics = data_health.get("marketplace_evening_window")
    if not isinstance(metrics, Mapping) or not marketplace_cfg:
        return None
    summary = dict(metrics)
    evening_cfg = getattr(marketplace_cfg, "evening_window", None)
    if evening_cfg is not None:
        summary.setdefault(
            "threshold_share_pct", getattr(evening_cfg, "min_share_pct", 20.0)
        )
        summary.setdefault(
            "threshold_days_pct", getattr(evening_cfg, "min_days_pct", 80.0)
        )
        summary.setdefault(
            "start_hour", getattr(evening_cfg, "start_hour", DEFAULT_EVENING_WINDOW[0])
        )
        summary.setdefault(
            "end_hour", getattr(evening_cfg, "end_hour", DEFAULT_EVENING_WINDOW[1])
        )
    return summary


def write_exec_summary(
    config: AppConfig,
    warehouse_path: Path,
    comms_path: Path,
    out_dir: Path,
    data_health: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    polars = _require_polars()
    _ = polars  # quiet linters
    slack_df = _read_parquet_frame(comms_path / "slack_messages.parquet")
    email_df = _read_parquet_frame(comms_path / "email_messages.parquet")
    nlq_df = _read_parquet_frame(comms_path / "nlq.parquet")
    predictions = _build_predictions(slack_df, email_df, nlq_df)
    theme_stats, source_counts = _aggregate_theme_stats(predictions)
    demand_scores = _compute_demand_scores(theme_stats, source_counts, config)
    latest_timestamp: Optional[datetime] = None
    for stats in theme_stats.values():
        if stats["timestamps"]:
            candidate = max(stats["timestamps"])
            if latest_timestamp is None or candidate > latest_timestamp:
                latest_timestamp = candidate
    reference_dt = latest_timestamp or _parse_generated_at(generated_at)
    recency_scores = _compute_recency_scores(theme_stats, reference_dt)
    tables_payload = _normalize_tables_payload(data_health.get("tables"))
    slos_model = getattr(config.report, "slos", None)
    if slos_model is None:
        slos_dict: Mapping[str, float] = {}
    elif hasattr(slos_model, "model_dump"):
        slos_dict = slos_model.model_dump()
    elif isinstance(slos_model, Mapping):
        slos_dict = slos_model
    else:
        slos_dict = {}
    table_severity = _compute_table_severity_map(tables_payload, slos_dict)
    monetization_cfg = getattr(config.report, "monetization", None)
    attach_targets = None
    if monetization_cfg is not None:
        if hasattr(monetization_cfg, "attach_targets"):
            attach_targets = getattr(monetization_cfg, "attach_targets", None)
        elif isinstance(monetization_cfg, Mapping):
            attach_targets = monetization_cfg.get("attach_targets")
    table_revenue = _compute_table_revenue_scores(
        warehouse_path, attach_targets=attach_targets
    )
    actions = _assemble_actions(
        theme_stats,
        demand_scores,
        recency_scores,
        table_severity,
        table_revenue,
        source_counts,
        config,
    )
    if not actions:
        fallback_themes = ["data_quality", "pipeline_health", "governance"]
        for theme in fallback_themes:
            if theme in theme_stats:
                continue
            actions.append(
                {
                    "theme": theme,
                    "demand": 0.0,
                    "revenue": 0.0,
                    "severity": 0.0,
                    "recency": 0.0,
                    "score": 0.0,
                    "confidence": 0.55,
                    "examples": [],
                    "summary": "No demand available.",
                }
            )
        actions = actions[:3]

    top_actions = select_top_actions(actions, k=3, min_diversity=3)
    if len(top_actions) < min(3, len(actions)):
        seen = {id(item) for item in top_actions}
        for action in actions:
            if id(action) in seen:
                continue
            top_actions.append(action)
            if len(top_actions) >= 3:
                break
    top_actions = top_actions[:3]
    if len(top_actions) < 3:
        placeholder_order = ["data_quality", "pipeline_health", "governance"]
        taken = {action["theme"] for action in top_actions}
        for theme in placeholder_order:
            if len(top_actions) >= 3:
                break
            if theme in taken:
                continue
            top_actions.append(
                {
                    "theme": theme,
                    "demand": 0.0,
                    "revenue": 0.0,
                    "severity": 0.0,
                    "recency": 0.0,
                    "score": 0.0,
                    "confidence": 0.55,
                    "examples": [],
                    "summary": "Insufficient demand to prioritize.",
                }
            )

    themes_payload = {"themes": actions}
    themes_path = out_dir / "themes.json"
    themes_path.write_text(json.dumps(themes_payload, indent=2), encoding="utf-8")

    themes_md_lines = ["# Top Themes"]
    for action in actions:
        themes_md_lines.append(
            f"## {action['theme'].title()}\n"
            f"Score: {action['score']:.2f} | Confidence: {action['confidence']:.2f}\n"
            f"- Demand: {action['demand']:.2f}\n"
            f"- Revenue: {action['revenue']:.2f}\n"
            f"- Severity: {action['severity']:.2f}\n"
            f"- Recency: {action['recency']:.2f}\n"
            f"Summary: {action['summary']}"
        )
    (out_dir / "themes.md").write_text("\n\n".join(themes_md_lines), encoding="utf-8")

    exec_summary = {
        "generated_at": generated_at,
        "window_days": getattr(config.report, "window_days", 30),
        "top_actions": top_actions,
        "notes": "Synthesized demand, severity, and revenue scores from generated data.",
    }
    invoice_summary = _summarize_invoice_aggregates(config, data_health)
    if invoice_summary:
        exec_summary["invoice_aggregates"] = invoice_summary
    evening_summary = _summarize_evening_window(config, data_health)
    if evening_summary:
        exec_summary["marketplace_evening_window"] = evening_summary
    exec_json = out_dir / "exec_summary.json"
    exec_json.write_text(json.dumps(exec_summary, indent=2), encoding="utf-8")

    exec_md_lines = ["# Executive Summary", f"Generated: {generated_at}"]
    for idx, action in enumerate(top_actions, 1):
        line = (
            f"{idx}. **{action['theme'].title()}** — "
            f"score {action['score']:.2f}, demand {action['demand']:.2f}, "
            f"severity {action['severity']:.2f}"
        )
        exec_md_lines.append(line)
        if action["examples"]:
            exec_md_lines.append(f"   e.g., {action['examples'][0]['text']}")
    if invoice_summary:
        exec_md_lines.append("")
        exec_md_lines.append("## Invoice Aggregates")
        checks = invoice_summary.get("checks") or []
        if checks:
            for check in checks:
                metric_name = check.get("metric") or check.get("name")
                value_text = _format_invoice_value(metric_name, check.get("value"))
                threshold_text = _format_invoice_value(
                    metric_name, check.get("threshold")
                )
                symbol = "≥" if check.get("comparator") == "min" else "≤"
                status = "PASS" if check.get("passed") else "FAIL"
                exec_md_lines.append(
                    f"- {metric_name}: {value_text} "
                    f"(limit {threshold_text}, {symbol}) — {status}"
                )
        else:
            metrics = invoice_summary.get("metrics", {})
            metric_lines = []
            for name in INVOICE_DISPLAY_METRICS:
                if name in metrics:
                    metric_lines.append(
                        f"{name}: {_format_invoice_value(name, metrics.get(name))}"
                    )
            if metric_lines:
                exec_md_lines.append("Metrics:")
                for entry in metric_lines:
                    exec_md_lines.append(f"  - {entry}")
    if evening_summary:
        exec_md_lines.append("")
        exec_md_lines.append("## Marketplace Evening Coverage")
        share_limit = evening_summary.get("threshold_share_pct", 0.0)
        days_limit = evening_summary.get("threshold_days_pct", 0.0)
        overall_value = evening_summary.get("overall_share_pct")
        days_value = evening_summary.get("days_pct")
        exec_md_lines.append(
            f"- Overall Share: {_format_invoice_value('overall_share_pct', overall_value)} "
            f"(limit {_format_invoice_value('overall_share_pct', share_limit)}, ≥)"
        )
        exec_md_lines.append(
            f"- Qualifying Days: {_format_invoice_value('days_pct', days_value)} "
            f"(limit {_format_invoice_value('days_pct', days_limit)}, ≥)"
        )
        exec_md_lines.append(
            f"- Window: {int(evening_summary.get('start_hour', DEFAULT_EVENING_WINDOW[0])):02d}:00–"
            f"{int(evening_summary.get('end_hour', DEFAULT_EVENING_WINDOW[1])):02d}:59 "
            f"over {int(evening_summary.get('window_days', 0))} days"
        )
    (out_dir / "exec_summary.md").write_text(
        "\n\n".join(exec_md_lines), encoding="utf-8"
    )
    return exec_summary


__all__ = [
    "build_entity_dictionary",
    "assess_budget_health",
    "run_entity_extraction_for_archetype",
    "write_data_health_report",
    "write_exec_summary",
]
