from __future__ import annotations

import math
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

from data_needs_reporter.config import AppConfig
from data_needs_reporter.report.llm import RepairingLLMClient
from data_needs_reporter.utils.cost_guard import CostGuard
from data_needs_reporter.utils.io import write_parquet_atomic

try:  # pragma: no cover - optional dependency
    import polars as _pl
except ImportError:  # pragma: no cover
    _pl = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import polars as pl  # noqa: F401

pl = _pl  # type: ignore[assignment]

_URL_PATTERN = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
ALLOWED_LINK_DOMAINS = {
    "looker",
    "dbt",
    "fivetran",
    "snowflake",
    "airflow",
    "montecarlo",
}
BLOCKED_LINK_PLACEHOLDER = "[blocked-link]"


def _ensure_polars():
    if pl is None:
        raise RuntimeError("polars is required for communications generation.")
    return pl


def _canonical_domain(host: str) -> str | None:
    if not host:
        return None
    normalized = host.lower()
    if normalized.startswith("www."):
        normalized = normalized[4:]
    parts = normalized.split(".")
    for allowed in ALLOWED_LINK_DOMAINS:
        if allowed in parts:
            return allowed
    return None


def _sanitize_links(text: str) -> tuple[str, List[str]]:
    if not text:
        return text, []

    domains: List[str] = []

    def _replace(match: re.Match[str]) -> str:
        url = match.group(0)
        host = urlparse(url).hostname or ""
        canonical = _canonical_domain(host)
        if canonical:
            domains.append(canonical)
            return url
        return BLOCKED_LINK_PLACEHOLDER

    sanitized = _URL_PATTERN.sub(_replace, text)
    return sanitized, domains


COMM_USER_ROLE_MIX: Dict[str, float] = {
    "executive": 0.12,
    "product": 0.26,
    "analytics": 0.24,
    "engineering": 0.22,
    "go_to_market": 0.16,
}


def write_empty_comms(out_dir: Path) -> None:
    polars = _ensure_polars()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    slack_schema: Iterable[Tuple[str, "pl.DataType"]] = (
        ("message_id", pl.Int64),
        ("thread_id", pl.Int64),
        ("user_id", pl.Int64),
        ("sent_at", pl.Datetime(time_zone="UTC")),
        ("channel", pl.Utf8),
        ("bucket", pl.Utf8),
        ("body", pl.Utf8),
        ("tokens", pl.Int32),
        ("link_domains", pl.List(pl.Utf8)),
        ("loaded_at", pl.Datetime(time_zone="UTC")),
    )

    email_schema: Iterable[Tuple[str, "pl.DataType"]] = (
        ("message_id", pl.Int64),
        ("thread_id", pl.Int64),
        ("sender_id", pl.Int64),
        ("recipient_ids", pl.List(pl.Int64)),
        ("subject", pl.Utf8),
        ("body", pl.Utf8),
        ("sent_at", pl.Datetime(time_zone="UTC")),
        ("bucket", pl.Utf8),
        ("tokens", pl.Int32),
        ("link_domains", pl.List(pl.Utf8)),
        ("loaded_at", pl.Datetime(time_zone="UTC")),
    )

    nlq_schema: Iterable[Tuple[str, "pl.DataType"]] = (
        ("query_id", pl.Int64),
        ("user_id", pl.Int64),
        ("submitted_at", pl.Datetime(time_zone="UTC")),
        ("text", pl.Utf8),
        ("parsed_intent", pl.Utf8),
        ("tokens", pl.Int32),
        ("loaded_at", pl.Datetime(time_zone="UTC")),
    )

    users_schema: Iterable[Tuple[str, "pl.DataType"]] = (
        ("user_id", pl.Int64),
        ("role", pl.Utf8),
        ("department", pl.Utf8),
        ("time_zone", pl.Utf8),
        ("active", pl.Boolean),
    )

    _write_empty(out_path / "slack_messages.parquet", slack_schema, polars)
    _write_empty(out_path / "email_messages.parquet", email_schema, polars)
    _write_empty(out_path / "nlq.parquet", nlq_schema, polars)
    _write_empty(out_path / "comms_users.parquet", users_schema, polars)


def _write_empty(
    path: Path, schema: Iterable[Tuple[str, "pl.DataType"]], polars
) -> None:
    df = polars.DataFrame(
        [polars.Series(name, [], dtype=dtype) for name, dtype in schema]
    )
    write_parquet_atomic(path, df)


def generate_comms(
    cfg: AppConfig,
    archetype: str,
    out_dir: Path,
    llm_client: RepairingLLMClient,
    guard: CostGuard,
    seed: int | None = None,
) -> Dict[str, Any]:
    polars = _ensure_polars()
    rng = random.Random(seed if seed is not None else cfg.comms.seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    users_df = _generate_users(polars, rng)
    write_parquet_atomic(out_path / "comms_users.parquet", users_df)
    user_ids = users_df["user_id"].to_list()

    time_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    time_end = time_start + timedelta(days=365)

    slack_stats = _generate_slack_messages(
        cfg, rng, llm_client, guard, out_path, user_ids, time_start, time_end
    )
    email_stats = _generate_email_messages(
        cfg, rng, llm_client, guard, out_path, user_ids, time_start, time_end
    )
    nlq_stats = _generate_nlq_queries(
        cfg, rng, llm_client, guard, out_path, user_ids, time_start, time_end
    )

    summary = {
        "slack_messages": slack_stats["count"],
        "email_messages": email_stats["count"],
        "nlq": nlq_stats["count"],
    }

    targets = {
        "slack": cfg.comms.slack_threads,
        "email": cfg.comms.email_threads,
        "nlq": cfg.comms.nlq,
    }

    estimates = _estimate_tokens(
        {
            "slack": slack_stats["tokens_list"],
            "email": email_stats["tokens_list"],
            "nlq": nlq_stats["tokens_list"],
        },
        targets,
        cfg.budget.safety_margin,
        guard,
    )

    quotas = _build_quotas(
        {
            "slack": slack_stats["day_bucket"],
            "email": email_stats["day_bucket"],
            "nlq": nlq_stats["day_bucket"],
        }
    )

    coverage = _coverage_report(
        {
            "slack": slack_stats["count"],
            "email": email_stats["count"],
            "nlq": nlq_stats["count"],
        },
        targets,
        {
            "slack": slack_stats["day_bucket"],
            "email": email_stats["day_bucket"],
            "nlq": nlq_stats["day_bucket"],
        },
        quotas,
    )

    guard.write_budget(
        out_path / "budget.json",
        extra={
            "messages": summary,
            "archetype": archetype,
            "estimate": estimates,
            "actual": {
                "slack": {
                    "count": slack_stats["count"],
                    "tokens": guard._channel_usage.get("slack", {}).get("tokens", 0),
                },
                "email": {
                    "count": email_stats["count"],
                    "tokens": guard._channel_usage.get("email", {}).get("tokens", 0),
                },
                "nlq": {
                    "count": nlq_stats["count"],
                    "tokens": guard._channel_usage.get("nlq", {}).get("tokens", 0),
                },
            },
            "coverage": coverage,
            "quotas": quotas,
        },
    )
    return summary


def _generate_users(polars, rng: random.Random, total_users: int = 200):
    rows: List[Dict[str, Any]] = []
    user_id = 1
    remaining = total_users
    roles = list(COMM_USER_ROLE_MIX.items())
    for index, (role, share) in enumerate(roles):
        if index == len(roles) - 1:
            role_count = remaining
        else:
            role_count = max(1, int(total_users * share))
            remaining -= role_count
        for _ in range(role_count):
            rows.append(
                {
                    "user_id": user_id,
                    "role": role,
                    "department": role.replace("_", " ").title(),
                    "time_zone": rng.choice(
                        ["America/Los_Angeles", "America/New_York", "UTC"]
                    ),
                    "active": rng.random() > 0.05,
                }
            )
            user_id += 1
    return polars.DataFrame(rows)


def _generate_slack_messages(
    cfg: AppConfig,
    rng: random.Random,
    llm_client: RepairingLLMClient,
    guard: CostGuard,
    out_path: Path,
    user_ids: List[int],
    start_time: datetime,
    end_time: datetime,
) -> Dict[str, Any]:
    polars = _ensure_polars()
    rows: List[Dict[str, Any]] = []
    message_id = 1
    channels = ["#data-quality", "#analytics", "#product-insights", "#data-eng"]
    buckets = ["data_quality", "governance", "analytics", "pipeline_health"]

    target = cfg.comms.slack_threads
    tokens_used = 0
    tokens_list: List[int] = []
    day_bucket: Dict[str, Dict[str, int]] = {}
    for _ in range(target):
        if guard.stopped_due_to_cap:
            break
        sent_at = _random_ts(rng, start_time, end_time)
        prompt = (
            "Generate a concise Slack message summarizing a data issue. "
            "Include bucket and estimated token usage."
        )
        response = llm_client.json_complete(prompt, temperature=0.2)
        content = response or {}
        raw_body = content.get("body", "Investigated anomaly in revenue dashboard.")
        body, link_domains = _sanitize_links(raw_body)
        bucket = content.get("bucket", rng.choice(buckets))
        tokens = int(content.get("tokens", max(20, len(body) // 4)))
        if not guard.record_message("slack", tokens):
            break
        tokens_used += tokens
        tokens_list.append(tokens)
        day_key = sent_at.date().isoformat()
        bucket_counts = day_bucket.setdefault(day_key, {})
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        rows.append(
            {
                "message_id": message_id,
                "thread_id": message_id,
                "user_id": rng.choice(user_ids),
                "sent_at": sent_at,
                "channel": rng.choice(channels),
                "bucket": bucket,
                "body": body,
                "tokens": tokens,
                "link_domains": link_domains,
                "loaded_at": sent_at + timedelta(minutes=rng.randint(2, 20)),
            }
        )
        message_id += 1

    df = (
        polars.DataFrame(rows)
        if rows
        else polars.DataFrame([polars.Series("message_id", [], dtype=polars.Int64)])
    )
    if rows:
        df = df.select(
            [
                polars.col("message_id").cast(polars.Int64),
                polars.col("thread_id").cast(polars.Int64),
                polars.col("user_id").cast(polars.Int64),
                polars.col("sent_at").cast(polars.Datetime(time_zone="UTC")),
                polars.col("channel").cast(polars.Utf8),
                polars.col("bucket").cast(polars.Utf8),
                polars.col("body").cast(polars.Utf8),
                polars.col("tokens").cast(polars.Int32),
                polars.col("link_domains").cast(polars.List(polars.Utf8)),
                polars.col("loaded_at").cast(polars.Datetime(time_zone="UTC")),
            ]
        )
    else:
        df = polars.DataFrame(
            [
                polars.Series("message_id", [], dtype=polars.Int64),
                polars.Series("thread_id", [], dtype=polars.Int64),
                polars.Series("user_id", [], dtype=polars.Int64),
                polars.Series("sent_at", [], dtype=polars.Datetime(time_zone="UTC")),
                polars.Series("channel", [], dtype=polars.Utf8),
                polars.Series("bucket", [], dtype=polars.Utf8),
                polars.Series("body", [], dtype=polars.Utf8),
                polars.Series("tokens", [], dtype=polars.Int32),
                polars.Series("link_domains", [], dtype=polars.List(polars.Utf8)),
                polars.Series("loaded_at", [], dtype=polars.Datetime(time_zone="UTC")),
            ]
        )
    write_parquet_atomic(out_path / "slack_messages.parquet", df)
    return {
        "count": df.height,
        "tokens": tokens_used,
        "tokens_list": tokens_list,
        "day_bucket": day_bucket,
    }


def _generate_email_messages(
    cfg: AppConfig,
    rng: random.Random,
    llm_client: RepairingLLMClient,
    guard: CostGuard,
    out_path: Path,
    user_ids: List[int],
    start_time: datetime,
    end_time: datetime,
) -> Dict[str, Any]:
    polars = _ensure_polars()
    rows: List[Dict[str, Any]] = []
    thread_id = 1

    target = cfg.comms.email_threads
    tokens_used = 0
    tokens_list: List[int] = []
    day_bucket: Dict[str, Dict[str, int]] = {}
    for _ in range(target):
        if guard.stopped_due_to_cap:
            break
        sent_at = _random_ts(rng, start_time, end_time)
        prompt = "Produce an email summary of a data need with subject, body, bucket, tokens."
        response = llm_client.json_complete(prompt, temperature=0.1)
        content = response or {}
        raw_body = content.get("body", "Requesting deeper analysis on retention funnel.")
        body, link_domains = _sanitize_links(raw_body)
        subject = content.get("subject", "Follow-up on data health")
        bucket = content.get("bucket", "data_quality")
        tokens = int(content.get("tokens", max(25, len(body) // 4)))
        if not guard.record_message("email", tokens):
            break
        tokens_used += tokens
        tokens_list.append(tokens)
        day_key = sent_at.date().isoformat()
        bucket_counts = day_bucket.setdefault(day_key, {})
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        sender = rng.choice(user_ids)
        recipients = rng.sample(user_ids, k=min(3, len(user_ids)))
        rows.append(
            {
                "message_id": thread_id,
                "thread_id": thread_id,
                "sender_id": sender,
                "recipient_ids": recipients,
                "subject": subject,
                "body": body,
                "sent_at": sent_at,
                "bucket": bucket,
                "tokens": tokens,
                "link_domains": link_domains,
                "loaded_at": sent_at + timedelta(minutes=rng.randint(10, 45)),
            }
        )
        thread_id += 1

    if rows:
        df = polars.DataFrame(rows).select(
            [
                polars.col("message_id").cast(polars.Int64),
                polars.col("thread_id").cast(polars.Int64),
                polars.col("sender_id").cast(polars.Int64),
                polars.col("recipient_ids").cast(polars.List(polars.Int64)),
                polars.col("subject").cast(polars.Utf8),
                polars.col("body").cast(polars.Utf8),
                polars.col("sent_at").cast(polars.Datetime(time_zone="UTC")),
                polars.col("bucket").cast(polars.Utf8),
                polars.col("tokens").cast(polars.Int32),
                polars.col("link_domains").cast(polars.List(polars.Utf8)),
                polars.col("loaded_at").cast(polars.Datetime(time_zone="UTC")),
            ]
        )
    else:
        df = polars.DataFrame(
            [
                polars.Series("message_id", [], dtype=polars.Int64),
                polars.Series("thread_id", [], dtype=polars.Int64),
                polars.Series("sender_id", [], dtype=polars.Int64),
                polars.Series("recipient_ids", [], dtype=polars.List(polars.Int64)),
                polars.Series("subject", [], dtype=polars.Utf8),
                polars.Series("body", [], dtype=polars.Utf8),
                polars.Series("sent_at", [], dtype=polars.Datetime(time_zone="UTC")),
                polars.Series("bucket", [], dtype=polars.Utf8),
                polars.Series("tokens", [], dtype=polars.Int32),
                polars.Series("link_domains", [], dtype=polars.List(polars.Utf8)),
                polars.Series("loaded_at", [], dtype=polars.Datetime(time_zone="UTC")),
            ]
        )
    write_parquet_atomic(out_path / "email_messages.parquet", df)
    return {
        "count": df.height,
        "tokens": guard._channel_usage.get("email", {}).get("tokens", 0),
        "tokens_list": tokens_list,
        "day_bucket": day_bucket,
    }


def _generate_nlq_queries(
    cfg: AppConfig,
    rng: random.Random,
    llm_client: RepairingLLMClient,
    guard: CostGuard,
    out_path: Path,
    user_ids: List[int],
    start_time: datetime,
    end_time: datetime,
) -> Dict[str, Any]:
    polars = _ensure_polars()
    rows: List[Dict[str, Any]] = []
    target = cfg.comms.nlq
    tokens_used = 0
    tokens_list: List[int] = []
    day_bucket: Dict[str, Dict[str, int]] = {}
    for query_id in range(1, target + 1):
        if guard.stopped_due_to_cap:
            break
        submitted_at = _random_ts(rng, start_time, end_time)
        prompt = "Produce an NLQ text about data needs with intent label and tokens."
        response = llm_client.json_complete(prompt, temperature=0.0)
        content = response or {}
        text = content.get("text", "How many active subscribers churned this month?")
        intent = content.get("parsed_intent", "data_gap")
        tokens = int(content.get("tokens", max(15, len(text) // 4)))
        if not guard.record_message("nlq", tokens):
            break
        tokens_used += tokens
        tokens_list.append(tokens)
        day_key = submitted_at.date().isoformat()
        bucket_counts = day_bucket.setdefault(day_key, {})
        bucket_counts[intent] = bucket_counts.get(intent, 0) + 1
        rows.append(
            {
                "query_id": query_id,
                "user_id": rng.choice(user_ids),
                "submitted_at": submitted_at,
                "text": text,
                "parsed_intent": intent,
                "tokens": tokens,
                "loaded_at": submitted_at + timedelta(minutes=rng.randint(1, 5)),
            }
        )

    if rows:
        df = polars.DataFrame(rows).select(
            [
                polars.col("query_id").cast(polars.Int64),
                polars.col("user_id").cast(polars.Int64),
                polars.col("submitted_at").cast(polars.Datetime(time_zone="UTC")),
                polars.col("text").cast(polars.Utf8),
                polars.col("parsed_intent").cast(polars.Utf8),
                polars.col("tokens").cast(polars.Int32),
                polars.col("loaded_at").cast(polars.Datetime(time_zone="UTC")),
            ]
        )
    else:
        df = polars.DataFrame(
            [
                polars.Series("query_id", [], dtype=polars.Int64),
                polars.Series("user_id", [], dtype=polars.Int64),
                polars.Series(
                    "submitted_at", [], dtype=polars.Datetime(time_zone="UTC")
                ),
                polars.Series("text", [], dtype=polars.Utf8),
                polars.Series("parsed_intent", [], dtype=polars.Utf8),
                polars.Series("tokens", [], dtype=polars.Int32),
                polars.Series("loaded_at", [], dtype=polars.Datetime(time_zone="UTC")),
            ]
        )
    write_parquet_atomic(out_path / "nlq.parquet", df)
    return {
        "count": df.height,
        "tokens": guard._channel_usage.get("nlq", {}).get("tokens", 0),
        "tokens_list": tokens_list,
        "day_bucket": day_bucket,
    }


def _random_ts(rng: random.Random, start: datetime, end: datetime) -> datetime:
    delta = end - start
    seconds = rng.randrange(int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def _estimate_tokens(
    tokens_by_source: Mapping[str, Sequence[int]],
    targets: Mapping[str, int],
    safety_margin: float,
    guard: CostGuard,
) -> Dict[str, Any]:
    estimates: Dict[str, Any] = {}
    total_tokens = 0
    for source, tokens in tokens_by_source.items():
        values = list(tokens)
        if values:
            values.sort()
            idx = max(0, int(math.ceil(0.9 * len(values)) - 1))
            p90 = values[idx]
        else:
            p90 = 0
        planned = targets.get(source, len(values))
        planned_tokens = (
            int(math.ceil(p90 * planned * (1 + safety_margin))) if p90 else 0
        )
        estimates[source] = {
            "p90_tokens": p90,
            "planned_messages": planned,
            "planned_tokens": planned_tokens,
        }
        total_tokens += planned_tokens
    total_cost = round(total_tokens * (guard.price_per_1k_tokens / 1000.0), 4)
    estimates["total_tokens"] = total_tokens
    estimates["total_cost_usd"] = total_cost
    return estimates


def _aggregate_bucket_totals(day_map: Mapping[str, Dict[str, int]]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for bucket_counts in day_map.values():
        for bucket, count in bucket_counts.items():
            totals[bucket] = totals.get(bucket, 0) + count
    return totals


def _build_quotas(
    day_bucket_counts: Mapping[str, Dict[str, Dict[str, int]]]
) -> Dict[str, Any]:
    quotas: Dict[str, Any] = {}
    for source, day_map in day_bucket_counts.items():
        bucket_totals = _aggregate_bucket_totals(day_map)
        total = sum(bucket_totals.values())
        quotas[source] = {
            "total": total,
            "day_bucket": day_map,
            "bucket_totals": bucket_totals,
        }
    return quotas


def _coverage_report(
    actual_counts: Mapping[str, int],
    targets: Mapping[str, int],
    day_bucket_counts: Mapping[str, Dict[str, Dict[str, int]]],
    quotas: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for source, actual in actual_counts.items():
        target = targets.get(source, 0)
        coverage_pct = actual / target if target else 1.0
        met_floor = coverage_pct >= 0.2 if target else True
        behavior = "continue"
        if not met_floor:
            behavior = "behavior_c_continue"

        per_bucket: Dict[str, Any] = {}
        bucket_actuals = _aggregate_bucket_totals(day_bucket_counts.get(source, {}))
        bucket_targets = quotas.get(source, {}).get("bucket_totals", {})
        bucket_keys = set(bucket_actuals) | set(bucket_targets)
        for bucket in sorted(bucket_keys):
            actual_bucket = bucket_actuals.get(bucket, 0)
            target_bucket = bucket_targets.get(bucket, 0)
            bucket_cov = (
                actual_bucket / target_bucket if target_bucket else 1.0
            )
            per_bucket[bucket] = {
                "actual": actual_bucket,
                "target": target_bucket,
                "coverage_pct": round(bucket_cov, 3),
            }

        report[source] = {
            "overall": {
                "actual": actual,
                "target": target,
                "coverage_pct": round(coverage_pct, 3),
                "met_floor": met_floor,
                "behavior": behavior,
            },
            "per_bucket": per_bucket,
        }
    return report


__all__ = ["COMM_USER_ROLE_MIX", "write_empty_comms", "generate_comms"]
