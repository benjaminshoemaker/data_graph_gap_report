from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from data_needs_reporter.report.llm import LLMClient, LLMError

try:  # pragma: no cover - optional dependency
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


def _ensure_polars():
    if pl is None:  # pragma: no cover
        raise RuntimeError("polars is required for entity extraction.")
    return pl


@dataclass
class EntityExtractionConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    cap_usd: float = 0.10
    token_price_per_1k: float = 0.002


PROMPT_TEMPLATE = (
    "You extract tables and columns referenced in enterprise questions.\n"
    "Allowed tables (with columns): {dictionary}\n"
    "Return JSON with keys tables (list of objects with name and confidence 0-1), "
    "columns (list of objects with table, column, confidence 0-1), and overall confidence 0-1.\n"
    "Conversation or query:\n{body}"
)


def _clamp_conf(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if v != v:  # NaN guard
        v = default
    return min(max(v, 0.0), 1.0)


def _normalize_table_items(
    raw_tables: Iterable[object],
    allowed_tables: Mapping[str, Iterable[str]],
    default_conf: float,
) -> List[Dict[str, object]]:
    seen: set[str] = set()
    items: List[Dict[str, object]] = []
    allowed = set(allowed_tables.keys())
    for entry in raw_tables:
        name: Optional[str] = None
        confidence = default_conf
        if isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("table")
            confidence = _clamp_conf(entry.get("confidence"), default_conf)
        elif isinstance(entry, str):
            name = entry
            confidence = default_conf
        if not name:
            continue
        canonical = str(name).strip().lower()
        if not canonical or canonical not in allowed or canonical in seen:
            continue
        items.append({"name": canonical, "confidence": confidence})
        seen.add(canonical)
    return items


def _normalize_column_items(
    raw_columns: Iterable[object],
    allowed_tables: Mapping[str, Iterable[str]],
    default_conf: float,
) -> List[Dict[str, object]]:
    allowed: Dict[str, set[str]] = {
        table: {col.lower() for col in columns}
        for table, columns in allowed_tables.items()
    }
    items: List[Dict[str, object]] = []
    seen: set[Tuple[str, str]] = set()
    for entry in raw_columns:
        table_name: Optional[str] = None
        column_name: Optional[str] = None
        confidence = default_conf

        if isinstance(entry, Mapping):
            table_name = entry.get("table") or entry.get("name")
            column_name = entry.get("column") or entry.get("field")
            confidence = _clamp_conf(entry.get("confidence"), default_conf)
        elif isinstance(entry, str):
            value = entry.strip()
            if "." in value:
                table_name, column_name = value.split(".", 1)
            else:
                column_name = value

        if not column_name:
            continue

        canonical_column = column_name.strip().lower()
        canonical_table = table_name.strip().lower() if table_name else None

        if canonical_table and canonical_table in allowed:
            if canonical_column not in allowed[canonical_table]:
                continue
            key = (canonical_table, canonical_column)
        else:
            matches = [tbl for tbl, cols in allowed.items() if canonical_column in cols]
            if len(matches) != 1:
                continue
            canonical_table = matches[0]
            key = (canonical_table, canonical_column)

        if key in seen:
            continue
        items.append(
            {
                "table": canonical_table,
                "column": canonical_column,
                "confidence": _clamp_conf(confidence, default_conf),
            }
        )
        seen.add(key)
    return items


def extract_entities(
    records: Sequence[Mapping[str, object]],
    llm_client: LLMClient,
    dictionary: Mapping[str, Sequence[str]],
    *,
    config: Optional[EntityExtractionConfig] = None,
    out_path: Optional[Path] = None,
    stats: Optional[Dict[str, float]] = None,
) -> List[Mapping[str, object]]:
    cfg = config or EntityExtractionConfig()
    polars = _ensure_polars()
    total_tokens = 0
    price_per_token = cfg.token_price_per_1k / 1000.0 if cfg.token_price_per_1k else 0.0
    max_tokens = (
        int(cfg.cap_usd / price_per_token) if price_per_token > 0 else float("inf")
    )
    dictionary_lower = {
        table.lower(): [col.lower() for col in columns]
        for table, columns in dictionary.items()
    }
    if dictionary_lower:
        dictionary_hint = ", ".join(
            f"{table} ({', '.join(sorted(columns))})"
            for table, columns in dictionary_lower.items()
        )
    else:
        dictionary_hint = "none"
    results: List[Mapping[str, object]] = []

    for record in records:
        if max_tokens != float("inf") and total_tokens >= max_tokens:
            break
        source = str(record.get("source") or "")
        body = str(record.get("body") or record.get("text") or "").strip()
        if not body:
            continue
        estimate_tokens = max(len(body), 50)
        if (
            max_tokens != float("inf")
            and price_per_token > 0
            and total_tokens + estimate_tokens > max_tokens
        ):
            break
        if total_tokens >= max_tokens:
            break
        prompt = PROMPT_TEMPLATE.format(dictionary=dictionary_hint, body=body)
        try:
            response = llm_client.json_complete(prompt, temperature=cfg.temperature)
        except LLMError:
            continue
        tokens_field = response.get("tokens")
        if isinstance(tokens_field, Mapping):
            tokens_used = tokens_field.get("total_tokens") or tokens_field.get(
                "prompt_tokens"
            )
        else:
            tokens_used = tokens_field
        try:
            tokens_used = int(tokens_used)
        except (TypeError, ValueError):
            tokens_used = estimate_tokens
        if tokens_used <= 0:
            tokens_used = estimate_tokens
        if max_tokens != float("inf") and total_tokens + tokens_used > max_tokens:
            break
        total_tokens += tokens_used
        default_conf = _clamp_conf(response.get("confidence"), 0.0)
        table_items = _normalize_table_items(
            response.get("tables", []), dictionary_lower, default_conf
        )
        column_items = _normalize_column_items(
            response.get("columns", []), dictionary_lower, default_conf
        )
        result = {
            "source": source,
            "record_id": record.get("record_id")
            or record.get("message_id")
            or record.get("query_id"),
            "message_id": record.get("message_id"),
            "query_id": record.get("query_id"),
            "tables": table_items,
            "columns": column_items,
        }
        results.append(result)

    if out_path is not None:
        df = polars.DataFrame(results) if results else polars.DataFrame([])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)

    if stats is not None:
        stats["total_tokens"] = float(total_tokens)
        stats["cost_usd"] = round(total_tokens * price_per_token, 6)

    return results


__all__ = ["extract_entities", "EntityExtractionConfig", "PROMPT_TEMPLATE"]
