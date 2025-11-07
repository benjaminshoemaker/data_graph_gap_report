from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from data_needs_reporter.report.llm import LLMClient, LLMError
from data_needs_reporter.utils.cost_guard import CostGuard

try:  # pragma: no cover - optional dependency
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

try:  # pragma: no cover - generated warehouse schemas
    from data_needs_reporter.generate.warehouse import (
        MARKETPLACE_TABLE_SCHEMAS,
        NEOBANK_TABLE_SCHEMAS,
    )
except ImportError:  # pragma: no cover
    NEOBANK_TABLE_SCHEMAS = {}
    MARKETPLACE_TABLE_SCHEMAS = {}


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
    "Respond with JSON array `entities` (or be the array) of objects like "
    '{{"table": "table_name", "column": "column_name", "confidence": 0-1}}. '
    "List at most 5 tables and 8 columns, lower-case, and only use the allowed dictionary.\n"
    "Conversation or query:\n{body}"
)

TABLE_CAP = 5
COLUMN_CAP = 8


def _clamp_conf(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if v != v:  # NaN guard
        v = default
    return min(max(v, 0.0), 1.0)


def build_entity_dictionary(archetype: str) -> Dict[str, Sequence[str]]:
    archetype_key = (archetype or "").lower()
    if archetype_key == "marketplace":
        schemas = MARKETPLACE_TABLE_SCHEMAS
    else:
        schemas = NEOBANK_TABLE_SCHEMAS
    dictionary: Dict[str, Sequence[str]] = {}
    for table, columns in schemas.items():
        table_name = str(table).strip().lower()
        if not table_name:
            continue
        dictionary[table_name] = [
            str(column).strip().lower()
            for column, _dtype in columns
            if str(column).strip()
        ]
    return dictionary


def _canonical_dictionary(
    dictionary: Mapping[str, Sequence[str]]
) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for table, columns in dictionary.items():
        table_name = str(table).strip().lower()
        if not table_name:
            continue
        column_list = []
        for column in columns:
            column_name = str(column).strip().lower()
            if column_name:
                column_list.append(column_name)
        if column_list:
            normalized[table_name] = sorted(set(column_list))
    return normalized


def _normalize_entities(
    payload: object, allowed_tables: Mapping[str, Sequence[str]]
) -> List[Dict[str, object]]:
    if isinstance(payload, Mapping):
        payload = (
            payload.get("entities") or payload.get("columns") or payload.get("items")
        )
    if not isinstance(payload, list):
        return []
    allowed = {table: set(columns) for table, columns in allowed_tables.items()}
    normalized: List[Dict[str, object]] = []
    seen: set[Tuple[str, str]] = set()
    for entry in payload:
        table_name: Optional[str] = None
        column_name: Optional[str] = None
        confidence_value: object = 0.0

        if isinstance(entry, Mapping):
            table_name = entry.get("table") or entry.get("name")
            column_name = entry.get("column") or entry.get("field")
            confidence_value = entry.get("confidence", entry.get("score", 0.0))
        elif isinstance(entry, str):
            value = entry.strip()
            if "." in value:
                table_name, column_name = value.split(".", 1)
            else:
                column_name = value

        table_key = (table_name or "").strip().lower()
        column_key = (column_name or "").strip().lower()
        if "." in column_key and not table_key:
            table_part, column_part = column_key.split(".", 1)
            table_key = table_part
            column_key = column_part
        if not column_key:
            continue

        allowed_columns = allowed.get(table_key)
        if not allowed_columns:
            matches = [tbl for tbl, cols in allowed.items() if column_key in cols]
            if len(matches) == 1:
                table_key = matches[0]
                allowed_columns = allowed.get(table_key)
        if not allowed_columns or column_key not in allowed_columns:
            continue

        key = (table_key, column_key)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "table": table_key,
                "column": column_key,
                "confidence": _clamp_conf(confidence_value, 0.0),
            }
        )
    return normalized


def _summarize_entities(
    items: List[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not items:
        return [], []
    table_scores: Dict[str, float] = {}
    for item in items:
        table = str(item["table"])
        confidence = float(item["confidence"])
        table_scores[table] = max(table_scores.get(table, 0.0), confidence)
    tables_sorted = sorted(
        table_scores.items(), key=lambda entry: (-entry[1], entry[0])
    )[:TABLE_CAP]
    tables = [{"name": name, "confidence": conf} for name, conf in tables_sorted]

    columns_sorted = sorted(
        items,
        key=lambda entry: (
            -float(entry["confidence"]),
            str(entry["table"]),
            str(entry["column"]),
        ),
    )[:COLUMN_CAP]
    columns = [
        {
            "table": entry["table"],
            "column": entry["column"],
            "confidence": float(entry["confidence"]),
        }
        for entry in columns_sorted
    ]
    return tables, columns


def extract_entities(
    records: Sequence[Mapping[str, object]],
    llm_client: LLMClient,
    dictionary: Mapping[str, Sequence[str]],
    *,
    config: Optional[EntityExtractionConfig] = None,
    out_path: Optional[Path] = None,
    guards: Optional[Dict[str, CostGuard]] = None,
) -> Tuple[List[Mapping[str, object]], Dict[str, Dict[str, float]]]:
    cfg = config or EntityExtractionConfig()
    polars = _ensure_polars()
    price_per_token = cfg.token_price_per_1k / 1000.0 if cfg.token_price_per_1k else 0.0
    max_tokens = (
        int(cfg.cap_usd / price_per_token) if price_per_token > 0 else float("inf")
    )
    dictionary_lower = _canonical_dictionary(dictionary)
    if dictionary_lower:
        dictionary_hint = ", ".join(
            f"{table} ({', '.join(sorted(columns))})"
            for table, columns in dictionary_lower.items()
        )
    else:
        dictionary_hint = "none"
    results: List[Mapping[str, object]] = []
    guard_pool: Dict[str, CostGuard] = dict(guards or {})
    coverage_counts: Dict[str, Dict[str, int]] = {}

    for record in records:
        source = str(record.get("source") or "").lower()
        entry = coverage_counts.setdefault(source, {"total": 0, "with_entities": 0})
        entry["total"] += 1
        if max_tokens == 0:
            continue
        body = str(record.get("body") or record.get("text") or "").strip()
        if not body:
            continue
        estimate_tokens = max(len(body), 50)
        if max_tokens != float("inf") and estimate_tokens > max_tokens:
            break
        prompt = PROMPT_TEMPLATE.format(dictionary=dictionary_hint, body=body)
        guard_key = source or "entities"
        guard_for_source = guard_pool.get(guard_key)
        if guard_for_source is None:
            guard_for_source = CostGuard(
                cap_usd=0.10,
                price_per_1k_tokens=cfg.token_price_per_1k or 0.002,
            )
            guard_pool[guard_key] = guard_for_source
        if guard_for_source.stopped_due_to_cap:
            continue
        try:
            response = llm_client.json_complete(
                prompt,
                temperature=cfg.temperature,
                guard=guard_for_source,
                channel=guard_key,
            )
        except LLMError:
            if guard_for_source.stopped_due_to_cap:
                continue
            continue
        normalized = _normalize_entities(response, dictionary_lower)
        if not normalized:
            continue
        tables, columns = _summarize_entities(normalized)
        if not tables and not columns:
            continue
        entry["with_entities"] += 1
        result = {
            "source": source,
            "record_id": record.get("record_id")
            or record.get("message_id")
            or record.get("query_id"),
            "message_id": record.get("message_id"),
            "query_id": record.get("query_id"),
            "tables": tables,
            "columns": columns,
        }
        results.append(result)

    if out_path is not None:
        df = polars.DataFrame(results) if results else polars.DataFrame([])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)

    coverage: Dict[str, Dict[str, float]] = {}
    total_messages = 0
    total_with_entities = 0
    for source, counts in coverage_counts.items():
        total = counts.get("total", 0)
        covered = counts.get("with_entities", 0)
        total_messages += total
        total_with_entities += covered
        coverage[source] = {
            "messages": float(total),
            "with_entities": float(covered),
            "coverage_pct": float(covered / total) if total else 0.0,
        }
    coverage["overall"] = {
        "messages": float(total_messages),
        "with_entities": float(total_with_entities),
        "coverage_pct": (
            float(total_with_entities / total_messages) if total_messages else 0.0
        ),
    }

    return results, coverage


__all__ = [
    "extract_entities",
    "EntityExtractionConfig",
    "PROMPT_TEMPLATE",
    "build_entity_dictionary",
]
