from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from data_needs_reporter.report.entities import EntityExtractionConfig, extract_entities
from data_needs_reporter.report.llm import LLMClient

try:  # pragma: no cover - optional dependency
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

try:
    from data_needs_reporter.generate.warehouse import (
        MARKETPLACE_TABLE_SCHEMAS,
        NEOBANK_TABLE_SCHEMAS,
    )
except ImportError:  # pragma: no cover - defensive fallback
    NEOBANK_TABLE_SCHEMAS = {}
    MARKETPLACE_TABLE_SCHEMAS = {}


def _require_polars() -> "pl":
    if pl is None:  # pragma: no cover
        raise RuntimeError("polars is required for entity extraction pipeline.")
    return pl


def build_entity_dictionary(archetype: str) -> Dict[str, Sequence[str]]:
    archetype_key = archetype.lower()
    if archetype_key == "marketplace":
        schemas = MARKETPLACE_TABLE_SCHEMAS
    else:
        schemas = NEOBANK_TABLE_SCHEMAS
    return {
        table.lower(): [column.lower() for column, _ in columns]
        for table, columns in schemas.items()
    }


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
) -> Tuple[Sequence[Mapping[str, object]], Dict[str, float]]:
    dictionary_map = {
        table.lower(): [col.lower() for col in columns]
        for table, columns in (dictionary or build_entity_dictionary(archetype)).items()
    }

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

    stats: Dict[str, float] = {}
    extraction_config = config or EntityExtractionConfig()
    results = extract_entities(
        records,
        llm_client,
        dictionary_map,
        config=extraction_config,
        out_path=out_path,
        stats=stats,
    )
    return results, stats


def select_top_actions(
    items: Sequence[Mapping[str, object]],
    top_k: int = 3,
    min_confidence: float = 0.55,
) -> List[Mapping[str, object]]:
    sorted_items = sorted(
        items,
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    selected: List[Mapping[str, object]] = []
    seen_themes: set[str] = set()
    overflow: List[Mapping[str, object]] = []

    for item in sorted_items:
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue
        theme = str(item.get("theme") or "")
        if theme not in seen_themes and len(selected) < top_k:
            selected.append(item)
            seen_themes.add(theme)
        else:
            overflow.append(item)

    if len(selected) < top_k:
        for item in overflow:
            selected.append(item)
            if len(selected) >= top_k:
                break
    return selected


__all__ = ["build_entity_dictionary", "run_entity_extraction_for_archetype", "select_top_actions"]
