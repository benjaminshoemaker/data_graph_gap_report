from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

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
    "You extract tables and columns referenced in enterprise questions. "
    "Return JSON with keys tables (list of strings), columns (list of strings), confidence (0-1 float). "
    "Conversation or query:\n{body}"
)


def extract_entities(
    records: Sequence[Mapping[str, object]],
    llm_client: LLMClient,
    source: str,
    config: Optional[EntityExtractionConfig] = None,
    out_path: Optional[Path] = None,
) -> List[Mapping[str, object]]:
    cfg = config or EntityExtractionConfig()
    polars = _ensure_polars()
    total_tokens = 0
    max_tokens = int(cfg.cap_usd / (cfg.token_price_per_1k / 1000.0))
    results: List[Mapping[str, object]] = []

    for record in records:
        if total_tokens >= max_tokens:
            break
        body = str(record.get("body") or record.get("text") or "")
        if not body:
            continue
        prompt = PROMPT_TEMPLATE.format(body=body)
        try:
            response = llm_client.json_complete(prompt, temperature=cfg.temperature)
        except LLMError:
            continue
        tables = response.get("tables", [])
        columns = response.get("columns", [])
        confidence = response.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        tables = [str(t).lower() for t in tables if isinstance(t, str)]
        columns = [str(c).lower() for c in columns if isinstance(c, str)]
        tokens = response.get("tokens")
        tokens_used = int(tokens) if isinstance(tokens, (int, float)) else 50
        total_tokens += tokens_used
        result = {
            "source": source,
            "message_id": record.get("message_id") or record.get("query_id"),
            "tables": tables,
            "columns": columns,
            "confidence": min(max(confidence, 0.0), 1.0),
        }
        results.append(result)

    if out_path is not None:
        df = polars.DataFrame(results) if results else polars.DataFrame([])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)

    return results


__all__ = ["extract_entities", "EntityExtractionConfig", "PROMPT_TEMPLATE"]
