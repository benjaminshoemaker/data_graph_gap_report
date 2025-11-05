from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from data_needs_reporter.report.llm import LLMClient, LLMError

try:  # pragma: no cover - optional dependency
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


def _ensure_polars():
    if pl is None:  # pragma: no cover
        raise RuntimeError("polars is required for classification output.")
    return pl


def _estimate_tokens_from_body(body: str) -> int:
    return max(1, len(body) // 4)


def pack_thread(
    messages: Sequence[Mapping[str, Any]],
    exec_user_ids: Optional[Iterable[int]] = None,
    score_field: str = "prefilter_score",
    max_messages: int = 20,
    max_tokens: int = 900,
) -> Dict[str, Any]:
    exec_ids = set(exec_user_ids or [])

    def time_value(msg: Mapping[str, Any]) -> float:
        sent_at = msg.get("sent_at")
        if isinstance(sent_at, datetime):
            dt = sent_at.astimezone(timezone.utc) if sent_at.tzinfo else sent_at
            return dt.timestamp()
        return float(msg.get("message_id") or 0)

    def sort_key(msg: Mapping[str, Any]) -> Tuple[float, Any]:
        return (time_value(msg), msg.get("message_id"))

    ordered = sorted(messages, key=sort_key)
    if not ordered:
        return {"messages": [], "token_total": 0}

    selected: List[Mapping[str, Any]] = []
    added_ids = set()
    token_total = 0

    def token_value(msg: Mapping[str, Any]) -> int:
        tokens = msg.get("tokens")
        if isinstance(tokens, (int, float)):
            return int(tokens)
        body = str(msg.get("body") or msg.get("text") or "")
        return _estimate_tokens_from_body(body)

    def add_message(msg: Mapping[str, Any], force: bool = False) -> None:
        nonlocal token_total
        mid = msg.get("message_id")
        if mid in added_ids:
            return
        tokens = token_value(msg)
        if not force:
            if len(selected) >= max_messages:
                return
            if token_total + tokens > max_tokens:
                return
        selected.append(msg)
        added_ids.add(mid)
        token_total += tokens

    root = ordered[0]
    add_message(root, force=True)

    exec_messages = [
        msg for msg in ordered if msg.get("user_id") in exec_ids and msg is not root
    ]
    exec_messages.sort(key=sort_key)
    for msg in exec_messages:
        add_message(msg, force=True)

    remaining = [msg for msg in ordered if msg.get("message_id") not in added_ids]

    def score_value(msg: Mapping[str, Any]) -> float:
        score = msg.get(score_field)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    remaining.sort(
        key=lambda msg: (
            score_value(msg),
            sort_key(msg)[0],
        ),
        reverse=True,
    )

    for msg in remaining:
        add_message(msg)

    selected.sort(key=sort_key)
    return {"messages": selected, "token_total": token_total}


def _build_prompt(packed_messages: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    for msg in packed_messages:
        sent_at = msg.get("sent_at")
        if isinstance(sent_at, datetime):
            stamp = sent_at.isoformat()
        else:
            stamp = str(sent_at)
        user = msg.get("user_id", "unknown")
        body = str(msg.get("body") or msg.get("text") or "")
        lines.append(f"[{stamp}] user {user}: {body}")
    conversation = "\n".join(lines)
    prompt = (
        "You are classifying data demand messages. "
        "Return JSON with keys theme (string) and relevance (float 0-1)."
        "\nConversation:\n" + conversation
    )
    return prompt


def classify_threads(
    threads: Sequence[Mapping[str, Any]],
    llm_client: LLMClient,
    source: str,
    exec_user_ids: Optional[Iterable[int]] = None,
    parse_error_limit: int = 50,
    temperature: float = 0.0,
    max_messages: int = 20,
    max_tokens: int = 900,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    consecutive_errors = 0
    for thread in threads:
        packed = pack_thread(
            thread.get("messages", []),
            exec_user_ids=exec_user_ids,
            max_messages=max_messages,
            max_tokens=max_tokens,
        )
        packed_messages = packed["messages"]
        if not packed_messages:
            continue
        prompt = _build_prompt(packed_messages)
        try:
            response = llm_client.json_complete(prompt, temperature=temperature)
            consecutive_errors = 0
        except LLMError as exc:
            consecutive_errors += 1
            if consecutive_errors >= parse_error_limit:
                raise LLMError(
                    f"Parsing failed {consecutive_errors} times for source {source}"
                ) from exc
            continue

        theme = response.get("theme")
        relevance = response.get("relevance", response.get("score", 0.0))
        try:
            relevance = float(relevance)
        except (TypeError, ValueError):
            relevance = 0.0

        predictions.append(
            {
                "thread_id": thread.get("thread_id"),
                "source": source,
                "theme": theme,
                "relevance": relevance,
                "confidence": response.get("confidence", relevance),
                "message_ids": [msg.get("message_id") for msg in packed_messages],
                "message_count": len(packed_messages),
                "token_count": packed.get("token_total", 0),
                "raw_response": response,
            }
        )

    return predictions


def save_predictions(predictions: Sequence[Mapping[str, Any]], path: Path) -> None:
    polars = _ensure_polars()
    df = polars.DataFrame(predictions) if predictions else polars.DataFrame([])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


__all__ = ["pack_thread", "classify_threads", "save_predictions"]
