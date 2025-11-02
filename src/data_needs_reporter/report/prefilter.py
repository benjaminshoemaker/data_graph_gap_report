from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

BUCKET_WEIGHTS: Mapping[str, float] = {
    "data_quality": 0.18,
    "pipeline_health": 0.16,
    "analytics": 0.12,
    "governance": 0.1,
}

KEYWORD_WEIGHTS: Mapping[str, float] = {
    "missing": 0.12,
    "null": 0.1,
    "downtime": 0.1,
    "blocked": 0.08,
    "backlog": 0.07,
    "alert": 0.06,
    "churn": 0.05,
    "corrupt": 0.1,
}

ROLE_WEIGHTS: Mapping[str, float] = {
    "analytics": 0.1,
    "executive": 0.08,
    "product": 0.06,
    "engineering": 0.05,
}

CHANNEL_WEIGHTS: Mapping[str, float] = {
    "#data-quality": 0.12,
    "#data-eng": 0.08,
    "#analytics": 0.06,
}

STRUCTURAL_MAX = 0.1


@dataclass
class PrefilterResult:
    record: Dict[str, object]
    score: float
    reasons: List[str]
    selected: bool


def prefilter_messages(
    records: Sequence[MutableMapping[str, object]],
    user_roles: Mapping[int, str],
    threshold: float = 0.35,
    fallback_per_day_channel: int = 30,
) -> List[PrefilterResult]:
    scored: List[PrefilterResult] = []
    for record in records:
        score, reasons = _score_record(record, user_roles)
        scored.append(
            PrefilterResult(
                record=dict(record),
                score=round(score, 4),
                reasons=reasons,
                selected=score >= threshold,
            )
        )

    if fallback_per_day_channel > 0:
        groups: Dict[Tuple[str, str], List[PrefilterResult]] = defaultdict(list)
        for result in scored:
            day = _extract_day(result.record)
            channel = str(result.record.get("channel", ""))
            groups[(day, channel)].append(result)

        for items in groups.values():
            items.sort(key=lambda r: r.score, reverse=True)
            for idx, result in enumerate(items):
                if not result.selected and idx < fallback_per_day_channel:
                    result.selected = True
                    result.reasons.append("fallback:top_channel_day")

    for result in scored:
        result.record["prefilter_score"] = result.score
        result.record["prefilter_reasons"] = result.reasons
        result.record["prefilter_selected"] = result.selected
    return scored


def _score_record(
    record: Mapping[str, object], user_roles: Mapping[int, str]
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    bucket = str(record.get("bucket", "")).lower()
    if bucket in BUCKET_WEIGHTS:
        score += BUCKET_WEIGHTS[bucket]
        reasons.append(f"bucket:{bucket}")

    body = str(record.get("body") or record.get("text") or "").lower()
    subject = str(record.get("subject", "")).lower()

    for keyword, weight in KEYWORD_WEIGHTS.items():
        pattern = rf"\b{re.escape(keyword)}\b"
        if re.search(pattern, body) or (subject and re.search(pattern, subject)):
            score += weight
            reasons.append(f"keyword:{keyword}")

    channel = str(record.get("channel", "")).lower()
    if channel in CHANNEL_WEIGHTS:
        score += CHANNEL_WEIGHTS[channel]
        reasons.append(f"channel:{channel}")

    user_id = record.get("user_id") or record.get("sender_id")
    if user_id is not None:
        role = user_roles.get(int(user_id))
        if role and role in ROLE_WEIGHTS:
            score += ROLE_WEIGHTS[role]
            reasons.append(f"role:{role}")

    text_length = len(body)
    structural = 0.0
    if text_length > 180:
        structural += 0.06
    elif text_length > 90:
        structural += 0.04
    if "\n" in body or "-" in body or ":" in subject:
        structural += 0.03
    structural = min(structural, STRUCTURAL_MAX)
    if structural:
        score += structural
        reasons.append("structure")

    if re.search(r"\b(last night|today|yesterday)\b", body):
        score += 0.05
        reasons.append("event:recent")

    if record.get("thread_id") and record.get("thread_id") != record.get("message_id"):
        score += 0.03
        reasons.append("thread_reply")

    return min(score, 0.99), reasons


def _extract_day(record: Mapping[str, object]) -> str:
    timestamp = record.get("sent_at") or record.get("submitted_at")
    if isinstance(timestamp, datetime):
        return timestamp.date().isoformat()
    return "unknown"


__all__ = ["prefilter_messages", "PrefilterResult"]
