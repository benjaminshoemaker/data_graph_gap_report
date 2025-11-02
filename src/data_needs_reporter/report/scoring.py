from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass
class ScoringWeights:
    revenue: float = 0.60
    demand: float = 0.25
    severity: float = 0.15


def trailing_median(values: Sequence[float], window: int = 3) -> float:
    trailing = list(values[-window:])
    if not trailing:
        return 1.0
    trailing.sort()
    mid = len(trailing) // 2
    if len(trailing) % 2 == 1:
        return trailing[mid] or 1.0
    return (trailing[mid - 1] + trailing[mid]) / 2 or 1.0


def normalize_revenue(revenue_series: Sequence[float], current_revenue: float) -> float:
    median = trailing_median(revenue_series)
    if median == 0:
        return 0.0
    return min(max(current_revenue / median, 0.0), 2.0)


def reweight_source_weights(
    base_weights: Mapping[str, float],
    observed_volumes: Mapping[str, int],
    caps: Mapping[str, float],
    min_weight: float = 0.15,
    max_weight: float = 0.60,
) -> Dict[str, float]:
    total_volume = sum(observed_volumes.values()) or 1
    scaled: Dict[str, float] = {}
    for source, base in base_weights.items():
        fraction = observed_volumes.get(source, 0) / total_volume
        scaled[source] = base * (0.5 + fraction)
    for source, weight in list(scaled.items()):
        scaled[source] = min(max(weight, min_weight), max_weight)
    total = sum(scaled.values()) or 1
    return {source: weight / total for source, weight in scaled.items()}


def compute_severity(overages: Mapping[str, float], slos: Mapping[str, float]) -> float:
    severity = 0.0
    for metric, value in overages.items():
        target = slos.get(metric)
        if target is None or target <= 0:
            continue
        ratio = value / target
        if ratio > 1:
            severity += min(ratio - 1, 1.0)
    return min(severity, 1.0)


def recency_decay(days_since_peak: float, half_life: float = 60.0) -> float:
    return math.exp(-max(days_since_peak, 0.0) / half_life)


def compute_score(
    revenue_score: float,
    demand_score: float,
    severity_score: float,
    weights: Optional[ScoringWeights] = None,
) -> float:
    w = weights or ScoringWeights()
    return (
        w.revenue * revenue_score
        + w.demand * demand_score
        + w.severity * severity_score
    )


def select_top_actions(
    items: Sequence[Mapping[str, float]],
    top_k: int = 3,
    min_confidence: float = 0.55,
) -> List[Mapping[str, float]]:
    sorted_items = sorted(items, key=lambda item: item.get("score", 0.0), reverse=True)
    selected: List[Mapping[str, float]] = []
    themes_included: set[str] = set()
    for item in sorted_items:
        if item.get("confidence", 0.0) < min_confidence:
            continue
        theme = str(item.get("theme") or "")
        if theme in themes_included:
            continue
        selected.append(item)
        themes_included.add(theme)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for item in sorted_items:
            if item in selected:
                continue
            if item.get("confidence", 0.0) < min_confidence:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break
    return selected


def compute_confidence(
    class_conf: float,
    entity_conf: float,
    coverage: float,
    agreement: float,
) -> float:
    return (
        0.45 * max(min(class_conf, 1.0), 0.0)
        + 0.25 * max(min(entity_conf, 1.0), 0.0)
        + 0.20 * max(min(coverage, 1.0), 0.0)
        + 0.10 * max(min(agreement, 1.0), 0.0)
    )


__all__ = [
    "trailing_median",
    "normalize_revenue",
    "reweight_source_weights",
    "compute_severity",
    "recency_decay",
    "compute_score",
    "select_top_actions",
    "compute_confidence",
    "ScoringWeights",
]
