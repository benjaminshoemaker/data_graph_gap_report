import math

import pytest

from data_needs_reporter.report.scoring import (
    compute_confidence,
    compute_score,
    compute_severity,
    normalize_revenue,
    recency_decay,
    reweight_source_weights,
    select_top_actions,
)


def test_normalize_revenue():
    values = [1000, 1100, 900, 950]
    current = 1200
    score = normalize_revenue(values, current)
    assert 1.0 <= score <= 1.5


def test_reweight_source_weights():
    base = {"nlq": 0.5, "slack": 0.3, "email": 0.2}
    volumes = {"nlq": 50, "slack": 30, "email": 20}
    weights = reweight_source_weights(base, volumes, base)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    for w in weights.values():
        assert 0.15 <= w <= 0.60


def test_compute_severity():
    over = {"key_null_pct": 0.03, "fk_orphan_pct": 0.07}
    slos = {"key_null_pct": 0.02, "fk_orphan_pct": 0.05}
    severity = compute_severity(over, slos)
    assert severity > 0.0


def test_recency_decay():
    assert recency_decay(0) == pytest.approx(1.0)
    assert recency_decay(60) == pytest.approx(math.exp(-1))


def test_compute_score():
    score = compute_score(0.8, 0.6, 0.4)
    assert score == pytest.approx(0.60 * 0.8 + 0.25 * 0.6 + 0.15 * 0.4)


def test_select_top_actions_diversity():
    items = [
        {"theme": "retention", "score": 0.9, "confidence": 0.8},
        {"theme": "retention", "score": 0.85, "confidence": 0.9},
        {"theme": "growth", "score": 0.8, "confidence": 0.8},
        {"theme": "quality", "score": 0.75, "confidence": 0.7},
    ]
    selected = select_top_actions(items)
    themes = {item["theme"] for item in selected}
    assert len(selected) == 3
    assert len(themes) >= 2


def test_compute_confidence_bounds():
    conf = compute_confidence(1.2, -0.1, 0.5, 0.5)
    assert 0.0 <= conf <= 1.0
