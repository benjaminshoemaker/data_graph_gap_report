import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data_needs_reporter.config import DEFAULT_CONFIG_PATH, load_config
from data_needs_reporter.generate.defects import apply_typical_neobank_defects
from data_needs_reporter.generate.warehouse import (
    generate_neobank_dims,
    generate_neobank_facts,
)
from data_needs_reporter.report.metrics import compute_data_health, detect_null_spikes
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


def test_detect_null_spikes_identifies_lift():
    polars = pytest.importorskip("polars")
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for day in range(10):
        day_time = base + timedelta(days=day)
        for idx in range(20):
            value = None if (day == 8 and idx < 15) else (idx + day)
            rows.append({"event_time": day_time + timedelta(minutes=idx), "value": value})
    df = polars.DataFrame(rows).select(
        [
            polars.col("event_time").cast(polars.Datetime(time_zone="UTC")),
            polars.col("value").cast(polars.Int64)
        ]
    )

    spikes = detect_null_spikes(df, "event_time", "value")
    assert len(spikes) == 1
    spike = spikes[0]
    assert spike["column"] == "value"
    assert spike["ds"] == "2024-01-09"
    assert spike["lift_pct"] >= 8.0


def test_data_health_metrics_neobank(tmp_path: Path) -> None:
    pytest.importorskip("polars")
    cfg = load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})
    generate_neobank_dims(cfg, tmp_path, seed=222)
    generate_neobank_facts(cfg, tmp_path, tmp_path, seed=333)
    apply_typical_neobank_defects(cfg, tmp_path, seed=444)

    metrics = compute_data_health("neobank", tmp_path)
    assert metrics

    tables = {row["table"]: row for row in metrics}
    assert "fact_card_transaction" in tables
    assert "fact_subscription_invoice" in tables

    txn = tables["fact_card_transaction"]
    assert 1.0 <= txn["key_null_pct"] <= 3.5
    assert 90.0 <= txn["fk_success_pct"] <= 99.5
    assert 1.0 <= txn["fk_orphan_pct"] <= 8.0
    assert 0.3 <= txn["dup_keys_pct"] <= 1.5
    assert 120 <= txn["p95_ingest_lag_min"] <= 240
    assert txn["null_spike_days"]
    spike_entry = txn["null_spike_days"][0]
    assert {"ds", "column", "lift_pct"} <= set(spike_entry.keys())

    invoices = tables["fact_subscription_invoice"]
    assert 1.0 <= invoices["key_null_pct"] <= 2.5
    assert 97.5 <= invoices["fk_success_pct"] <= 99.0
    assert 0.2 <= invoices["dup_keys_pct"] <= 0.8
    assert 120 <= invoices["p95_ingest_lag_min"] <= 240
