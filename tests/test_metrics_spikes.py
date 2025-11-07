from __future__ import annotations

import pytest

from data_needs_reporter.report.metrics import detect_null_spikes


def test_detect_null_spikes_flags_lift() -> None:
    daily = [
        {"date": "2024-01-01", "pct": 1.5},
        {"date": "2024-01-02", "pct": 1.8},
        {"date": "2024-01-03", "pct": 1.6},
        {"date": "2024-01-04", "pct": 12.2},
    ]

    spikes = detect_null_spikes(daily, window=3, threshold_pp=5.0)
    assert spikes
    spike = spikes[0]
    assert spike["date"] == "2024-01-04"
    assert spike["pct"] == pytest.approx(12.2)
    assert spike["baseline_pct"] == pytest.approx(1.6)
    assert spike["lift_pp"] == pytest.approx(10.6, abs=1e-3)


def test_detect_null_spikes_requires_history() -> None:
    daily = [
        {"date": "2024-01-01", "pct": 0.5},
        {"date": "2024-01-02", "pct": 9.0},
    ]
    spikes = detect_null_spikes(daily, window=3, threshold_pp=5.0)
    assert spikes == []
