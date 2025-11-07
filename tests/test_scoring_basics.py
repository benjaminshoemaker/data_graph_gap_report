from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from data_needs_reporter.report.scoring import (
    compute_source_demand_weights,
    compute_weighted_theme_shares,
    trailing_monthly_revenue_median,
)


def test_normalize_source_weights_enforces_bounds_and_sum():
    base = {"nlq": 0.5, "slack": 0.3, "email": 0.2}
    volumes = {"nlq": 10, "slack": 5, "email": 5}
    bounds = {"min": 0.15, "max": 0.60}

    weights = compute_source_demand_weights(
        base,
        volumes,
        min_weight=bounds["min"],
        max_weight=bounds["max"],
    )

    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-6)
    for value in weights.values():
        assert bounds["min"] - 1e-6 <= value <= bounds["max"] + 1e-6

    # Higher observed volume should keep nlq weight >= slack >= email.
    assert weights["nlq"] >= weights["slack"] >= weights["email"]


def test_trailing_monthly_revenue_median_uses_last_three_months():
    tz = timezone.utc
    start = datetime(2024, 1, 1, tzinfo=tz)
    monthly_totals = {
        start + timedelta(days=30 * idx): value
        for idx, value in enumerate([100.0, 110.0, 90.0, 120.0])
    }
    expected_median = sorted([110.0, 90.0, 120.0])[1]
    assert trailing_monthly_revenue_median(monthly_totals) == expected_median


def test_theme_shares_preserve_relative_ordering_with_weights():
    tz = timezone.utc
    months = {
        datetime(2024, 1, 1, tzinfo=tz): 80.0,
        datetime(2024, 2, 1, tzinfo=tz): 120.0,
        datetime(2024, 3, 1, tzinfo=tz): 100.0,
    }
    median = trailing_monthly_revenue_median(months)
    assert median == sorted([80.0, 120.0, 100.0])[1]

    source_weights = compute_source_demand_weights(
        {"nlq": 0.5, "slack": 0.3, "email": 0.2},
        {"nlq": 40, "slack": 20, "email": 10},
        min_weight=0.15,
        max_weight=0.60,
    )

    items = [
        {
            "source": "slack",
            "day": "2024-03-01",
            "bucket": "am",
            "theme": "quality",
            "weight": 1.5,
        },
        {
            "source": "slack",
            "day": "2024-03-01",
            "bucket": "pm",
            "theme": "latency",
            "weight": 0.5,
        },
        {
            "source": "nlq",
            "day": "2024-03-02",
            "bucket": "am",
            "theme": "quality",
            "weight": 2.0,
        },
        {
            "source": "email",
            "day": "2024-03-02",
            "bucket": "pm",
            "theme": "tracking",
            "weight": 0.4,
        },
    ]

    strata_totals = {
        ("slack", "2024-03-01", "am"): 3.0,
        ("slack", "2024-03-01", "pm"): 1.0,
        ("nlq", "2024-03-02", "am"): 3.5,
        ("email", "2024-03-02", "pm"): 0.5,
    }

    shares = compute_weighted_theme_shares(
        items,
        strata_totals,
        source_weights=source_weights,
        stratum_fields=("source", "day", "bucket"),
        base_weight_field="weight",
    )

    assert pytest.approx(sum(shares.values()), rel=1e-6) == 1.0
    assert shares["quality"] > shares["tracking"] > shares["latency"]
