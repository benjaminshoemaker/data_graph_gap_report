from __future__ import annotations

from data_needs_reporter.utils.rand import (
    churn_schedule,
    daily_weights,
    hourly_weights,
    lognormal_sample,
    make_rng,
    negative_binomial_sample,
    poisson_sample,
    zipf_weights,
)


def test_poisson_sample_deterministic() -> None:
    rng1 = make_rng(42)
    rng2 = make_rng(42)
    seq1 = [poisson_sample(rng1, 3.2) for _ in range(5)]
    seq2 = [poisson_sample(rng2, 3.2) for _ in range(5)]
    assert seq1 == seq2

    rng3 = make_rng(41)
    seq3 = [poisson_sample(rng3, 3.2) for _ in range(5)]
    assert seq1 != seq3


def test_negative_binomial_sample_deterministic() -> None:
    rng1 = make_rng(99)
    rng2 = make_rng(99)
    seq1 = [negative_binomial_sample(rng1, 5.0, 0.4) for _ in range(6)]
    seq2 = [negative_binomial_sample(rng2, 5.0, 0.4) for _ in range(6)]
    assert seq1 == seq2

    rng3 = make_rng(98)
    seq3 = [negative_binomial_sample(rng3, 5.0, 0.4) for _ in range(6)]
    assert seq1 != seq3


def test_lognormal_sample_deterministic() -> None:
    rng1 = make_rng(7)
    rng2 = make_rng(7)
    seq1 = [lognormal_sample(rng1, 0.0, 0.5) for _ in range(3)]
    seq2 = [lognormal_sample(rng2, 0.0, 0.5) for _ in range(3)]
    assert seq1 == seq2

    rng3 = make_rng(8)
    seq3 = [lognormal_sample(rng3, 0.0, 0.5) for _ in range(3)]
    assert seq1 != seq3


def test_hourly_weights_deterministic() -> None:
    rng1 = make_rng(123)
    rng2 = make_rng(123)
    weights1 = hourly_weights(rng1)
    weights2 = hourly_weights(rng2)
    assert weights1 == weights2
    assert len(weights1) == 24
    assert abs(sum(weights1) - 1.0) < 1e-9

    rng3 = make_rng(321)
    weights3 = hourly_weights(rng3)
    assert weights1 != weights3


def test_daily_weights_deterministic() -> None:
    rng1 = make_rng(55)
    rng2 = make_rng(55)
    weights1 = daily_weights(rng1)
    weights2 = daily_weights(rng2)
    assert weights1 == weights2
    assert len(weights1) == 7
    assert abs(sum(weights1) - 1.0) < 1e-9

    rng3 = make_rng(56)
    weights3 = daily_weights(rng3)
    assert weights1 != weights3


def test_zipf_weights_normalized() -> None:
    weights = zipf_weights(5, 1.2)
    assert len(weights) == 5
    assert abs(sum(weights) - 1.0) < 1e-9
    assert weights[0] > weights[-1]


def test_churn_schedule_deterministic() -> None:
    rng1 = make_rng(777)
    rng2 = make_rng(777)
    schedule1 = churn_schedule(
        rng1, population=10, monthly_churn_rate=0.2, max_months=6
    )
    schedule2 = churn_schedule(
        rng2, population=10, monthly_churn_rate=0.2, max_months=6
    )
    assert schedule1 == schedule2

    rng3 = make_rng(778)
    schedule3 = churn_schedule(
        rng3, population=10, monthly_churn_rate=0.2, max_months=6
    )
    assert schedule1 != schedule3
