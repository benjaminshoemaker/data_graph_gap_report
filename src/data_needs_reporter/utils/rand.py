from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence

DEFAULT_HOURLY_PROFILE: Sequence[float] = (
    0.05,
    0.03,
    0.02,
    0.02,
    0.02,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    0.85,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.07,
)

DEFAULT_DAILY_PROFILE: Sequence[float] = (
    1.0,  # Monday
    1.0,  # Tuesday
    1.0,  # Wednesday
    1.05,  # Thursday
    1.1,  # Friday
    0.85,  # Saturday
    0.75,  # Sunday
)


def make_rng(seed: Optional[int]) -> random.Random:
    """Return a random number generator seeded for deterministic output."""
    return random.Random(seed)


def poisson_sample(rng: random.Random, lam: float) -> int:
    """Sample from a Poisson distribution with parameter ``lam``."""
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if lam == 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def negative_binomial_sample(rng: random.Random, r: float, p: float) -> int:
    """Sample from a negative binomial distribution (number of failures)."""
    if r <= 0:
        raise ValueError("r must be > 0")
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    # Sample lambda from the Gamma mixture then draw a Poisson variate.
    gamma_lambda = rng.gammavariate(r, (1 - p) / p)
    return poisson_sample(rng, gamma_lambda)


def lognormal_sample(rng: random.Random, mu: float, sigma: float) -> float:
    """Sample from a log-normal distribution."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return rng.lognormvariate(mu, sigma)


def hourly_weights(
    rng: random.Random,
    base_profile: Sequence[float] = DEFAULT_HOURLY_PROFILE,
    jitter: float = 0.05,
) -> List[float]:
    """Return normalized hourly weights with optional jitter."""
    if len(base_profile) != 24:
        raise ValueError("base_profile must contain 24 values")
    weights = _apply_jitter(rng, base_profile, jitter)
    return _normalize(weights)


def daily_weights(
    rng: random.Random,
    base_profile: Sequence[float] = DEFAULT_DAILY_PROFILE,
    jitter: float = 0.05,
) -> List[float]:
    """Return normalized daily weights (Mon-Sun) with optional jitter."""
    if len(base_profile) != 7:
        raise ValueError("base_profile must contain 7 values")
    weights = _apply_jitter(rng, base_profile, jitter)
    return _normalize(weights)


def zipf_weights(n: int, alpha: float) -> List[float]:
    """Return Zipf-distributed weights for ``n`` ranks."""
    if n <= 0:
        return []
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    base = [1 / ((idx + 1) ** alpha) for idx in range(n)]
    return _normalize(base)


def churn_schedule(
    rng: random.Random,
    population: int,
    monthly_churn_rate: float,
    max_months: int = 24,
) -> List[Optional[int]]:
    """Return the churn month (0-indexed) or None if retained."""
    if population < 0:
        raise ValueError("population must be non-negative")
    if not 0 <= monthly_churn_rate <= 1:
        raise ValueError("monthly_churn_rate must be between 0 and 1")
    if max_months <= 0:
        raise ValueError("max_months must be positive")

    results: List[Optional[int]] = []
    for _ in range(population):
        churn_month: Optional[int] = None
        for month in range(max_months):
            if rng.random() < monthly_churn_rate:
                churn_month = month
                break
        results.append(churn_month)
    return results


def _apply_jitter(
    rng: random.Random, profile: Sequence[float], jitter: float
) -> List[float]:
    if jitter <= 0:
        return list(profile)
    jitter = min(jitter, 1.0)
    return [
        max(value * rng.uniform(1 - jitter, 1 + jitter), 0.0) for value in profile
    ]


def _normalize(values: Iterable[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("values must sum to a positive number")
    return [value / total for value in values]


__all__ = [
    "DEFAULT_DAILY_PROFILE",
    "DEFAULT_HOURLY_PROFILE",
    "churn_schedule",
    "daily_weights",
    "hourly_weights",
    "lognormal_sample",
    "make_rng",
    "negative_binomial_sample",
    "poisson_sample",
    "zipf_weights",
]
