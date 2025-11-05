from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def _to_datetime(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        iso_value = value
        if iso_value.endswith("Z"):
            iso_value = iso_value[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(iso_value)
        except ValueError:
            return None
    return None


def _month_key(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return datetime(dt.year, dt.month, 1, tzinfo=dt.tzinfo)


def trailing_monthly_revenue_median(
    monthly_totals: Mapping[datetime, float],
    window_months: int = 3,
) -> float:
    if not monthly_totals:
        return 0.0
    valid_months = [
        month for month in monthly_totals.keys() if isinstance(month, datetime)
    ]
    if not valid_months:
        return 0.0
    valid_months.sort()
    window = max(1, min(window_months, len(valid_months)))
    selected_months = valid_months[-window:]
    values = [monthly_totals[month] for month in selected_months]
    return trailing_median(values, window=len(values))


def reweight_source_weights(
    base_weights: Mapping[str, float],
    observed_volumes: Mapping[str, int],
    caps: Optional[Mapping[str, float]] = None,
    min_weight: float = 0.15,
    max_weight: float = 0.60,
) -> Dict[str, float]:
    """Reweight source demand shares based on observed volume with clamp bounds."""
    if not base_weights:
        return {}

    if caps:
        min_weight = float(caps.get("min", min_weight))
        max_weight = float(caps.get("max", max_weight))
    min_weight = max(0.0, min_weight)
    max_weight = max(min_weight, max_weight)

    total_volume = sum(observed_volumes.get(source, 0) for source in base_weights)
    if total_volume <= 0:
        total_volume = 1

    raw_weights: Dict[str, float] = {}
    for source, base in base_weights.items():
        volume_fraction = observed_volumes.get(source, 0) / total_volume
        raw_weights[source] = max(base, 0.0) * (0.5 + volume_fraction)

    raw_sum = sum(raw_weights.values())
    if raw_sum <= 0.0:
        uniform = 1.0 / len(base_weights)
        return {source: uniform for source in base_weights}

    normalized = {source: weight / raw_sum for source, weight in raw_weights.items()}

    weights: Dict[str, float] = {
        source: min(max(weight, min_weight), max_weight)
        for source, weight in normalized.items()
    }

    total = sum(weights.values())
    if total <= 0.0:
        uniform = 1.0 / len(base_weights)
        return {source: uniform for source in base_weights}

    eps = 1e-9
    if total > 1.0 + eps:
        surplus = total - 1.0
        adjustable = {
            source: weights[source] - min_weight
            for source in weights
            if weights[source] > min_weight + eps
        }
        while surplus > eps and adjustable:
            total_room = sum(adjustable.values())
            if total_room <= eps:
                break
            for source in list(adjustable):
                weight_room = adjustable[source]
                if weight_room <= 0.0:
                    del adjustable[source]
                    continue
                share = surplus * (weight_room / total_room)
                reduction = min(weight_room, share)
                weights[source] -= reduction
                surplus -= reduction
                adjustable[source] = weights[source] - min_weight
                if adjustable[source] <= eps:
                    del adjustable[source]
                if surplus <= eps:
                    break
    elif total < 1.0 - eps:
        deficit = 1.0 - total
        adjustable = {
            source: max_weight - weights[source]
            for source in weights
            if max_weight - weights[source] > eps
        }
        while deficit > eps and adjustable:
            total_room = sum(adjustable.values())
            if total_room <= eps:
                break
            for source in list(adjustable):
                weight_room = adjustable[source]
                if weight_room <= 0.0:
                    del adjustable[source]
                    continue
                share = deficit * (weight_room / total_room)
                addition = min(weight_room, share)
                weights[source] += addition
                deficit -= addition
                adjustable[source] = max_weight - weights[source]
                if adjustable[source] <= eps:
                    del adjustable[source]
                if deficit <= eps:
                    break

    total = sum(weights.values())
    if total <= 0.0:
        uniform = 1.0 / len(base_weights)
        return {source: uniform for source in base_weights}
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        scale = 1.0 / total
        for source in weights:
            weights[source] = max(
                min(weights[source] * scale, max_weight),
                min_weight,
            )

    final_total = sum(weights.values()) or 1.0
    return {source: weights[source] / final_total for source in weights}


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


def post_stratified_item_weights(
    items: Sequence[Mapping[str, object]],
    strata_totals: Mapping[Tuple[str, ...], float],
    *,
    stratum_fields: Sequence[str] = ("source", "day", "bucket"),
    base_weight_field: Optional[str] = None,
) -> List[float]:
    """Compute post-stratified weights per item based on strata population totals."""
    if not items:
        return []

    def _as_float(value: object, default: float = 1.0) -> float:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        if math.isnan(numeric) or numeric < 0:
            return default
        return numeric

    def _stratum_key(item: Mapping[str, object]) -> Tuple[str, ...]:
        return tuple(str(item.get(field, "")) for field in stratum_fields)

    base_weights: List[float] = []
    strata_keys: List[Tuple[str, ...]] = []
    sample_totals: Dict[Tuple[str, ...], float] = {}
    for item in items:
        key = _stratum_key(item)
        base_weight = (
            _as_float(item.get(base_weight_field), 1.0) if base_weight_field else 1.0
        )
        base_weights.append(base_weight)
        strata_keys.append(key)
        sample_totals[key] = sample_totals.get(key, 0.0) + base_weight

    weights: List[float] = []
    for key, base_weight in zip(strata_keys, base_weights):
        sample_total = sample_totals.get(key, 0.0)
        if sample_total <= 0.0 or base_weight <= 0.0:
            weights.append(0.0)
            continue
        target_total = float(strata_totals.get(key, sample_total))
        if target_total <= 0.0:
            weights.append(0.0)
            continue
        weights.append(base_weight * (target_total / sample_total))
    return weights


def compute_weighted_theme_shares(
    items: Sequence[Mapping[str, object]],
    strata_totals: Mapping[Tuple[str, ...], float],
    *,
    source_weights: Optional[Mapping[str, float]] = None,
    stratum_fields: Sequence[str] = ("source", "day", "bucket"),
    theme_field: str = "theme",
    base_weight_field: Optional[str] = None,
) -> Dict[str, float]:
    """Aggregate theme shares using post-stratified item weights."""
    if not items:
        return {}

    weights = post_stratified_item_weights(
        items,
        strata_totals,
        stratum_fields=stratum_fields,
        base_weight_field=base_weight_field,
    )

    source_theme_totals: Dict[str, Dict[str, float]] = {}
    for item, weight in zip(items, weights):
        if weight <= 0.0:
            continue
        source = str(item.get("source", ""))
        theme = str(item.get(theme_field) or "unknown")
        per_source = source_theme_totals.setdefault(source, {})
        per_source[theme] = per_source.get(theme, 0.0) + weight

    if not source_theme_totals:
        return {}

    if source_weights:
        contributions: Dict[str, float] = {}
        active_sources = {
            source: sum(theme_totals.values())
            for source, theme_totals in source_theme_totals.items()
            if sum(theme_totals.values()) > 0.0
        }
        if not active_sources:
            return {}
        normalization = sum(
            source_weights.get(source, 0.0) for source in active_sources
        )
        if normalization <= 0.0:
            return {}
        for source, total_weight in active_sources.items():
            if total_weight <= 0.0:
                continue
            source_weight = source_weights.get(source, 0.0)
            if source_weight <= 0.0:
                continue
            scaled_source_weight = source_weight / normalization
            theme_totals = source_theme_totals[source]
            for theme, amount in theme_totals.items():
                share_within_source = amount / total_weight
                contributions[theme] = contributions.get(theme, 0.0) + (
                    scaled_source_weight * share_within_source
                )
        total = sum(contributions.values())
        if total <= 0.0:
            return {}
        return {theme: value / total for theme, value in contributions.items()}

    aggregated: Dict[str, float] = {}
    total_weight = 0.0
    for theme_totals in source_theme_totals.values():
        for theme, amount in theme_totals.items():
            aggregated[theme] = aggregated.get(theme, 0.0) + amount
            total_weight += amount
    if total_weight <= 0.0:
        return {}
    return {theme: amount / total_weight for theme, amount in aggregated.items()}


def compute_neobank_revenue_risk(
    transactions: Sequence[Mapping[str, object]],
    invoices: Sequence[Mapping[str, object]],
    affected_transaction_ids: Sequence[int],
    affected_invoice_ids: Sequence[int],
    *,
    window_months: int = 3,
) -> Dict[str, float]:
    """Estimate revenue at risk for neobank using interchange + subscriptions."""
    txn_records = list(transactions or [])
    invoice_records = list(invoices or [])
    affected_txn_set = {int(tid) for tid in affected_transaction_ids if tid is not None}
    affected_invoice_set = {int(iid) for iid in affected_invoice_ids if iid is not None}

    monthly_totals: Dict[datetime, float] = {}
    interchange_at_risk = 0.0
    subscription_at_risk = 0.0

    for txn in txn_records:
        txn_id = txn.get("txn_id") or txn.get("transaction_id")
        try:
            txn_id_int = int(txn_id) if txn_id is not None else None
        except (TypeError, ValueError):
            txn_id_int = None
        amount_cents = _safe_float(txn.get("amount_cents"), 0.0)
        interchange_bps = _safe_float(txn.get("interchange_bps"), 0.0)
        interchange_usd = (amount_cents / 100.0) * (interchange_bps / 10000.0)
        event_dt = _to_datetime(txn.get("event_time") or txn.get("loaded_at"))
        month = _month_key(event_dt)
        if month is not None:
            monthly_totals[month] = monthly_totals.get(month, 0.0) + interchange_usd
        if txn_id_int is not None and txn_id_int in affected_txn_set:
            interchange_at_risk += interchange_usd

    for invoice in invoice_records:
        invoice_id = invoice.get("invoice_id")
        try:
            invoice_id_int = int(invoice_id) if invoice_id is not None else None
        except (TypeError, ValueError):
            invoice_id_int = None
        amount_cents = _safe_float(invoice.get("amount_cents"), 0.0)
        subscription_usd = amount_cents / 100.0
        period_dt = _to_datetime(
            invoice.get("period_start")
            or invoice.get("paid_at")
            or invoice.get("period_end")
        )
        month = _month_key(period_dt)
        if month is not None:
            monthly_totals[month] = monthly_totals.get(month, 0.0) + subscription_usd
        if invoice_id_int is not None and invoice_id_int in affected_invoice_set:
            subscription_at_risk += subscription_usd

    revenue_at_risk = interchange_at_risk + subscription_at_risk

    median_revenue = trailing_monthly_revenue_median(
        monthly_totals, window_months=window_months
    )

    if median_revenue > 0.0:
        revenue_risk_ratio = min(max(revenue_at_risk / median_revenue, 0.0), 1.0)
    else:
        revenue_risk_ratio = 0.0

    return {
        "interchange_at_risk_usd": interchange_at_risk,
        "subscription_at_risk_usd": subscription_at_risk,
        "revenue_at_risk_usd": revenue_at_risk,
        "median_monthly_revenue_usd": median_revenue,
        "revenue_risk_ratio": revenue_risk_ratio,
    }


def compute_marketplace_revenue_risk(
    orders: Sequence[Mapping[str, object]],
    payments: Sequence[Mapping[str, object]],
    affected_order_ids: Sequence[int],
    *,
    take_rate: float = 0.12,
    window_months: int = 3,
) -> Dict[str, float]:
    """Estimate revenue at risk for marketplace using GMV and take rate."""

    order_time_lookup: Dict[int, Optional[datetime]] = {}
    for order in orders or []:
        order_id = order.get("order_id")
        try:
            order_id_int = int(order_id) if order_id is not None else None
        except (TypeError, ValueError):
            continue
        order_time_lookup[order_id_int] = _to_datetime(order.get("order_time"))

    affected_orders = {
        int(order_id) for order_id in affected_order_ids if order_id is not None
    }

    effective_take_rate = max(float(take_rate), 0.0)

    monthly_revenue: Dict[datetime, float] = {}
    gmv_at_risk = 0.0
    net_revenue_at_risk = 0.0
    total_gmv = 0.0
    total_platform_fee = 0.0

    for payment in payments or []:
        order_id = payment.get("order_id")
        try:
            order_id_int = int(order_id) if order_id is not None else None
        except (TypeError, ValueError):
            order_id_int = None

        gmv_usd = _safe_float(payment.get("buyer_paid_cents"), 0.0) / 100.0
        platform_fee_usd = _safe_float(payment.get("platform_fee_cents"), 0.0) / 100.0

        total_gmv += gmv_usd
        if platform_fee_usd > 0.0:
            total_platform_fee += platform_fee_usd

        order_dt = order_time_lookup.get(order_id_int)
        if order_dt is None:
            order_dt = _to_datetime(payment.get("captured_at"))

        month = _month_key(order_dt)
        if month is not None:
            monthly_revenue.setdefault(month, 0.0)

        if order_id_int is not None and order_id_int in affected_orders:
            gmv_at_risk += gmv_usd

        # Defer net revenue calculation until take rate finalized.

    if effective_take_rate <= 0.0 and total_gmv > 0.0:
        effective_take_rate = total_platform_fee / total_gmv if total_gmv else 0.0
        if effective_take_rate < 0.0:
            effective_take_rate = 0.0

    # Second pass to accumulate monthly revenue and net revenue at risk
    for payment in payments or []:
        order_id = payment.get("order_id")
        try:
            order_id_int = int(order_id) if order_id is not None else None
        except (TypeError, ValueError):
            order_id_int = None

        gmv_usd = _safe_float(payment.get("buyer_paid_cents"), 0.0) / 100.0
        platform_fee_usd = _safe_float(payment.get("platform_fee_cents"), 0.0) / 100.0
        net_revenue = (
            platform_fee_usd
            if platform_fee_usd > 0.0
            else gmv_usd * effective_take_rate
        )

        order_dt = order_time_lookup.get(order_id_int)
        if order_dt is None:
            order_dt = _to_datetime(payment.get("captured_at"))
        month = _month_key(order_dt)
        if month is not None:
            monthly_revenue[month] = monthly_revenue.get(month, 0.0) + net_revenue

        if order_id_int is not None and order_id_int in affected_orders:
            net_revenue_at_risk += gmv_usd * effective_take_rate

    median_revenue = trailing_monthly_revenue_median(
        monthly_revenue, window_months=window_months
    )

    if median_revenue > 0.0:
        revenue_risk_ratio = min(max(net_revenue_at_risk / median_revenue, 0.0), 1.0)
    else:
        revenue_risk_ratio = 0.0

    return {
        "gmv_at_risk_usd": gmv_at_risk,
        "net_revenue_at_risk_usd": net_revenue_at_risk,
        "median_monthly_revenue_usd": median_revenue,
        "revenue_risk_ratio": revenue_risk_ratio,
        "take_rate_used": effective_take_rate,
    }


__all__ = [
    "trailing_median",
    "normalize_revenue",
    "trailing_monthly_revenue_median",
    "reweight_source_weights",
    "compute_severity",
    "recency_decay",
    "compute_score",
    "compute_confidence",
    "post_stratified_item_weights",
    "compute_weighted_theme_shares",
    "compute_neobank_revenue_risk",
    "compute_marketplace_revenue_risk",
    "ScoringWeights",
]
