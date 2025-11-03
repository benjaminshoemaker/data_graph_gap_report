from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, MutableMapping, Tuple

from data_needs_reporter.config import AppConfig
from data_needs_reporter.utils.io import write_parquet_atomic
from data_needs_reporter.utils.rand import poisson_sample, zipf_weights

try:  # pragma: no cover - optional dependency handling
    import polars as _pl
except ImportError:  # pragma: no cover - optional dependency handling
    _pl = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import polars as pl  # noqa: F401

pl = _pl  # type: ignore[assignment]

SchemaMapping = Dict[str, Tuple[Tuple[str, "pl.DataType"], ...]]

NEOBANK_MERCHANT_SECTORS: Dict[str, Tuple[int, ...]] = {
    "groceries": (5411, 5412, 5499, 5422, 5441),
    "restaurants": (5812, 5814, 5494, 5462, 5811),
    "travel": (3000, 3001, 3376, 4112, 4011),
    "transport": (4121, 4789, 4131, 7512, 7523),
    "digital_services": (5734, 5735, 5815, 7372, 4899),
    "health": (5912, 8062, 8011, 8099, 5914),
    "retail": (5651, 5942, 5944, 5948, 5999),
    "entertainment": (7832, 7922, 7929, 7996, 7999),
}

NEOBANK_MCC_TO_SECTOR: Dict[int, str] = {
    code: sector for sector, codes in NEOBANK_MERCHANT_SECTORS.items() for code in codes
}

if pl is not None:
    UTC_DATETIME = pl.Datetime(time_zone="UTC")

    NEOBANK_TABLE_SCHEMAS: SchemaMapping = {
        "dim_customer": (
            ("customer_id", pl.Int64),
            ("created_at", UTC_DATETIME),
            ("kyc_status", pl.Utf8),
        ),
        "dim_account": (
            ("account_id", pl.Int64),
            ("customer_id", pl.Int64),
            ("type", pl.Utf8),
            ("created_at", UTC_DATETIME),
            ("status", pl.Utf8),
        ),
        "dim_card": (
            ("card_id", pl.Int64),
            ("account_id", pl.Int64),
            ("status", pl.Utf8),
            ("activated_at", UTC_DATETIME),
        ),
        "dim_merchant": (
            ("merchant_id", pl.Int64),
            ("mcc", pl.Int32),
            ("name", pl.Utf8),
        ),
        "dim_plan": (
            ("plan_id", pl.Int64),
            ("name", pl.Utf8),
            ("price_cents", pl.Int64),
            ("cadence", pl.Utf8),
        ),
        "fact_card_transaction": (
            ("txn_id", pl.Int64),
            ("card_id", pl.Int64),
            ("merchant_id", pl.Int64),
            ("event_time", UTC_DATETIME),
            ("amount_cents", pl.Int64),
            ("interchange_bps", pl.Float64),
            ("channel", pl.Utf8),
            ("auth_result", pl.Utf8),
            ("loaded_at", UTC_DATETIME),
        ),
        "fact_subscription_invoice": (
            ("invoice_id", pl.Int64),
            ("customer_id", pl.Int64),
            ("plan_id", pl.Int64),
            ("period_start", UTC_DATETIME),
            ("period_end", UTC_DATETIME),
            ("paid_at", UTC_DATETIME),
            ("amount_cents", pl.Int64),
            ("loaded_at", UTC_DATETIME),
        ),
    }

    MARKETPLACE_TABLE_SCHEMAS: SchemaMapping = {
        "dim_buyer": (
            ("buyer_id", pl.Int64),
            ("created_at", UTC_DATETIME),
            ("country", pl.Utf8),
        ),
        "dim_seller": (
            ("seller_id", pl.Int64),
            ("created_at", UTC_DATETIME),
            ("country", pl.Utf8),
            ("shop_status", pl.Utf8),
        ),
        "dim_category": (
            ("category_id", pl.Int64),
            ("name", pl.Utf8),
            ("parent_id", pl.Int64),
        ),
        "dim_listing": (
            ("listing_id", pl.Int64),
            ("seller_id", pl.Int64),
            ("category_id", pl.Int64),
            ("created_at", UTC_DATETIME),
            ("status", pl.Utf8),
            ("price_cents", pl.Int64),
        ),
        "fact_order": (
            ("order_id", pl.Int64),
            ("buyer_id", pl.Int64),
            ("order_time", UTC_DATETIME),
            ("currency", pl.Utf8),
            ("subtotal_cents", pl.Int64),
            ("tax_cents", pl.Int64),
            ("shipping_cents", pl.Int64),
            ("discount_cents", pl.Int64),
            ("loaded_at", UTC_DATETIME),
        ),
        "fact_order_item": (
            ("order_id", pl.Int64),
            ("line_id", pl.Int32),
            ("listing_id", pl.Int64),
            ("seller_id", pl.Int64),
            ("qty", pl.Int32),
            ("item_price_cents", pl.Int64),
            ("loaded_at", UTC_DATETIME),
        ),
        "fact_payment": (
            ("order_id", pl.Int64),
            ("captured_at", UTC_DATETIME),
            ("buyer_paid_cents", pl.Int64),
            ("seller_earnings_cents", pl.Int64),
            ("platform_fee_cents", pl.Int64),
            ("loaded_at", UTC_DATETIME),
        ),
        "snapshot_listing_daily": (
            ("ds", pl.Date),
            ("listing_id", pl.Int64),
            ("status", pl.Utf8),
        ),
    }

    WAREHOUSE_SCHEMAS: Dict[str, SchemaMapping] = {
        "neobank": NEOBANK_TABLE_SCHEMAS,
        "marketplace": MARKETPLACE_TABLE_SCHEMAS,
    }
else:  # pragma: no cover - executed only when polars missing
    UTC_DATETIME = None  # type: ignore[assignment]
    NEOBANK_TABLE_SCHEMAS = {}
    MARKETPLACE_TABLE_SCHEMAS = {}
    WAREHOUSE_SCHEMAS = {}


def write_empty_warehouse(archetype: str, out_dir: Path) -> None:
    _ensure_polars()
    archetype_key = archetype.lower()
    if archetype_key not in WAREHOUSE_SCHEMAS:
        raise ValueError(f"Unsupported archetype: {archetype}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    schemas = WAREHOUSE_SCHEMAS[archetype_key]
    for table_name, schema in schemas.items():
        df = _empty_dataframe(schema)
        write_parquet_atomic(out_path / f"{table_name}.parquet", df)


def generate_neobank_dims(
    cfg: AppConfig, out_dir: Path, seed: int | None
) -> Dict[str, int]:
    polars = _ensure_polars()
    rng = random.Random(seed if seed is not None else cfg.warehouse.seed)

    scale = cfg.warehouse.scale.lower()
    if scale == "evaluation":
        customer_count = 3500
    else:
        customer_count = 1000

    base_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    months = max(cfg.warehouse.months, 1)
    total_days = months * 30

    def random_ts(start: datetime, max_days: int) -> datetime:
        delta_days = rng.randrange(max_days + 1)
        delta_minutes = rng.randrange(24 * 60)
        return start + timedelta(days=delta_days, minutes=delta_minutes)

    customers = []
    for cid in range(1, customer_count + 1):
        created_at = random_ts(base_start, total_days - 1)
        kyc_status = rng.choices(
            ["pending", "verified", "refused"],
            weights=[0.2, 0.7, 0.1],
        )[0]
        customers.append(
            {
                "customer_id": cid,
                "created_at": created_at,
                "kyc_status": kyc_status,
            }
        )

    accounts = []
    account_id = 1
    account_types = ["checking", "savings", "credit"]
    account_weights = [0.55, 0.2, 0.25]
    account_statuses = ["active", "dormant", "closed"]
    account_status_weights = [0.75, 0.18, 0.07]
    for customer in customers:
        num_accounts = 1
        if rng.random() < 0.35:
            num_accounts += 1
        if rng.random() < 0.12:
            num_accounts += 1

        for _ in range(num_accounts):
            account_type = rng.choices(account_types, weights=account_weights)[0]
            created_at = random_ts(customer["created_at"], 120)
            status = rng.choices(account_statuses, weights=account_status_weights)[0]
            accounts.append(
                {
                    "account_id": account_id,
                    "customer_id": customer["customer_id"],
                    "type": account_type,
                    "created_at": created_at,
                    "status": status,
                }
            )
            account_id += 1

    cards = []
    card_id = 1
    card_statuses = ["active", "inactive"]
    for account in accounts:
        base_probability = 0.85 if account["type"] == "credit" else 0.45
        if account["type"] == "savings":
            base_probability = 0.25
        num_cards = 1 if rng.random() < base_probability else 0
        if account["type"] == "credit" and rng.random() < 0.25:
            num_cards += 1
        if num_cards == 0:
            continue

        for _ in range(num_cards):
            activated_at = random_ts(account["created_at"], 60)
            status = rng.choices(card_statuses, weights=[0.85, 0.15])[0]
            cards.append(
                {
                    "card_id": card_id,
                    "account_id": account["account_id"],
                    "status": status,
                    "activated_at": activated_at,
                }
            )
            card_id += 1

    merchants = []
    sector_names = list(NEOBANK_MERCHANT_SECTORS.keys())
    sector_weights = [0.15, 0.15, 0.12, 0.12, 0.14, 0.1, 0.12, 0.1]
    merchant_count = 800 if scale == "evaluation" else 200
    merchant_id = 1
    for sector, weight in zip(sector_names, sector_weights):
        sector_target = max(int(merchant_count * weight), 1)
        codes = list(NEOBANK_MERCHANT_SECTORS[sector])
        for idx in range(sector_target):
            mcc = rng.choice(codes)
            merchants.append(
                {
                    "merchant_id": merchant_id,
                    "mcc": mcc,
                    "name": f"{sector.replace('_', ' ').title()} Merchant {merchant_id}",
                }
            )
            merchant_id += 1

    # ensure total merchant_count reached
    while len(merchants) < merchant_count:
        sector = rng.choices(sector_names, weights=sector_weights)[0]
        mcc = rng.choice(NEOBANK_MERCHANT_SECTORS[sector])
        merchants.append(
            {
                "merchant_id": merchant_id,
                "mcc": mcc,
                "name": f"{sector.replace('_', ' ').title()} Merchant {merchant_id}",
            }
        )
        merchant_id += 1

    plans = [
        {"plan_id": 1, "name": "Basic", "price_cents": 0, "cadence": "monthly"},
        {"plan_id": 2, "name": "Standard", "price_cents": 499, "cadence": "monthly"},
        {"plan_id": 3, "name": "Premium", "price_cents": 999, "cadence": "monthly"},
        {"plan_id": 4, "name": "Premium+", "price_cents": 1899, "cadence": "annual"},
    ]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    counts = {
        "dim_customer": _write_table(
            "neobank", "dim_customer", customers, out_path, polars
        ),
        "dim_account": _write_table(
            "neobank", "dim_account", accounts, out_path, polars
        ),
        "dim_card": _write_table("neobank", "dim_card", cards, out_path, polars),
        "dim_merchant": _write_table(
            "neobank", "dim_merchant", merchants, out_path, polars
        ),
        "dim_plan": _write_table("neobank", "dim_plan", plans, out_path, polars),
    }
    return counts


def generate_neobank_facts(
    cfg: AppConfig,
    dims_dir: Path,
    out_dir: Path,
    seed: int | None = None,
) -> Dict[str, int]:
    polars = _ensure_polars()
    rng = random.Random((seed if seed is not None else cfg.warehouse.seed) + 7)

    dims_path = Path(dims_dir)
    cards_df = polars.read_parquet(dims_path / "dim_card.parquet")
    accounts_df = polars.read_parquet(dims_path / "dim_account.parquet")
    customers_df = polars.read_parquet(dims_path / "dim_customer.parquet")
    merchants_df = polars.read_parquet(dims_path / "dim_merchant.parquet")
    plans_df = polars.read_parquet(dims_path / "dim_plan.parquet")

    account_type_lookup = {
        int(row["account_id"]): row["type"] for row in accounts_df.to_dicts()
    }
    cards = [dict(row) for row in cards_df.to_dicts() if row["status"] == "active"]
    merchants = [dict(row) for row in merchants_df.to_dicts()]
    customers = [dict(row) for row in customers_df.to_dicts()]

    if not merchants:
        raise ValueError("dim_merchant must exist before generating transactions.")

    plan_lookup = {int(row["plan_id"]): row for row in plans_df.to_dicts()}
    premium_plan = plan_lookup.get(3) or next(iter(plan_lookup.values()))

    base_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    months = max(cfg.warehouse.months, 1)
    total_days = months * 30
    end_time = base_start + timedelta(days=total_days)

    day_of_week_factor = {
        0: 0.94,
        1: 0.96,
        2: 0.99,
        3: 1.02,
        4: 1.08,
        5: 1.25,
        6: 1.18,
    }

    base_type_rate = {
        "credit": 2.3,
        "checking": 1.6,
        "savings": 0.35,
    }

    transactions: list[dict[str, Any]] = []
    txn_id = 1
    merchant_ids = [int(row["merchant_id"]) for row in merchants]

    for card in cards:
        account_id = int(card["account_id"])
        account_type = account_type_lookup.get(account_id, "checking")
        activated_at: datetime = card["activated_at"]
        card_start = max(base_start, activated_at)
        start_offset_days = max(0, (card_start - base_start).days)

        for day_offset in range(start_offset_days, total_days):
            day_start = base_start + timedelta(days=day_offset)
            if day_start >= end_time:
                break

            dow = day_start.weekday()
            seasonal_factor = 1 + 0.12 * math.sin((2 * math.pi * day_offset) / 30.0)
            lam = (
                base_type_rate.get(account_type, 1.0)
                * day_of_week_factor[dow]
                * seasonal_factor
            )
            lam = max(lam, 0.05)
            events_today = poisson_sample(rng, lam)
            if events_today == 0:
                continue

            for _ in range(events_today):
                minute_of_day = int(
                    min(
                        max(
                            rng.triangular(0, 24 * 60, 14 * 60),
                            0,
                        ),
                        24 * 60 - 1,
                    )
                )
                event_time = day_start + timedelta(minutes=minute_of_day)
                if event_time >= end_time:
                    continue

                merchant_id = rng.choice(merchant_ids)
                amount = int(max(150, rng.lognormvariate(3.8, 0.6) * 100))
                interchange_bps = round(rng.uniform(80.0, 240.0), 2)
                channel = rng.choices(
                    ["card_present", "card_not_present", "digital_wallet"],
                    weights=[0.55, 0.25, 0.20],
                )[0]
                auth_result = (
                    "captured"
                    if rng.random() < 0.965
                    else rng.choice(["declined", "reversed"])
                )

                transactions.append(
                    {
                        "txn_id": txn_id,
                        "card_id": int(card["card_id"]),
                        "merchant_id": merchant_id,
                        "event_time": event_time,
                        "amount_cents": amount,
                        "interchange_bps": interchange_bps,
                        "channel": channel,
                        "auth_result": auth_result,
                        "loaded_at": event_time,
                    }
                )
                txn_id += 1

    invoices: list[dict[str, Any]] = []
    invoice_id = 1
    churn_rate = 0.018
    subscriber_rate = 0.08
    plan_id = int(premium_plan["plan_id"])
    price_cents = int(premium_plan.get("price_cents", 999))

    for customer in customers:
        customer_created: datetime = customer["created_at"]
        if rng.random() >= subscriber_rate:
            continue

        start_month_index = max(0, (customer_created - base_start).days // 30)
        current_period_start = base_start + timedelta(days=start_month_index * 30)
        active = True

        while active and current_period_start < end_time:
            period_end = current_period_start + timedelta(days=30)
            paid_at = period_end + timedelta(minutes=rng.randint(30, 18 * 60))
            if paid_at > end_time:
                paid_at = period_end

            invoices.append(
                {
                    "invoice_id": invoice_id,
                    "customer_id": int(customer["customer_id"]),
                    "plan_id": plan_id,
                    "period_start": current_period_start,
                    "period_end": period_end,
                    "paid_at": paid_at,
                    "amount_cents": price_cents,
                    "loaded_at": paid_at,
                }
            )
            invoice_id += 1

            current_period_start = period_end
            if rng.random() < churn_rate:
                active = False

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    return {
        "fact_card_transaction": _write_table(
            "neobank", "fact_card_transaction", transactions, out_path, polars
        ),
        "fact_subscription_invoice": _write_table(
            "neobank", "fact_subscription_invoice", invoices, out_path, polars
        ),
    }


def generate_marketplace_dims(
    cfg: AppConfig, out_dir: Path, seed: int | None = None
) -> Dict[str, int]:
    polars = _ensure_polars()
    rng = random.Random(seed if seed is not None else cfg.warehouse.seed + 101)

    scale = cfg.warehouse.scale.lower()
    if scale == "evaluation":
        buyer_count = 9000
        seller_count = 2500
    else:
        buyer_count = 3000
        seller_count = 800

    listings_target = seller_count * 12

    base_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    months = max(cfg.warehouse.months, 1)
    total_days = months * 30

    buyers = []
    for bid in range(1, buyer_count + 1):
        created_at = base_start + timedelta(
            days=rng.randrange(total_days), minutes=rng.randrange(24 * 60)
        )
        country = rng.choices(
            ["US", "CA", "GB", "AU", "DE", "FR", "ES", "MX"],
            weights=[0.55, 0.08, 0.07, 0.05, 0.08, 0.06, 0.05, 0.06],
        )[0]
        buyers.append({"buyer_id": bid, "created_at": created_at, "country": country})

    sellers = []
    seller_statuses = ["active", "paused", "suspended"]
    status_weights = [0.82, 0.12, 0.06]
    for sid in range(1, seller_count + 1):
        created_at = base_start + timedelta(
            days=rng.randrange(total_days), minutes=rng.randrange(24 * 60)
        )
        country = rng.choices(
            ["US", "CA", "UK", "DE", "FR", "NL", "IT", "JP"],
            weights=[0.58, 0.1, 0.08, 0.07, 0.06, 0.04, 0.04, 0.03],
        )[0]
        sellers.append(
            {
                "seller_id": sid,
                "created_at": created_at,
                "country": country,
                "shop_status": rng.choices(seller_statuses, weights=status_weights)[0],
            }
        )

    categories = []
    subcategory_names = [
        "Accessories",
        "Art",
        "Home",
        "Jewelry",
        "Vintage",
        "Clothing",
        "Craft",
        "Gadgets",
    ]
    category_id = 1
    top_level_ids = []
    for idx in range(10):
        top_id = category_id
        categories.append(
            {"category_id": top_id, "name": f"Category {idx + 1}", "parent_id": None}
        )
        top_level_ids.append(top_id)
        category_id += 1
        for sub_name in subcategory_names:
            categories.append(
                {
                    "category_id": category_id,
                    "name": f"{sub_name} {idx + 1}",
                    "parent_id": top_id,
                }
            )
            category_id += 1

    category_ids = [c["category_id"] for c in categories if c["parent_id"] is not None]
    category_weights = zipf_weights(len(category_ids), 1.1)

    seller_ids = [s["seller_id"] for s in sellers]

    listings = []
    listing_id = 1
    for seller_id in seller_ids:
        listing_count = max(5, int(rng.lognormvariate(2.0, 0.6)))
        for _ in range(listing_count):
            category_id = rng.choices(category_ids, weights=category_weights)[0]
            created_at = base_start + timedelta(
                days=rng.randrange(total_days), minutes=rng.randrange(24 * 60)
            )
            status = rng.choices(
                ["active", "inactive", "draft"], weights=[0.78, 0.12, 0.10]
            )[0]
            price_cents = int(max(500, rng.lognormvariate(4.2, 0.7) * 100))
            listings.append(
                {
                    "listing_id": listing_id,
                    "seller_id": seller_id,
                    "category_id": category_id,
                    "created_at": created_at,
                    "status": status,
                    "price_cents": price_cents,
                }
            )
            listing_id += 1
    if len(listings) < listings_target:
        for seller_id in seller_ids:
            if len(listings) >= listings_target:
                break
            category_id = rng.choices(category_ids, weights=category_weights)[0]
            created_at = base_start + timedelta(
                days=rng.randrange(total_days), minutes=rng.randrange(24 * 60)
            )
            price_cents = int(max(500, rng.lognormvariate(4.2, 0.7) * 100))
            listings.append(
                {
                    "listing_id": listing_id,
                    "seller_id": seller_id,
                    "category_id": category_id,
                    "created_at": created_at,
                    "status": "active",
                    "price_cents": price_cents,
                }
            )
            listing_id += 1

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    counts = {
        "dim_buyer": _write_table("marketplace", "dim_buyer", buyers, out_path, polars),
        "dim_seller": _write_table(
            "marketplace", "dim_seller", sellers, out_path, polars
        ),
        "dim_category": _write_table(
            "marketplace", "dim_category", categories, out_path, polars
        ),
        "dim_listing": _write_table(
            "marketplace", "dim_listing", listings, out_path, polars
        ),
    }
    return counts


def generate_marketplace_facts(
    cfg: AppConfig,
    dims_dir: Path,
    out_dir: Path,
    seed: int | None = None,
) -> Dict[str, int]:
    polars = _ensure_polars()
    rng = random.Random((seed if seed is not None else cfg.warehouse.seed) + 211)

    dims_path = Path(dims_dir)
    buyers_df = polars.read_parquet(dims_path / "dim_buyer.parquet")
    sellers_df = polars.read_parquet(dims_path / "dim_seller.parquet")
    listings_df = polars.read_parquet(dims_path / "dim_listing.parquet")

    buyer_ids = [int(b) for b in buyers_df["buyer_id"]]
    sellers = [dict(row) for row in sellers_df.to_dicts()]

    listings_by_seller: Dict[int, list[dict[str, Any]]] = {}
    for row in listings_df.to_dicts():
        listings_by_seller.setdefault(int(row["seller_id"]), []).append(dict(row))

    seller_ids = [
        int(s["seller_id"])
        for s in sellers
        if listings_by_seller.get(int(s["seller_id"]))
    ]
    seller_weights = zipf_weights(len(seller_ids), 1.05) if seller_ids else []
    if not seller_ids:
        raise ValueError("Listings are required to generate marketplace facts.")

    base_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    months = max(cfg.warehouse.months, 1)
    total_days = months * 30
    end_time = base_start + timedelta(days=total_days)

    day_weights = [0.92, 0.96, 1.0, 1.08, 1.14, 1.32, 1.26]

    base_orders = 130 if cfg.warehouse.scale.lower() == "evaluation" else 60
    orders: list[dict[str, Any]] = []
    order_items: list[dict[str, Any]] = []
    payments: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    order_id = 1
    for day_offset in range(total_days):
        day_start = base_start + timedelta(days=day_offset)
        if day_start >= end_time:
            break
        seasonal = 1 + 0.1 * math.sin((2 * math.pi * day_offset) / 14.0)
        lam = base_orders * day_weights[day_start.weekday()] * seasonal
        events_today = poisson_sample(rng, lam)

        for _ in range(events_today):
            seller_id = rng.choices(seller_ids, weights=seller_weights)[0]
            seller_listings = listings_by_seller[seller_id]
            if not seller_listings:
                continue
            buyer_id = rng.choice(buyer_ids)
            order_time = day_start + timedelta(minutes=rng.randrange(24 * 60))
            if order_time >= end_time:
                continue

            item_count = rng.choices([1, 2, 3, 4], weights=[0.55, 0.28, 0.12, 0.05])[0]
            subtotal = 0
            line_items = []
            for line_idx in range(1, item_count + 1):
                listing = rng.choice(seller_listings)
                qty = max(1, int(rng.lognormvariate(0.2, 0.6)))
                price = int(listing["price_cents"])
                subtotal += price * qty
                line_items.append(
                    {
                        "order_id": order_id,
                        "line_id": line_idx,
                        "listing_id": int(listing["listing_id"]),
                        "seller_id": seller_id,
                        "qty": qty,
                        "item_price_cents": price,
                        "loaded_at": order_time,
                    }
                )

            if subtotal <= 0:
                continue

            discount = int(subtotal * 0.05) if rng.random() < 0.2 else 0
            taxable_amount = subtotal - discount
            tax_cents = int(round(taxable_amount * 0.0875))
            shipping_cents = int(rng.uniform(400, 900))
            platform_fee_cents = int(round(subtotal * 0.12))
            buyer_paid = taxable_amount + tax_cents + shipping_cents
            seller_earnings = taxable_amount - platform_fee_cents

            orders.append(
                {
                    "order_id": order_id,
                    "buyer_id": buyer_id,
                    "order_time": order_time,
                    "currency": "USD",
                    "subtotal_cents": subtotal,
                    "tax_cents": tax_cents,
                    "shipping_cents": shipping_cents,
                    "discount_cents": discount,
                    "loaded_at": order_time,
                }
            )
            order_items.extend(line_items)

            captured_at = min(
                order_time + timedelta(minutes=rng.randint(30, 12 * 60)), end_time
            )
            payments.append(
                {
                    "order_id": order_id,
                    "captured_at": captured_at,
                    "buyer_paid_cents": buyer_paid,
                    "seller_earnings_cents": max(seller_earnings, 0),
                    "platform_fee_cents": platform_fee_cents,
                    "loaded_at": captured_at,
                }
            )
            order_id += 1

    final_snapshot_date = (end_time - timedelta(days=1)).date()
    for listing in listings_df.to_dicts():
        snapshots.append(
            {
                "ds": final_snapshot_date,
                "listing_id": int(listing["listing_id"]),
                "status": listing["status"],
            }
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    counts = {
        "fact_order": _write_table(
            "marketplace", "fact_order", orders, out_path, polars
        ),
        "fact_order_item": _write_table(
            "marketplace", "fact_order_item", order_items, out_path, polars
        ),
        "fact_payment": _write_table(
            "marketplace", "fact_payment", payments, out_path, polars
        ),
        "snapshot_listing_daily": _write_table(
            "marketplace", "snapshot_listing_daily", snapshots, out_path, polars
        ),
    }
    return counts


def _empty_dataframe(schema: Iterable[Tuple[str, "pl.DataType"]]):
    polars = _ensure_polars()
    series = [polars.Series(name, [], dtype=dtype) for name, dtype in schema]
    if not series:
        return polars.DataFrame()
    return polars.DataFrame(series)


def _write_table(
    archetype: str,
    table: str,
    records: Iterable[MutableMapping[str, Any]],
    out_path: Path,
    polars,
) -> int:
    schema = WAREHOUSE_SCHEMAS[archetype][table]
    record_list = list(records)
    if record_list:
        df = polars.DataFrame(record_list)
        df = df.select([polars.col(column).cast(dtype) for column, dtype in schema])
    else:
        df = _empty_dataframe(schema)
    write_parquet_atomic(out_path / f"{table}.parquet", df)
    return df.height


def _ensure_polars():
    if pl is None:
        raise RuntimeError(
            "polars is required for warehouse generation. "
            "Install it to enable this feature."
        )
    return pl


__all__ = [
    "NEOBANK_TABLE_SCHEMAS",
    "MARKETPLACE_TABLE_SCHEMAS",
    "WAREHOUSE_SCHEMAS",
    "NEOBANK_MERCHANT_SECTORS",
    "NEOBANK_MCC_TO_SECTOR",
    "write_empty_warehouse",
    "generate_neobank_dims",
    "generate_neobank_facts",
    "generate_marketplace_dims",
    "generate_marketplace_facts",
]
