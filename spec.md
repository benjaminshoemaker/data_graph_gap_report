# Data-Needs Reporter — MVP Developer Specification

## Scope

CLI-only MVP that generates synthetic warehouse data and communications, classifies themes, computes data-health metrics, and emits a decision report. No ingestion from real systems. No web UI. No owners, tickets, or dashboards in outputs.

## Personas

* Runner: data leader.
* Audience: executives.
* Goal: credible top-3 actions backed by data gaps and demand signals.

## Architecture

### Stack

* Python 3.11, Typer, Poetry.
* DuckDB + Parquet for storage.
* PyArrow/Polars for data frames.
* Local time zone for business rules: `America/Los_Angeles`. Store timestamps in UTC.

### Repo

```
data-needs-reporter/
├─ pyproject.toml
├─ README.md
├─ LICENSE                 # none for MVP, file can be empty or omitted
├─ .gitignore
├─ .env.example
├─ configs/
│  └─ default.yaml
├─ meta/                   # oracle labels
├─ data/                   # generated warehouses
├─ comms/                  # generated corpora
├─ reports/                # outputs
├─ tests/
│  ├─ test_generators.py
│  ├─ test_metrics.py
│  ├─ test_classifier.py
│  └─ test_cli.py
└─ src/data_needs_reporter/
   ├─ __init__.py
   ├─ cli.py
   ├─ config.py            # Pydantic models + YAML load/validate
   ├─ utils/{io.py,duck.py,rand.py,cost_guard.py,logging.py}
   ├─ generate/{warehouse.py,comms.py,defects.py}
   ├─ report/{metrics.py,scoring.py,classify.py,run.py,plots.py}
   └─ eval/labels_eval.py
```

### Data Flow

1. `gen-warehouse` → Parquet + DuckDB database per archetype.
2. `gen-comms` → Slack, Email, NLQ Parquet corpora.
3. `run-report` → prefilter + sample-to-fit + LLM classify + LLM entity extract + metrics + scoring → `exec_summary.json`, `data_health.{json,csv}`, `themes.json`, figures.
4. Optional: `eval-labels` against oracle labels.
5. `validate` quality gates.

## CLI

### Commands

* `dnr init`
* `dnr gen-warehouse --archetype {neobank|marketplace} --scale evaluation --quality typical --trajectory T1 --tz America/Los_Angeles --months 18 --seed 42 --out data/{arch}`
* `dnr gen-comms --archetype {neobank|marketplace} --slack 3000 --email 800 --nlq 1000 --seed 43 --out comms/{arch}`
* `dnr run-report --warehouse data/{arch} --comms comms/{arch} --out reports/{arch} --window 30d --api-cap-usd 1.0`
* `dnr eval-labels --pred runs/{run_id}/preds --labels meta --out reports/eval/{run_id}`
* `dnr validate --warehouse data/{arch} --comms comms/{arch} --out reports/{arch}/qc --strict`
* `dnr quickstart [--fast] [--no-llm] [--seed N]`

### Config precedence

Default → YAML (`configs/default.yaml` or `--config`) → env → CLI flags.

### Install

* Primary: PyPI + pipx (`pipx install data-needs-reporter`).
* Dev: Poetry.

## Synthetic Warehouse

### Archetypes

* **B2C neobank** and **Etsy-like marketplace**. Scale: **Evaluation**. Quality profile: **Typical**. Trajectory: **T1 steady baseline**.
* Time: 18 months of events. Event grain: minute. Reporting grain: daily + monthly.

### Schemas

#### Neobank

* `dim_customer(customer_id, created_at, kyc_status)`
* `dim_account(account_id, customer_id, type, created_at, status)`
* `dim_card(card_id, account_id, status, activated_at)`
* `dim_merchant(merchant_id, mcc, name)`
* `dim_plan(plan_id, name, price_cents, cadence)`
* `fact_card_transaction(txn_id, card_id, merchant_id, event_time, amount_cents, interchange_bps, channel, auth_result, loaded_at)`
* `fact_subscription_invoice(invoice_id, customer_id, plan_id, period_start, period_end, paid_at, amount_cents, loaded_at)`

#### Marketplace

* `dim_buyer(buyer_id, created_at, country)`
* `dim_seller(seller_id, created_at, country, shop_status)`
* `dim_category(category_id, name, parent_id)`
* `dim_listing(listing_id, seller_id, category_id, created_at, status, price_cents)`
* `fact_order(order_id, buyer_id, order_time, currency, subtotal_cents, tax_cents, shipping_cents, discount_cents, loaded_at)`
* `fact_order_item(order_id, line_id, listing_id, seller_id, qty, item_price_cents, loaded_at)`
* `fact_payment(order_id, captured_at, buyer_paid_cents, seller_earnings_cents, platform_fee_cents, loaded_at)`
* `snapshot_listing_daily(ds, listing_id, status)`

### Taxonomies

* Neobank merchants: **compact** 40 MCC across 8 sectors; ~6k merchants.
* Marketplace categories: **compact** 10 top-level × ~8 leaf each; ~200k listings.
* Orders are single-seller; `fact_payment` one row per order.

### Behavioral Generation (key points)

* Neobank spend volume via non-homogeneous Poisson; lognormal amounts; MCC mix; auth_result rates; 8% premium subs at $9.99; churn 1.8%/mo.
* Marketplace buyer propensity ≈0.8 orders/buyer/mo; items/order distribution; category price medians; weekend and hour peaks.
* All timestamps stored UTC; daily windows computed in `America/Los_Angeles`.

### Defect Injection (Typical)

* Key nulls 2%±0.5 on required keys.
* FK fail 5%±1 (70% missing id, 30% temporal mismatch).
* Duplicates 0.7%±0.2 (per table’s business key).
* p95 ingest lag mostly 30–180 min; 5% between 4–12 h; `loaded_at = event_time + lag`.
* Null spikes: two per quarter, +10% absolute nulls for one day on specified columns.
* One minor schema write gap per archetype with backfill.
* Correlated comms generated near events.

### Monetization constants

* **Neobank**: Interchange mean 1.5% bps by MCC; Subscription $9.99; GMV = sum captured txn amounts; Revenue = Interchange + Subscription.
* **Marketplace**: GMV = sum(qty*item_price_cents) only; take rate 12%; Net revenue = platform fees; payments captured order-day.

### KPI definitions

* Neobank: Active customers (30d), New account opens, Card spend, Interchange revenue, Subscription revenue.
* Marketplace: GMV, Take rate, Net revenue, Orders & AOV, Active buyers (30d), Active sellers & live listings.
* KPI filters de-dup by earliest `loaded_at`; include only captured payments/txns.

## Synthetic Communications

### Volumes and window

* Window: **12 months**.
* Per archetype caps: Slack **3k threads** (4–8 msgs/thread, ≤80 tokens/msg), Email **800 messages** (≤180 tokens), NLQ **1k** (≤30 tokens).

### PII and IDs

* **None**. Use synthetic IDs only (`u001`, `t042`, `m901`, etc.). No real names, emails, or URLs.

### Roles (for weighting only)

Exec 3%, data_eng 12%, analytics_eng 8%, bi 7%, product_analytics 5%, eng_platform 10%, pm 10%, finance 5%, other 40%.

### Category buckets

* `platform`, `analytics`, `domain` on both Slack and Email. Stored per message: `channel_cat` or `list_cat`.

### Schemas

`slack_messages.parquet`

* `message_id, thread_id, channel_id, channel_cat, ts, user_id, text, reply_to, reply_count, has_link, link_domains[], has_attachment, attachment_names[], has_code`

`email_messages.parquet`

* `message_id, thread_id, ts, from_id, to_ids[], cc_ids[], list_cat, subject, body_text, has_attachment, attachment_names[], has_link, link_domains[], has_code`

`comms_users.parquet`

* `source("slack"|"email"), user_id, role`

`nlq.parquet`

* `query_id, user_id, app, ts, query_text, generated_sql, success_flag, latency_ms, result_rows`

### Generation

* **LLM-only** text generation with **$1 cap per archetype**; concurrency 5; caching on disk (`.cache/llm`).
* Slack thread template: 4–8 messages, IDs only, allowed link domains placeholders, event flags respected.
* Email template: single message bodies, IDs only.
* NLQ template: short query text + lightweight SQL. **Do not execute** by default. Flags: `--nlq-parse-only`, `--nlq-execute` (synthetic only).

## Prefiltering and Sampling

### Prefilter (Slack/Email)

* Bucket allowlist.
* Keyword score upweights data terms; downweights near-match negatives.
* Structural signals: SQL/code/link presence.
* Role weighting and thread rules (root + replies).
* Event correlation boost within ±24h of injected spikes/gaps.
* Keep threads with score ≥0.35, else top 30/day/channel by score. Persist `prefilter_score` and reasons.

### Budgeting: sample-to-fit

* Mode: `sample_to_fit` with **25% safety margin**.
* Preflight p90 tokens to estimate cost.
* Allocate quotas per source and per day×category bucket with minimum per bucket; random sampling with seed.
* Coverage floors: **20%** per source after prefiltering and budgeting. If a source misses the floor, **continue** and keep weights unchanged (behavior C).
* Demand weights: base NLQ 0.50, Slack 0.30, Email 0.20, then reweight by volume; cap final per-source weight to [0.15, 0.60].

## LLM Classification

### Engine

* Provider: OpenAI. Model: `gpt-4o-mini`. Temperature 0.0. Max output tokens 64. Concurrency 5.
* **Cap**: **$1.00 per archetype**. Hard stop before overrun.

### Scope

* Slack/Email: thread-level labels using all kept messages; NLQ: theme only.

### Labels

* Relevance: `positive | hard_negative | negative` (Slack/Email only).
* Themes: `metric_definition_mismatch | join_keys_orphans | freshness_ingest_lag | duplicates_dedup | access_permissions_pii | new_field_schema_request | revenue_coverage_mapping | attribution_funnel_gaps`.

### Thread input strategy

* **Role-aware**: always include root and all exec messages; then top-scored messages newest→oldest up to **900 input tokens** and **max 20 messages**.

### Prompt

* System and JSON I/O as approved. Force JSON only.

### Met vs unmet examples

* Use **heuristic** inference on Slack/Email (keywords like fixed/merged/resolved, links to dbt/Looker, closure patterns).
* NLQ examples can appear but do not define met/unmet on their own.

### Safeguards

* Timeout 15s, 2 retries with backoff, JSON repair once, then `parse_error`.
* Stop a source after 50 consecutive parse errors; others continue.
* On cap hit: sample-to-fit prevents mid-run failure.

## LLM Entity Extraction

* Scope: Slack/Email with `relevance=positive` and all NLQs.
* Model: `gpt-4o-mini`. Cap **$0.10 per archetype**.
* Input: item text + known table/column dictionaries per archetype.
* Output: `tables[{name,confidence}]`, `columns[{table,column,confidence}]`. Persist `entities.parquet`.
* Caching enabled.

## Metrics and Scoring

### Data-health metrics (`report/metrics.py`)

* Key-null %, FK success %, Orphan %, Duplicate %, p95 ingest lag, null spikes detection.
* Temporal join rule: `fact_ts >= dim.created_at` when present.
* Outputs per table and overall SLO compliance. Daily series drive figures.
* MVP-complete guardrails:
  - Invoice aggregates (neobank): compute `missing_pct`, `on_time_pct`, `dup_key_pct`, `p95_payment_lag_min` from `fact_subscription_invoice`. Thresholds via `report.invoice_aggregates.slos` (max/min per metric). Results surfaced in `data_health.json` under `aggregates_by_table.fact_subscription_invoice` and in `invoice_aggregates{metrics,checks,passed}`.
  - Evening window (marketplace): compute evening share and qualifying-day %, with thresholds from `report.marketplace.evening_window{min_share_pct,min_days_pct}`. Results surfaced in `data_health.json` under `marketplace_evening_window{daily_share_pct,overall_share_pct,days_pct,checks,passed}`.

### SLOs (config)

* Key null ≤1.0%, Orphans ≤2.0%, Dup keys ≤0.2%, p95 lag ≤90 min.

### Demand computation

* Weighted theme shares across sources using reweighted and capped source weights and post-stratified thread weights.

### Prioritization

* Score per gap: `0.60*RevenueImpact + 0.25*Demand + 0.15*HealthSeverity`, decayed by `exp(−days_since_peak/60)`.
* RevenueImpact: neobank uses Interchange + Subs at risk; marketplace uses net revenue at risk. Normalize by trailing 3-month median monthly revenue.
* HealthSeverity: scaled SLO overage.
* Output top-3 actions with **theme diversity enforced**.

### Confidence and gating

* Confidence = `0.45*ClassConf + 0.25*EntityConf + 0.20*Coverage + 0.10*Agreement`.
* Hide actions with Confidence < 0.55.

## Reports

### Files

* `reports/{arch}/exec_summary.{json,md}`
* `reports/{arch}/data_health.{json,csv}`
* `reports/{arch}/themes.{json,md}`
* `reports/{arch}/figures/*.png`
* `reports/index.md` (from `quickstart`)
* Guardrail additions (MVP complete):
  - `data_health.json` includes `aggregates_by_table.fact_subscription_invoice` and `invoice_aggregates` (when enabled or thresholds configured).
  - `data_health.json` includes `marketplace_evening_window` with thresholds and `checks` when marketplace payments exist.
  - `exec_summary.md` includes a “Data Health” section with any failing guardrails, plus detailed “Invoice Aggregates” and “Marketplace Evening Coverage” summaries.

### Schemas

Use the approved `exec_summary.json`, `data_health.json`, `themes.json`, plus long-form CSV. Report window default **30 days**.

## Oracle Labels and Evaluation

* Oracle label files under `meta/` for Slack, Email, NLQ (themes; plus relevance for Slack/Email).
* `eval-labels` joins predictions by id and outputs accuracy, macro-F1, per-class metrics, confusion matrices, calibration, coverage, and CIs.
* Default gates: Slack/Email macro-F1(theme) ≥0.72 and macro-F1(relevance) ≥0.75; NLQ macro-F1(theme) ≥0.72; coverage ≥0.95.

## Validation Suite

`dnr validate --strict` fails if any gate is breached:

* Schema presence and types.
* Volume targets within ±10%.
* Data-quality targets hit; spikes detected.
* Seasonality and hour-of-day patterns.
* Taxonomy coverage and caps.
* Monetization sanity bands.
* Trajectory sanity.
* Comms volumes and bucket mix.
* Theme mix within ±8 pp.
* NLQ token caps and parse rates.
* Correlation hooks near injected events.
* Reproducibility with same seeds.
* LLM spend ≤ caps.
* MVP-complete guardrails:
  - Invoice aggregate SLOs (missing, on-time, dup keys, p95 payment lag) evaluated against configured thresholds.
  - Marketplace evening window guardrails (overall evening share ≥ min_share_pct; qualifying days ≥ min_days_pct).

## Error Handling and Exit Codes

* `run-report` exit policy **B**: non-zero only on internal errors.
* `--strict` makes `run-report` non-zero if any source <20% coverage or if budget forced early stop.
* Budgeting writes `{run}/budget.json` with cap, estimate, actual, and bucket coverage.
* JSON parse errors excluded from demand; counts reported.
* LLM network failures after retries mark source partial; continue others.

## Performance Budgets (Eval scale, 4-core laptop)

* `gen-warehouse` ≤ 3 min/archetype
* `gen-comms` ≤ 4 min/archetype
* `run-report` ≤ 5 min/archetype
* `validate` ≤ 3 min/archetype
* Peak RAM ≤ 4 GB

## Config (excerpt)

```yaml
paths: {data: "data", comms: "comms", reports: "reports", meta: "meta"}

warehouse: {archetypes: ["neobank","marketplace"], scale: "evaluation", quality: "typical",
  trajectory: "T1", tz: "America/Los_Angeles", months: 18, seed: 42}

comms: {slack_threads: 3000, email_threads: 800, nlq: 1000, sample_policy: "event_aware", seed: 43}

classification:
  engine: "llm"
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.0
  max_output_tokens: 64
  concurrency: 5
  api_cap_usd: 1.0
  prefilter_threshold: 0.35
  env_key_var: "OPENAI_API_KEY"

entities:
  provider: "openai"
  model: "gpt-4o-mini"
  api_cap_usd: 0.10

report:
  scoring_weights: {revenue: 0.60, demand: 0.25, severity: 0.15}
  slos: {key_null_pct: 1.0, fk_orphan_pct: 2.0, dup_keys_pct: 0.2, p95_ingest_lag_min: 90}
  window_days: 30
  demand_base_weights: {nlq: 0.50, slack: 0.30, email: 0.20}
  demand_weight_caps: {min: 0.15, max: 0.60}
  revenue_norm: "median_3m"

budget:
  mode: "sample_to_fit"
  safety_margin: 0.25
  coverage_floor_pct: 20
cache: {enabled: true, dir: ".cache/llm"}
```

## Testing Plan

### Unit

* Generators: schema, volumes, seasonality, monetization bands.
* Defects: nulls, duplicates, FK failures, lag distribution, spikes.
* Metrics: key-null, orphan, dup, p95 lag, spike detection, SLO evaluation.
* Scoring: component normalization, decay, diversity selection.

### Integration

* End-to-end: `quickstart --no-llm` should produce `data_health` and “themes skipped”.
* With LLM: seeded run respecting caps; verify `budget.json`, coverage floors, and cached calls.
* Validation: `validate --strict` on fresh outputs passes with Typical profile.
* MVP-complete: invoice aggregates SLOs and marketplace evening-window guardrails reported and validated.

### Golden files

* Seeded small run snapshots for `data_health.json`, `themes.json` structure, and plots hashes.

### Performance

* Timings under budgets; memory profile under 4 GB.

### Error paths

* Simulate parse failures; confirm exclusion and reporting.
* Force cap breach via tiny cap; verify sample-to-fit stays under and marks coverage.

## Non-Goals (MVP)

* No real connectors.
* No JIRA/merge requests.
* No owners or routing.
* No web UI.
* No license decision.

## Runbook

### Quickstart

```
pipx install data-needs-reporter
dnr quickstart --seed 42
```

### Typical dev loop

```
dnr gen-warehouse --archetype neobank --out data/neobank
dnr gen-comms --archetype neobank --out comms/neobank
dnr run-report --warehouse data/neobank --comms comms/neobank --out reports/neobank
dnr validate --warehouse data/neobank --comms comms/neobank --out reports/neobank/qc --strict
```

This is complete for implementation.
