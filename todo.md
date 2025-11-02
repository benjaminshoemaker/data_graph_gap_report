# Data‑Needs Reporter — TODO

Use this as a strict checklist. Each item has clear acceptance. Keep runs seeded.

---

## 0. Repo + Tooling

- [x] Create repo skeleton per spec
  - [x] `pyproject.toml` with Poetry, deps: typer[all], pydantic, duckdb, pyarrow, polars, matplotlib, pytest
  - [x] `src/data_needs_reporter/` package scaffold
  - [x] `tests/` scaffold with pytest.ini
  - [x] `.gitignore`, `.env.example`, `README.md`
- [x] Configure console script `dnr`
- [x] Add pre-commit hooks (black, isort, ruff)
- [ ] CI: GitHub Actions
  - [x] Python 3.11 matrix on Linux/macOS
  - [x] Cache Poetry and pip
  - [x] Run unit + integration tests
  - [ ] Upload coverage
- [ ] Makefile/justfile targets
  - [ ] `make test`, `make lint`, `make fmt`, `make e2e`

**Done when:** `dnr --help` works, CI is green on empty skeleton.

---

## 1. Config + CLI

- [x] Implement `config.py` (Pydantic)
  - [x] Models mirror approved config excerpt
  - [x] `load_config` precedence: defaults → YAML → env → flags
- [x] Implement `cli.py` Typer app
  - [x] Global flags: `--config`, `--api-cap-usd`, `--window`, `--no-llm`
  - [x] Command stubs:
    - [x] init
    - [x] gen-warehouse
    - [x] gen-comms
    - [x] run-report
    - [x] validate
    - [x] eval-labels
    - [x] quickstart
- [x] Default YAML under `configs/default.yaml`
- [x] Tests
  - [x] `dnr --version` prints `0.1.0`
  - [x] Precedence tests (env beats YAML, flags beat env)

**Done when:** CLI parses flags, config validates, tests pass.

---

## 2. Utils

- [x] `utils/io.py`
  - [x] Atomic JSON/Parquet/CSV read‑write
  - [x] Auto‑mkdir, temp‑rename
- [x] `utils/duck.py`
  - [x] `open_db`, `attach_parquet_dir`, `safe_query` with basic guards
- [x] `utils/logging.py`
  - [x] `init_logger`, run_id context manager
- [ ] `utils/rand.py`
  - [ ] Seeded RNG helpers, distributions used by generators
- [x] `utils/cost_guard.py`
  - [x] Token estimate, cap tracking, write `budget.json`

**Done when:** round‑trip IO, simple DuckDB query, logs carry run_id in tests.

---

## 3. Warehouse: Schemas + Baseline Generation

- [x] Schema constants for both archetypes (columns + dtypes)
- [x] Command: `dnr gen-warehouse --dry-run` writes empty Parquet with correct schema
- **Neobank**
  - [x] Generate dims: customer, account, card, merchant, plan
  - [x] Generate facts: card transactions, subscription invoices (baseline, no defects)
- **Marketplace**
  - [x] Generate dims: buyer, seller, category, listing
  - [x] Generate facts: order, order_item, payment, snapshot_listing_daily
- [ ] Attach DuckDB and run sanity SQL
  - [ ] Time bounds correct
  - [ ] Basic KPI queries run

**Done when:** row counts within Evaluation targets (±10% allowed before defects), keys unique, RI holds.

---

## 4. Defect Injection (Typical)

- [x] Implement `defects.py`
  - [x] Key nulls (2%±0.5 on required keys)
  - [x] FK failures (5%±1; 70% missing id, 30% temporal)
  - [x] Duplicates (0.7%±0.2 of business keys)
  - [x] Ingest lag with p95 120–240 min; 5% tail 4–12h; set `loaded_at`
  - [x] Null spikes 2×/qtr (+10% absolute) for one day, per target columns
  - [x] Schema gap window + backfill (per archetype)
- [ ] Calibration loop emits `data_quality_summary.json`

**Done when:** post‑injection metrics measured in tests match targets.

---

## 5. Comms: Schemas, Users, LLM Generation

- [x] Write Parquet schemas:
  - [x] `slack_messages`, `email_messages`, `comms_users`, `nlq`
- [x] Users with role mix; IDs only
- [x] LLM client + on‑disk cache (`.cache/llm`)
  - [x] Provider: OpenAI; model: `gpt-4o-mini`; JSON‑only guard; timeout 15s; retries 2; repair once
  - [x] Mock client for tests
- [x] Generate Slack/Email/NLQ
  - [x] Volumes: Slack 3k threads, Email 800, NLQ 1k; 12‑month window
  - [x] Token caps per item
  - [ ] Allowed link domains placeholders
  - [x] `$1` cap per archetype with `cost_guard`
  - [x] Write `budget.json`

**Done when:** generation stops before cap, volumes within ±5%, cache hits on rerun.

---

## 6. Prefilter + Sampling + Coverage

- [x] Implement prefilter scoring (bucket allowlist, keywords, structure, roles, event correlation)
- [x] Threshold 0.35; else top‑30/day/channel fallback
- [x] Preflight token p90 per source
- [ ] Sample‑to‑fit allocator
  - [x] 25% safety margin
  - [x] Day×bucket quotas with minimum per bucket
- [x] Coverage floors 20% per source
  - [x] Behavior C if floor missed (continue, weights unchanged)

**Done when:** `budget.json` reports estimates, actuals, coverage by bucket; floors enforced.

---

## 7. Metrics Engine + Plots

- [ ] Compute per‑table:
  - [ ] key_null_pct, fk_success_pct, orphan_pct, dup_key_pct, p95_ingest_lag_min
  - [ ] Null spike detection (daily vs 7‑day median +8.0)
- [x] SLO check using config thresholds
- [x] Plots:
  - [x] `lag_p95_daily.png`, `key_null_pct_daily.png`, `orphan_pct_daily.png`, `dup_key_pct_bar.png`, `theme_demand_monthly.png`

**Done when:** values match injected signals; images written with stable hashes in tests.

---

## 8. Classification

- [ ] Role‑aware thread packing (root + all execs + top‑scored newest→oldest)
  - [ ] Caps: ≤900 input tokens, ≤20 messages
- [x] LLM classify
  - [x] Slack/Email: theme + relevance
  - [x] NLQ: theme only
  - [x] Guardrails: parse repair, 50 consecutive parse error stop per source
- [x] Save predictions Parquet with confidences
- [ ] Extract tables/columns via LLM
  - [ ] Cap spend $0.10
  - [ ] Cache reused across runs
  - [ ] Save `entities.parquet`

**Done when:** predictions saved; parse errors excluded from demand with counts.

---

## 9. Entity Extraction

- [ ] LLM extract tables/columns from positives (Slack/Email) and all NLQ
- [ ] Use known dictionaries; return up to 5 tables and 8 columns with confidences
- [ ] Cap spend at $0.10 per archetype; cache enabled
- [ ] Save `entities.parquet`

**Done when:** outputs limited to known names; confidences in [0,1].

---

## 10. Demand, Revenue, Severity, Recency

- [ ] Demand
  - [ ] Post‑stratified per item weights
  - [ ] Source weights base: NLQ 0.50, Slack 0.30, Email 0.20
  - [ ] Reweight by volume; cap per source to [0.15, 0.60]
- [ ] Revenue impact
  - [ ] Neobank: Interchange + Subs at risk
  - [ ] Marketplace: Net revenue at risk
  - [ ] Normalize by trailing 3‑month median monthly revenue
- [x] Health severity vs SLO overage
- [x] Recency decay: exp(−days_since_peak/60)

**Done when:** unit tests verify formula outputs on fixtures.

---

## 11. Prioritization, Confidence, Diversity

- [x] Score = 0.60*Revenue + 0.25*Demand + 0.15*Severity
- [ ] Enforce theme diversity for top‑3
- [x] Confidence = 0.45*ClassConf + 0.25*EntityConf + 0.20*Coverage + 0.10*Agreement
- [x] Gate: hide actions with Confidence < 0.55

**Done when:** synthetic cases show correct diversity and gating.

---

## 12. Reports

- [x] `data_health.json` and `data_health.csv` per schema
- [x] `themes.json` and `themes.md` with examples
- [x] `exec_summary.json` and `exec_summary.md` with top‑3
- [x] Figures saved under `figures/`
- [x] Validate against JSON schemas in tests

**Done when:** schema validation passes; files exist; MD renders.

---

## 13. Validation Suite

- [ ] `dnr validate --strict`
  - [ ] Schema presence and types
  - [ ] Volume ±10%
  - [ ] Quality targets (nulls, orphans, dup, lag)
  - [ ] Seasonality and hour peaks
  - [ ] Taxonomy and category caps
  - [ ] Monetization sanity bands
  - [ ] Trajectory sanity
  - [ ] Comms volumes and bucket mix
  - [ ] Theme mix within ±8 pp
  - [ ] NLQ token caps; parse‑only success ≥95% (if enabled)
  - [ ] Event correlation hooks
  - [ ] Reproducibility with same seeds
  - [ ] LLM spend ≤ caps

**Done when:** failing gate returns non‑zero under `--strict`; passing run returns zero.

---

## 14. Oracle Labels + Evaluation

- [ ] `meta/labels_slack.parquet`, `meta/labels_email.parquet`, `meta/labels_nlq.parquet`
- [ ] `dnr eval-labels`
  - [ ] Accuracy, macro‑F1, per‑class P/R/F1
  - [ ] Confusion matrices, ECE if probs
  - [ ] 1k bootstrap CIs
  - [ ] Gates: Slack/Email macro‑F1(theme) ≥0.72; relevance ≥0.75; NLQ theme ≥0.72; coverage ≥0.95
- [ ] Outputs under `reports/eval/{run_id}`

**Done when:** metrics computed; gates enforced.

---

## 15. Quickstart + Packaging

- [ ] `dnr quickstart` runs init → gen‑warehouse/comms → run‑report for both archetypes
  - [ ] `--fast` (10% volumes)
  - [ ] `--no-llm` outputs data‑health and “themes skipped”
  - [ ] Write `reports/index.md`
- [ ] Packaging metadata for PyPI
  - [ ] `dnr` entry point tested
  - [ ] Version bump script

**Done when:** `pipx install` and `dnr quickstart` work locally.

---

## 16. Performance + Memory

- [ ] Ensure Evaluation budgets on 4‑core laptop
  - [ ] `gen-warehouse` ≤ 3 min/archetype
  - [ ] `gen-comms` ≤ 4 min/archetype
  - [ ] `run-report` ≤ 5 min/archetype
  - [ ] `validate` ≤ 3 min/archetype
  - [ ] Peak RAM ≤ 4 GB
- [ ] Add simple benchmark script and CI timing guard (informational)

**Done when:** local benchmark meets targets.

---

## 17. Error Handling + Exit Codes

- [ ] Non‑zero only on internal errors (`run-report`)
- [ ] `--strict` escalates coverage/budget breaches to non‑zero
- [ ] LLM parse error policy: repair once, count excluded, stop after 50 consecutive per source
- [ ] Budget guard never exceeds cap; sample‑to‑fit allocator stops early
- [ ] Atomic writes avoid partial artifacts

**Done when:** simulated failures behave per policy; logs clear.

---

## 18. Docs

- [ ] README sections
  - [x] Quickstart
  - [x] Commands and flags
  - [x] Config precedence
  - [ ] Seeds and reproducibility
  - [ ] Spend caps and budgeting
  - [ ] Validation and eval
  - [x] Troubleshooting (parse errors, low coverage, cap hits)
- [ ] `CONTRIBUTING.md` (tests, style, releases)
- [ ] `docs/architecture.md` (module map and flows)

**Done when:** a new developer can run end‑to‑end from README.

---

## 19. Final E2E

- [x] End‑to‑end smoke test
  - [x] init → gen‑warehouse(neobank) → gen‑comms(neobank with mock LLM) → run‑report → validate --strict
- [ ] Golden snapshots for `data_health.json`, `themes.json` structure, and figure hashes
- [ ] Tag `v0.1.0`

**Done when:** e2e test green; artifacts produced; tag pushed.

---

## Command snippets (for manual checks)

```bash
poetry run dnr init
poetry run dnr gen-warehouse --archetype neobank --out data/neobank
poetry run dnr gen-comms --archetype neobank --out comms/neobank
poetry run dnr run-report --warehouse data/neobank --comms comms/neobank --out reports/neobank --api-cap-usd 1.0
poetry run dnr validate --warehouse data/neobank --comms comms/neobank --out reports/neobank/qc --strict
poetry run dnr quickstart --seed 42
