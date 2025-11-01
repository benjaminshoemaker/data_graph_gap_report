1) Step‑by‑step build blueprint
1.1 Modules and responsibilities

CLI (cli.py): command routing, config resolution, logging setup.

Config (config.py): Pydantic models, load/merge precedence, validation.

Utils

io.py: Parquet/JSON/CSV read‑write, path helpers, hashing.

duck.py: DuckDB session helper, schema DDL, safe query.

rand.py: seeded RNG wrappers, distributions.

logging.py: structured logs, run IDs.

cost_guard.py: token estimate, cap tracking, cache dir, budget JSON.

Generate

warehouse.py: archetype data generation per schema, taxonomy, behavior.

defects.py: inject nulls, FK fails, duplicates, lag, spikes, schema gaps.

comms.py: Slack/Email/NLQ synthesis using LLM; non‑PII IDs; role mix; buckets.

Report

metrics.py: table metrics, spike detection, SLO evaluation.

classify.py: prefilter, sample‑to‑fit budgeting, LLM client, JSON parsing guards.

run.py: orchestrate report window, demand weights, entity extraction, confidence, diversity top‑3, emit JSON+MD.

scoring.py: revenue risk, demand, severity, recency decay, tie breakers.

plots.py: lag, key‑null, orphan, dup bars, theme demand.

Eval

labels_eval.py: join predictions vs oracle, macro‑F1, per‑class metrics, CIs.

1.2 Data models and formats

Warehouse: Parquet tables per schema; DuckDB attached for SQL and metrics.

Comms: slack_messages.parquet, email_messages.parquet, comms_users.parquet, nlq.parquet.

Reports: exec_summary.json|md, data_health.json|csv, themes.json|md, figures/*.png, budget.json.

Meta: labels_*.parquet.

1.3 Runtime flows

gen‑warehouse: synthesize base → inject defects → write Parquet → optional DuckDB DB.

gen‑comms: synthesize Slack/Email/NLQ text per caps → write Parquet.

run‑report:

Load config and comms.

Prefilter per rules.

Preflight token p90 and compute quotas with safety margin and coverage floors.

Classify themes (+relevance for Slack/Email).

Entity extract on positive Slack/Email and all NLQ.

Load warehouse, compute metrics.

Compute demand shares with post‑strat weights and source weight caps.

Compute revenue risk, severity; rank; enforce theme diversity; compute confidence; gate.

Emit files and figures.

validate: schema, volume, quality targets, seasonality, taxonomy, monetization, trajectory, comms mix, theme mix, NLQ caps, event correlation, reproducibility, spend caps.

eval‑labels: supervised metrics and gates.

1.4 Error handling

Config: fail fast if invalid; show offending field.

IO: atomic writes to temp then rename, create dirs, checksum for golden tests.

LLM: timeout 15s, 2 retries, JSON repair once, consecutive‑parse error hard‑stop per source.

Budget: never exceed caps; sample‑to‑fit allocation; emit budget.json.

Exit codes: non‑zero on internal errors; --strict escalates coverage/budget breaches.

1.5 Testing strategy

TDD per module.

Golden snapshots for report shapes and plot hashes.

Deterministic seeds.

Mock LLM with fixed JSON to test pipelines and caps.

2) Iterative chunks (milestones)

Scaffold + Config + CLI skeleton

IO and DuckDB utilities

Warehouse generator (baseline schemas)

Defect injection

Comms generator (LLM stubs + caps + caching)

Prefilter + sampling + budget guard

Metrics engine + SLO evaluation + plots

Classification pipeline (LLM client, JSON guard, parse errors)

Entity extraction

Scoring + prioritization + confidence + diversity

Report assembly to JSON/MD + figures

Validation suite

Oracle eval tool

Quickstart + packaging

Performance hardening

3) Finer breakdown to implementable steps

For each milestone, steps have acceptance tests.

M1 Scaffold + Config + CLI

Repo layout and pyproject.toml.

Typer CLI with dnr --help.

Config models and precedence; load YAML+env+flags; write configs/default.yaml.

dnr init: create folders and default config.

M2 IO and DuckDB utils

io.py: write/read Parquet/JSON/CSV; ensure atomic write; test round‑trip.

duck.py: open DB, attach Parquet, execute safe SQL; test simple query.

logging.py: run IDs, structured logs; test context manager.

M3 Warehouse generator

Schema constants; empty Parquet writers with types.

Neobank base synth: dims, basic facts with distributions.

Marketplace base synth.

Attach DuckDB and verify basic KPI queries pass.

M4 Defects

Inject key nulls, FK fails, duplicates.

Lag and spikes with loaded_at; schema gap windows + backfill.

Calibration to hit target rates; emit summary.

M5 Comms generator

Comms schemas; comms_users.parquet with role mix.

LLM client interface + on‑disk cache; mock provider in tests.

Slack/Email/NLQ generation respecting caps, buckets, roles, windows.

M6 Prefilter + sampling + budget

Prefilter score; reasons logged; daily/category top‑k fallback.

Preflight token estimate; sample‑to‑fit allocator with safety margin.

Coverage floors and behavior C; write budget.json.

M7 Metrics + plots

Compute per‑table metrics; detect spikes; SLO comparison.

Plot functions write PNGs; deterministic hash in tests.

M8 Classification

Role‑aware thread builder and token cap.

LLM classify Slack/Email threads and NLQ; parse+retry+guard.

Store predictions Parquet with confidences.

M9 Entity extraction

LLM extract tables/columns from positive Slack/Email and all NLQ; cache; write entities.parquet.

M10 Scoring + prioritization

Revenue normalization (3‑month median), demand weights and caps, severity, recency.

Diversity top‑3, confidence compute, gating.

M11 Reports

Assemble data_health.json|csv, themes.json|md, exec_summary.json|md.

Index figures and write reports/.../figures.

M12 Validation

Implement dnr validate checks and --strict.

Seeded end‑to‑end test passes Typical profile.

M13 Eval tool

eval-labels metrics, confusion, CIs, gates.

M14 Quickstart + packaging

quickstart both archetypes; --no-llm mode.

PyPI packaging metadata; entry‑point script; README quickstart.

M15 Performance

Benchmarks and memory limits; regression guard.

4) Final micro‑steps (right‑sized)

Each task results in running tests green and integrated code.

Create project skeleton, Typer app prints version.

Config models with precedence; env and CLI override tests.

Init command writes folders and default YAML; idempotency.

IO atomic JSON/Parquet write; temp rename; round‑trip tests.

DuckDB helper connect, SQL run, teardown; simple select test.

Logging run ID context; file and console handlers.

Schema definitions and empty tables written; validate dtypes.

Neobank dims synthesis; row counts and key uniqueness tests.

Neobank facts generation basic; time bounds and channels distribution tests.

Marketplace dims; uniqueness and counts.

Marketplace facts; category and seller Zipf sanity tests.

Inject key nulls; measure ~2%±0.5.

Inject FK fails; produce ~5%±1, with mix split.

Inject dup keys; hit 0.7%±0.2; earliest loaded_at wins in KPI tests.

Inject lag with p95 in 120–240 min; spike days flagged.

Schema gaps and backfill windows; warehouse write paused then backfilled.

Comms schemas written; comms_users role distribution exact.

LLM client interface and cache adapter; mock returns; JSON parse guard.

Slack generator: threads 3k, tokens/msg ≤80, IDs only; forbidden domains absent.

Email generator: 800, ≤180 tokens; IDs only.

NLQ generator: 1k, ≤30 tokens; non‑execution; parse‑only optional.

Prefilter scoring; keep ≥0.35 or top‑k fallback; reasons present.

Preflight token p90; allocator computes quotas under cap with 25% margin.

Budget report written; coverage floors enforced; behavior C.

Metrics engine computes key‑null, FK, dup, p95 lag; spike detection; SLO flags.

Plots produced; hashes stable under seed.

Role‑aware thread packer; include root and execs; ≤900 tokens and ≤20 msgs.

Classification end‑to‑end with mock LLM; predictions saved.

Entity extraction with mock; entities.parquet saved.

Revenue impact, demand weights with caps, severity, recency; unit tests per formula.

Diversity selection and confidence; gating hides low confidence.

Assemble JSON and MD outputs with schemas; schema validation.

Validation command gates; --strict fail paths tested.

Eval metrics with CIs; macro‑F1 gates.

Quickstart e2e, --no-llm path outputs data_health and “themes skipped”.

Packaging and CLI entry point; pipx run smoke test.

Benchmark on sample laptop; enforce budgets; memory under 4 GB.

5) TDD code‑generation prompts

Each prompt is self‑contained and builds on prior steps. Copy to your code‑gen LLM one at a time.

Prompt 1 — Project scaffold and CLI stub
You are implementing step 1.

Goal: scaffold a Python 3.11 project named `data-needs-reporter` with Typer CLI stub `dnr`.

Requirements:
- Create the repo layout shown in the spec.
- Add `pyproject.toml` with Poetry config, dependencies: typer[all], pydantic, duckdb, pyarrow, polars, matplotlib.
- Create `src/data_needs_reporter/cli.py` with Typer app exposing `--version` and root command that prints help.
- Add `src/data_needs_reporter/__init__.py` with `__version__ = "0.1.0"`.
- Configure console script entry point: `dnr`.
- Add `.gitignore` and `.env.example`.
- Add minimal `README.md` with install and `dnr --help`.

Tests (write first):
- `tests/test_cli.py`: verifies `dnr --version` prints `0.1.0` and exit code 0; `dnr` prints help.

Deliverables:
- All files created and tests passing via `pytest`.
- No other functionality yet.

Prompt 2 — Config models and precedence
You are implementing step 2.

Goal: Pydantic config loader with precedence: defaults → YAML → env → CLI.

Requirements:
- Create `src/data_needs_reporter/config.py` with Pydantic models mirroring the spec’s `Config` excerpt.
- Implement `load_config(default_path, override_yaml_path_or_none, env, cli_overrides)` returning a validated config instance.
- CLI flags to expose: `--config`, `--api-cap-usd`, `--window`, `--no-llm` (bool), wires into Typer group but only parsed/stored.
- Add `configs/default.yaml` with spec defaults.
- Respect precedence in `load_config`.

Tests:
- `tests/test_config.py`: 
  - default load equals YAML values,
  - env var `LLM_API_CAP_USD` overrides YAML,
  - CLI override `--api-cap-usd 0.5` wins over env,
  - invalid types raise ValidationError with field name.

Deliverables: passing tests.

Prompt 3 — init command
You are implementing step 3.

Goal: `dnr init` scaffolds folders and default YAML, idempotent.

Requirements:
- Implement `init` in `cli.py`.
- Create dirs from config: `data/`, `comms/`, `reports/`, `meta/`, `.cache/llm/`.
- If files exist, do nothing; print created paths.
- Copy default YAML if missing.

Tests:
- `tests/test_cli_init.py`: 
  - running `dnr init` creates dirs,
  - second run makes no changes,
  - default.yaml present.

Deliverables: passing tests.

Prompt 4 — IO utilities
You are implementing step 4.

Goal: atomic read/write helpers.

Requirements:
- `src/data_needs_reporter/utils/io.py`:
  - `write_json_atomic(path, obj)`,
  - `read_json(path)`,
  - `write_parquet_atomic(path, dataframe)`,
  - `read_parquet(path)`,
  - `write_csv_atomic(path, dataframe)`.
- Use tmp file + rename pattern.
- Ensure parent dirs created.

Tests:
- `tests/test_io.py`: round-trip JSON, Parquet, CSV; verify atomic rename and directory creation.

Deliverables: passing tests.

Prompt 5 — DuckDB utility
You are implementing step 5.

Goal: DuckDB helper.

Requirements:
- `src/data_needs_reporter/utils/duck.py`:
  - `open_db(db_path_or_none)`: returns connection; in-memory if None.
  - `attach_parquet_dir(conn, name, dir_path)`: create views for all `.parquet` in dir.
  - `safe_query(conn, sql, params=None)`: basic guard + returns Polars DataFrame.
- Minimal SQL injection guard: block semicolons and `COPY/ATTACH/INSTALL`.

Tests:
- `tests/test_duck.py`: in-memory db, create small temp Parquet, attach, run `SELECT COUNT(*)`.

Deliverables: passing tests.

Prompt 6 — Logging and run IDs
You are implementing step 6.

Goal: structured logging and run IDs.

Requirements:
- `src/data_needs_reporter/utils/logging.py`:
  - `init_logger(name, level="INFO")` returns logger,
  - `with run_context(logger) -> (run_id, context manager)` adds run_id to records.
- Hook CLI to init logging and create a run_id per command.

Tests:
- `tests/test_logging.py`: logs contain run_id and message; context exits cleanly.

Deliverables: passing tests.

Prompt 7 — Warehouse schema writers
You are implementing step 7.

Goal: schema constants and empty writers.

Requirements:
- `src/data_needs_reporter/generate/warehouse.py`:
  - constants for neobank and marketplace schemas (columns + dtypes),
  - function `write_empty_warehouse(archetype, out_dir)`.
- Wire `dnr gen-warehouse` to call `write_empty_warehouse` when `--dry-run`.

Tests:
- `tests/test_generators.py`: empty Parquet files exist with expected columns and dtypes.

Deliverables: passing tests.

Prompt 8 — Neobank dim synthesis
You are implementing step 8.

Goal: generate neobank dimension tables with seeded RNG.

Requirements:
- `generate_neobank_dims(cfg, out_dir, seed)` returns counts and writes:
  - `dim_customer`, `dim_account`, `dim_card`, `dim_merchant`, `dim_plan`.
- Distributions per spec; timestamps UTC.

Tests:
- `tests/test_generators.py`: 
  - unique keys, 
  - referential integrity between account→customer and card→account,
  - merchant sectors coverage >= 8.

Deliverables: passing tests.

Prompt 9 — Neobank facts baseline
You are implementing step 9.

Goal: generate `fact_card_transaction` and `fact_subscription_invoice` baseline (no defects yet).

Requirements:
- Behavior per spec: non-homogeneous Poisson, lognormal amounts, channels mix, auth_result, subs attach 8%, churn 1.8%/mo.
- Add `loaded_at = event_time` initially.

Tests:
- `tests/test_generators.py`:
  - event_time within 18 months,
  - captured rate ~96.5%±1,
  - subs attach in 6–10%.

Deliverables: passing tests.

Prompt 10 — Marketplace dims and facts baseline
You are implementing step 10.

Goal: marketplace dims and facts baseline.

Requirements:
- Write `dim_buyer`, `dim_seller`, `dim_category`, `dim_listing`.
- Facts: `fact_order`, `fact_order_item`, `fact_payment`, `snapshot_listing_daily`.
- Single-seller orders; category and seller Zipf.

Tests:
- `tests/test_generators.py`:
  - items/order distribution sanity,
  - weekend factor 1.15–1.45,
  - take rate ~12%±1.

Deliverables: passing tests.

Prompt 11 — Defect injection
You are implementing step 11.

Goal: apply Typical profile defects.

Requirements:
- `defects.py` with functions:
  - `inject_key_nulls(df, cols, rate_pct)`,
  - `inject_fk_failures(fact_df, dim_df_map, rule, rate_pct, mix)`,
  - `inject_duplicates(df, key_col, rate_pct)`,
  - `apply_ingest_lag(df, event_col, loaded_at_col)`,
  - `inject_null_spikes(df, key_col, schedule)`,
  - `inject_schema_gap(...)`.
- Expose from `gen-warehouse` as a pipeline with calibration loop to hit target ranges.

Tests:
- `tests/test_generators.py`:
  - measured rates within ± tolerances,
  - p95 lag within 120–240 min,
  - spike days detected later by metrics test fixture.

Deliverables: passing tests.

Prompt 12 — Comms schemas and users
You are implementing step 12.

Goal: write comms schemas and users.

Requirements:
- `generate/comms.py`:
  - create `slack_messages.parquet`, `email_messages.parquet`, `comms_users.parquet`, `nlq.parquet` with no rows yet.
  - users table roles per mix.

Tests:
- `tests/test_generators.py`: schema fields present; role distribution sums to 100%.

Deliverables: passing tests.

Prompt 13 — LLM client + cache + mocks
You are implementing step 13.

Goal: LLM client interface with on-disk cache and mock provider.

Requirements:
- `llm.py` in `report/`:
  - `LLMClient(provider, model, api_key_env, timeout_s, max_output_tokens)`,
  - `.json_complete(prompt, temperature)` returns dict; enforces JSON and caps.
  - cache layer under `.cache/llm/{hash}.json`.
- Mock client for tests returning fixed JSON.

Tests:
- `tests/test_classifier.py`: cache hit on second call; parse error triggers one repair attempt; then raises.

Deliverables: passing tests.

Prompt 14 — Comms LLM generation with caps
You are implementing step 14.

Goal: generate Slack/Email/NLQ content via LLM within $1 cap per archetype.

Requirements:
- Implement generation functions that:
  - obey token caps per item and totals,
  - write Parquet rows,
  - respect IDs only, roles, buckets, windows.
- Wire `dnr gen-comms` to call them.
- Add `cost_guard.py` to track estimated tokens and dollars; write `budget.json`.

Tests:
- `tests/test_classifier.py`:
  - stop before exceeding cap,
  - volumes equal requested counts within ±5% for Slack/Email/NLQ under cap.

Deliverables: passing tests.

Prompt 15 — Prefilter and reasons
You are implementing step 15.

Goal: prefilter Slack/Email with scores and reasons.

Requirements:
- Implement rules: bucket allowlist, keyword boosts, structural signals, role weights, event correlation, thread rules; threshold 0.35, else top‑30/day/channel fallback.
- Persist `prefilter_score` and reasons.

Tests:
- `tests/test_classifier.py`: high-signal samples pass threshold; near-match negatives demoted.

Deliverables: passing tests.

Prompt 16 — Sample‑to‑fit budgeter
You are implementing step 16.

Goal: preflight token p90; allocate quotas with 25% safety; coverage floors.

Requirements:
- Compute quotas per source and day×bucket; enforce min per bucket; random seed.
- Behavior C if a source misses 20% floor; weights unchanged.
- Emit `budget.json` with estimate, actual, coverage by bucket.

Tests:
- `tests/test_classifier.py`: quotas sum under cap; floors enforced; budget.json schema.

Deliverables: passing tests.

Prompt 17 — Metrics engine + plots
You are implementing step 17.

Goal: compute data-health metrics and write figures.

Requirements:
- `report/metrics.py`: implement key‑null, fk_success/orphan, dup_key, p95 lag, spike detection, SLO compare.
- `report/plots.py`: five figures per spec using matplotlib.

Tests:
- `tests/test_metrics.py`: metrics match injected rates within tolerances; spike detection works; images exist; hash stable under seed.

Deliverables: passing tests.

Prompt 18 — Role‑aware thread packing and classify
You are implementing step 18.

Goal: pack threads and classify via LLM.

Requirements:
- Pack rule: include root and all exec messages, then top‑scored newest‑first; caps 900 tokens, 20 messages.
- Call LLM with classification prompt; handle retries and parse repair; stop after 50 consecutive parse errors per source.
- Save predictions Parquet.

Tests:
- `tests/test_classifier.py`: packing respects caps; predictions schema correct; parse error handling.

Deliverables: passing tests.

Prompt 19 — Entity extraction
You are implementing step 19.

Goal: extract tables/columns via LLM on positives and all NLQ.

Requirements:
- Use approved prompt; cap spend at $0.10 per archetype; reuse cache.
- Save `entities.parquet`.

Tests:
- `tests/test_classifier.py`: entity results limited to known dictionaries; confidences in [0,1].

Deliverables: passing tests.

Prompt 20 — Scoring, diversity top‑3, confidence, gating
You are implementing step 20.

Goal: implement scoring formulas and selection.

Requirements:
- Revenue normalization 3‑month median; demand weights with volume reweight and caps [0.15, 0.60]; severity vs SLO; recency decay.
- Select top‑3 with theme diversity; compute confidence; drop items <0.55.

Tests:
- `tests/test_metrics.py`: unit tests for formulas; diversity enforced under contrived inputs.

Deliverables: passing tests.

Prompt 21 — Report assembly
You are implementing step 21.

Goal: write `data_health.json|csv`, `themes.json|md`, `exec_summary.json|md` and figures.

Requirements:
- Validate against JSON schemas embedded in tests.
- `dnr run-report` orchestrates full pipeline and writes outputs plus `budget.json`.

Tests:
- `tests/test_cli.py`: e2e run on tiny dataset produces all files; JSON schema valid.

Deliverables: passing tests.

Prompt 22 — Validation command
You are implementing step 22.

Goal: `dnr validate --strict` executes QC gates.

Requirements:
- Implement all checks; exit non‑zero on `--strict` failure; write `qc_summary.json` and `qc_checks.csv`.

Tests:
- `tests/test_cli.py`: failing gate returns non‑zero under `--strict`; passing run returns zero.

Deliverables: passing tests.

Prompt 23 — Oracle labels eval
You are implementing step 23.

Goal: `dnr eval-labels` computes macro‑F1, per‑class metrics, confusion, CIs; gates.

Requirements:
- Join by id; coverage; optional top‑k; calibration ECE if probs present.
- Output summary JSON and CSVs.

Tests:
- `tests/test_classifier.py`: synthetic labels vs deterministic predictions yield expected metrics; gates pass/fail as configured.

Deliverables: passing tests.

Prompt 24 — Quickstart, packaging, performance
You are implementing step 24.

Goal: `dnr quickstart` both archetypes; `--no-llm` path; package for PyPI.

Requirements:
- Quickstart writes `reports/index.md` linking outputs.
- `--no-llm` skips comms and marks themes skipped.
- Add build metadata and console script; verify `pipx run` style script locally in tests with `subprocess`.

Tests:
- `tests/test_cli.py`: quickstart produces index; no‑llm produces data_health only.
- `tests/test_perf.py`: timing under budgets on small test sample (use loose thresholds).

Deliverables: passing tests.

6) Final wiring prompts
Prompt 25 — End‑to‑end smoke and docs
You are implementing step 25.

Goal: end‑to‑end smoke tests and README updates.

Requirements:
- Add `tests/test_end_to_end.py` running: init → gen‑warehouse (neobank) → gen‑comms (neobank with tiny counts and mock LLM) → run‑report → validate --strict.
- Update README with quickstart, commands, config precedence, and troubleshooting.
- Ensure no orphan modules remain; CLI imports all implemented modules.

Deliverables: tests pass and documentation updated.


These prompts implement the full MVP in small, testable increments. Each step integrates into the CLI. No dead code.