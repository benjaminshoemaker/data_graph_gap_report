# Architecture Overview

Data Needs Reporter is a CLI application that synthesizes warehouses, communications, metrics,
and evaluation artifacts. This document maps the major modules, describes data flow between them,
and highlights invariants enforced across the pipeline.

```
┌─────────────┐    ┌────────────────────┐    ┌──────────────────────┐
│ configs/    │    │ data_needs_reporter│    │ reports/             │
│ default.yaml┼───▶│ generate/          ├───▶│  data_health.json     │
└─────────────┘    │  report/           │    │  themes.json          │
                   │  eval/             │    │  exec_summary.*       │
                   └────────────────────┘    └──────────────────────┘
                            ▲                         ▲
                            │                         │
                     communications/            qc/ + eval outputs
```

## Module Map

- `data_needs_reporter/cli.py`
  - Typer entry point; wires commands (`init`, `gen-warehouse`, `gen-comms`, `run-report`, `validate`, `eval-labels`, `quickstart`).
  - Loads configuration precedence (default file → overrides → env → CLI flags).
  - Handles logging context, run IDs, and atomic write helpers.

- `data_needs_reporter/config.py`
  - Pydantic models for typed configuration.
  - Environment/CLI override helpers.
  - Notable invariants: config must parse to `AppConfig`, default path resolves for packaged installs.

- `data_needs_reporter/generate/`
  - `warehouse.py`: Synthetic fact/dim tables, manifests, schema (Polars-based). Writes to `data/<archetype>/`.
    - Invariants: deterministic by seed, writes `.dry_run` marker when invoked with `--dry-run`.
  - `comms.py`: Slack/Email/NLQ corpora, coverage/quotas, budget accounting via `CostGuard`. Outputs to `comms/<archetype>/`.
    - Invariants: stops when `CostGuard` indicates cap exhaustion, enforces coverage floors.
  - `defects.py`: Injects data quality issues for evaluation scenarios.

- `data_needs_reporter/report/`
  - `metrics.py`: Data health calculations, SLO validation, reproducibility, coverage checks.
  - `scoring.py`: Combines demand/severity/revenue into top actions.
  - `run.py`: Entity extraction and scoring orchestrator; now also exposes `assess_budget_health`.
  - `llm.py`: LLM client with cache, repair attempts, and parse error handling.
  - `prefilter.py`: Message filtering heuristics before LLM calls.
  - `plots.py`: Matplotlib outputs stored under `reports/<arch>/figures/`.

- `data_needs_reporter/eval/`
  - `labels_eval.py`: F1 and coverage gates for label predictions. Cached parquet loads from `meta/`.

- `data_needs_reporter/utils/`
  - `io.py`: Atomic JSON/CSV/Parquet writes; *all* writes route through here to avoid partial files.
  - `cost_guard.py`: Token accounting with safety margin; exposes budget snapshots.
  - `hashing.py`: Manifest creation for integrity checks.
  - `rand.py`, `duck.py`, `logging.py`: RNG helpers, DuckDB utilities, and structured logging context.

- `tests/`
  - Mirrors module layout: CLI, generators, metrics, IO, performance, end-to-end.
  - Performance test is opt-in locally and skipped in CI to avoid flakiness.

## Data Flow

```
configs/default.yaml
        │
        ▼
 data_needs_reporter.config.AppConfig
        │
        ├─▶ generate.warehouse   ─┐
        │                         ├─▶ run-report metrics ─▶ reports/<arch>/...
        └─▶ generate.comms ───────┘

comms/<arch>/budget.json
        │
        └─▶ report.run.assess_budget_health ─▶ warnings / strict exit

comms/<arch>/slack_messages.parquet
        │
        ├─▶ report.prefilter + report.classify ─▶ evaluations
        └─▶ eval/labels_eval (predictions vs meta/)
```

### CLI Command Flow

1. **init**
   - Copies default config, creates artifact directories, ensures cache.
2. **gen-warehouse**
   - Uses `generate.warehouse` to synthesize parquet tables, schema manifest, hash manifest.
   - Dry-run writes empty tables + `.dry_run` marker.
3. **gen-comms**
   - Uses `generate.comms` with `CostGuard` to sample Slack/Email/NLQ corpora; writes budget (`budget.json`), coverage, quota metadata.
4. **run-report**
   - Loads metrics, scoring, entity extraction, plots; writes report bundle under `reports/<arch>/`.
   - Emits warnings when budget coverage is low or caps triggered; `--strict` upgrades warnings to exit code 1.
5. **validate**
   - Validates warehouse + comms via metrics gates; writes `qc_summary.json` and `qc_checks.csv`.
6. **eval-labels**
   - Compares predictions vs `meta/` ground truth; produces per-source precision/recall and gated pass/fail.

### Key Invariants

- **Determinism:** All generators accept seeds from config/env/CLI; outputs must be reproducible under identical seeds.
- **Atomic I/O:** JSON/CSV/Parquet writes must use `write_*_atomic`. No partial files should survive failures.
- **Budget Enforcement:** `CostGuard` must stop before exceeding `cap_usd` minus safety margin; `budget.json` documents coverage and spend outcomes.
- **Validation Strictness:** `dnr validate --strict` only exits non-zero when checks fail; default mode warns but allows downstream steps.
- **Parse Error Handling:** LLM classification records parse errors with `parse_error=True` and halts after the configured consecutive limit.
- **Performance Budgets:** `tests/test_perf.py` bounds CLI runtimes to catch regressions.

## Diagrams

```
+-----------------+          +-------------------+          +-----------------+
|  gen-warehouse  |          |    gen-comms      |          |    run-report   |
|-----------------|          |-------------------|          |-----------------|
| data/<arch>     |   +----▶ | comms/<arch>      |   +----▶ | reports/<arch>  |
| schema.json     |   |      | budget.json       |   |      | figures/*.png   |
| manifest.json   |   |      | slack/email/nlq   |   |      | exec_summary.*  |
+-----------------+   |      +-------------------+   |      +-----------------+
                      |                            |
                      |                            ▼
                      |                  validate (qc_summary/qc_checks)
                      ▼
            eval-labels (reports/eval/<run>/)
```

## Release Notes Checklist

- Update version via `make bump VERSION=X.Y.Z`.
- Ensure README performance targets and spend-cap examples still match behaviour.
- Regenerate wheels (`poetry build`) if packaging changes.
- Capture architectural changes here when modules move or invariants change.

---

Questions or improvements? Open an issue or update this document in the same PR as your code change.
