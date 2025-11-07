# Changelog

## 0.1.2 — 2025-11-07

- Shipped the initial MVP of Data Needs Reporter: deterministic warehouse/comms generators, LLM-backed classification with spend caps, entity extraction, and exec-ready reporting.
- Hardened the toolchain with end-to-end pytest coverage, budget enforcement tests, and validation gates spanning schema, volume, quality, taxonomy, monetization, communications, and reproducibility.
- Added `dnr quickstart` to run the full synthetic workflow (init → gen-warehouse → gen-comms → run-report → validate) in one command for both archetypes, with `--fast` and `--no-llm` modes for local smoke runs.
- Documented CLI usage, configuration precedence, and troubleshooting so contributors can reproduce runs, update goldens, and keep CI green.
