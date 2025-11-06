# Data Needs Reporter

Command-line toolbox that generates synthetic warehouse activity, communications, and exec-ready reports for data-needs prioritisation.

## Installation

Install from PyPI:

```bash
pip install data-needs-reporter
dnr --help
```

Run ad-hoc without polluting your environment:

```bash
pipx run data-needs-reporter -- --version
```

For local development work:

```bash
poetry install
poetry run dnr --help
```

## Quickstart

Generate sample data, reports, and budgets for both archetypes:

```bash
poetry run dnr quickstart
```

Use `--no-llm` to skip LLM calls and `--fast` to run with evaluation-scale volumes.

Artifacts land under `reports/` with an index at `reports/index.md` linking to each archetype.

## Performance Targets

Local smoke budgets (Apple M2, Poetry install, `polars` available):

- `dnr gen-warehouse --archetype neobank --out data/neobank --dry-run` ≤ 3.0s
- `dnr gen-comms --archetype neobank --out comms/neobank` ≤ 3.5s
- `dnr run-report --warehouse data/neobank --comms comms/neobank --out reports/neobank` ≤ 2.5s
- `dnr validate --warehouse data/neobank --comms comms/neobank --out reports/neobank/qc --strict` ≤ 2.0s

## Reproducibility & Seeds

Default seeds are baked into `configs/default.yaml` so every command is deterministic when you reuse that config and cache:

- `warehouse.seed` drives synthetic warehouse generation.
- `comms.seed` powers the communications sampler and mock LLM responses.
- `.cache/llm/` stores responses keyed by prompt+model, eliminating reruns when seeds and inputs match.

To reproduce a run, make a copy of the config, adjust the seeds, and pass it to each command:

```bash
cp configs/default.yaml configs/run123.yaml
sed -i '' 's/seed: 42/seed: 123/g' configs/run123.yaml   # warehouse seed
sed -i '' 's/seed: 43/seed: 456/g' configs/run123.yaml   # comms seed
dnr gen-warehouse --config configs/run123.yaml --archetype neobank --out data/run123
dnr gen-comms --config configs/run123.yaml --archetype neobank --out comms/run123
dnr run-report --config configs/run123.yaml --warehouse data/run123 --comms comms/run123 --out reports/run123
```

You can also override seeds via environment variables (e.g., `WAREHOUSE_SEED=99 COMMS_SEED=77`) or CLI flags (`dnr gen-warehouse --warehouse-seed 99 ...`). Any run with identical seeds, inputs, and cache contents will yield the same outputs.

## Spend Caps & Budget Files

LLM usage is bounded by a per-run cap:

- Override with `--api-cap-usd` (or `classification.api_cap_usd` in config). Default: `$1.00`.
- `budget.mode` is `sample_to_fit`, so generators trim message volumes before hitting the cap.
- `budget.safety_margin` (default `0.25`) keeps a 25% buffer; `token_budget` in `budget.json` reflects the capped allowance after the buffer.
- `budget.coverage` surfaces per-source coverage; a coverage floor of `budget.coverage_floor_pct` (default 20%) drives the `behavior` and `met_floor` flags.

Sample excerpt from `comms/neobank/budget.json`:

```json
{
  "cap_usd": 1.0,
  "price_per_1k_tokens": 0.002,
  "safety_margin": 0.25,
  "token_budget": 225000,
  "tokens_used": 180000,
  "cost_usd": 0.36,
  "stopped_due_to_cap": false,
  "coverage": {
    "slack": {
      "overall": {
        "actual": 3000,
        "target": 3000,
        "coverage_pct": 1.0,
        "met_floor": true,
        "behavior": "continue"
      }
    }
  }
}
```

If `stopped_due_to_cap` flips to true, generation halted early; re-run with a larger cap (`--api-cap-usd 2.5`) or adjust seeds/config to produce smaller batches.

## Validation & Label Evaluation

Quality checks and label review tooling live in `dnr validate` and `dnr eval-labels`:

- `dnr validate --warehouse data/neobank --comms comms/neobank --out reports/neobank/qc --strict`
  - Writes `reports/neobank/qc/qc_summary.json` (pass/fail, exit code) and `qc_checks.csv` (per-gate detail).
  - Gates cover schema, volume, quality, seasonality, taxonomy, monetization, trajectory, comms coverage, theme mix, event correlation, reproducibility, and spend caps. `--strict` flips the process to non-zero exit when any gate fails.
- `dnr eval-labels --pred runs/latest/preds --labels meta --out reports/eval/latest`
  - Expects predictions under `runs/latest/preds` and reference labels in `meta/`.
  - Emits precision/recall summary at `reports/eval/latest/summary.json` and per-source breakdowns in CSV/JSON.

Use these commands after generation to gate releases or inspect model drift; re-run with the same seeds/config to compare outputs across commits.

## Core Commands

```bash
poetry run dnr init                       # scaffold directories and default config
poetry run dnr gen-warehouse --archetype neobank --dry-run
poetry run dnr gen-comms --archetype neobank
poetry run dnr run-report --warehouse data/neobank --comms comms/neobank --out reports/neobank
poetry run dnr validate --warehouse data/neobank --comms comms/neobank --out reports/neobank/qc --strict
poetry run dnr eval-labels --pred runs/latest/preds --labels meta --out reports/eval/latest
```

Global options are supplied before the subcommand, for example `dnr --config myconfig.yaml run-report ...`.

## Configuration Precedence

1. Built-in defaults (`configs/default.yaml`)
2. `--config` file (JSON/YAML)
3. Environment variables
4. CLI flags

Paths in the config are relative to the working directory by default (`data/`, `comms/`, `reports/`, `meta/`).

## Troubleshooting

- **Polars missing**: install optional dependency `polars` (`poetry add polars`) to enable Parquet generation.
- **Quickstart fails on LLM**: rerun with `--no-llm` to create report scaffolds without classification.
- **Validation fails under `--strict`**: inspect `qc_summary.json` and `qc_checks.csv` under the provided output directory for failing gates.

## Development setup

Install the git hooks once:

```bash
pre-commit install
```

Run the linters/formatters manually:

```bash
pre-commit run --all-files
```

Common make targets are available via Poetry:

```bash
make test   # pytest
make lint   # ruff + isort --check-only
make fmt    # black + isort
make e2e    # quick quickstart flow
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
details about local tooling, tests, style expectations, and the release process.
