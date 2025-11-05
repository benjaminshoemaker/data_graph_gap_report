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
