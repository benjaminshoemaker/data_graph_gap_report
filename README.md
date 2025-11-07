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

## Usage

Global options apply to every command:

```
Usage: dnr [OPTIONS] COMMAND [ARGS]...

Options:
  --version, -V            Show the application version and exit.
  --config PATH            Path to an alternate YAML configuration file.
  --api-cap-usd FLOAT      Override the LLM API spend cap in USD.
  --window INTEGER         Override report window size in days.
  --no-llm                 Disable LLM-powered classification for this run.
  --help                   Show this message and exit.
```

### `dnr init`

Scaffold the default directories and config file:

```
Usage: dnr init
```

### `dnr gen-warehouse`

```
Usage: dnr gen-warehouse [OPTIONS]

Options:
  --archetype TEXT  Warehouse archetype to generate (neobank or marketplace).  [required]
  --out PATH        Output directory for generated warehouse files.            [required]
  --dry-run         Write empty schema tables only.
  --help            Show this message and exit.
```

### `dnr gen-comms`

```
Usage: dnr gen-comms [OPTIONS]

Options:
  --archetype TEXT  Archetype to generate communications for (neobank or marketplace).  [required]
  --out PATH        Output directory for generated communications.                      [required]
  --help            Show this message and exit.
```

### `dnr run-report`

```
Usage: dnr run-report [OPTIONS]

Options:
  --warehouse PATH  Path to warehouse data.      [required]
  --comms PATH      Path to communications data. [required]
  --out PATH        Output directory for reports. [required]
  --strict          Exit with non-zero status if coverage or budget issues are detected.
  --help            Show this message and exit.
```

### `dnr validate`

```
Usage: dnr validate [OPTIONS]

Options:
  --warehouse PATH  [required]
  --comms PATH      [required]
  --out PATH        [required]
  --strict          Fail the process when any gate fails.
  --help            Show this message and exit.
```

### `dnr eval-labels`

```
Usage: dnr eval-labels [OPTIONS]

Options:
  --pred PATH    Directory with prediction parquet files.     [required]
  --labels PATH  Directory containing oracle label parquet files. [required]
  --out PATH     Evaluation report output directory.          [required]
  --strict       Exit with non-zero code when any evaluation gate fails.
  --help         Show this message and exit.
```

### `dnr quickstart`

```
Usage: dnr quickstart [OPTIONS]

Options:
  --fast     Run with reduced volumes.
  --no-llm   Skip all LLM calls.
  --help     Show this message and exit.
```

## Performance Targets

Local smoke budgets (Apple M2, Poetry install, `polars` available) use the quickstart-scale config that `scripts/bench.py` writes before running each command with `--no-llm`:

| Command | Target (s) |
| --- | --- |
| `dnr gen-warehouse --archetype neobank --out <tmp>/warehouse` | ≤ 3.0 |
| `dnr gen-comms --archetype neobank --out <tmp>/comms` | ≤ 3.5 |
| `dnr run-report --warehouse <tmp>/warehouse --comms <tmp>/comms --out <tmp>/reports` | ≤ 2.5 |
| `dnr validate --warehouse <tmp>/warehouse --comms <tmp>/comms --out <tmp>/reports/qc --strict` | ≤ 2.0 |

Run the smoke bench anytime with:

```bash
poetry run python scripts/bench.py --report-format table
```

The script prints elapsed seconds and peak RSS for each command, writes JSON when `--report-format json` is passed, and the CI job publishes the latest numbers in the workflow summary. Use it locally with `--fail-on-budget` to gate pushes before opening a PR.

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
