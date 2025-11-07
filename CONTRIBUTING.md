# Contributing Guide

Thanks for improving Data Needs Reporter! This guide covers the local tooling,
coding style, test expectations, and release process used by the project.

## Getting Started

1. Install Poetry and Python 3.11.
2. Install dependencies and dev extras:
   ```bash
   poetry install
   ```
3. Install pre-commit hooks (required for every contributor):
   ```bash
   pre-commit install
   ```
4. Verify the CLI wiring:
   ```bash
   poetry run dnr --help
   ```

Optional dependencies such as `polars` are automatically installed through the
default Poetry lockfile. If you encounter missing wheels on your platform,
install them manually inside the Poetry environment.

## Local Setup

Follow this quick routine before you start a dev session:

1. Install or refresh dependencies inside the Poetry environment:
   ```bash
   poetry install
   ```
2. Format imports and code to match CI:
   ```bash
   make fmt
   ```
3. Run the linters (`ruff`, `black`, `isort`) via:
   ```bash
   make lint
   ```
4. Execute the full test suite:
   ```bash
   make test
   ```
5. When touching snapshot expectations under `tests/goldens/`, regenerate them
   deterministically:
   ```bash
   UPDATE_GOLDENS=1 poetry run pytest tests/test_end_to_end.py
   ```

These steps mirror the CI pipeline, so keeping them green locally avoids churn
in preflight checks.

## Tooling Overview

- **Poetry** manages dependencies, virtualenvs, and packaging metadata.
- **pre-commit** runs `black`, `ruff`, and `isort` before every commit. Never
  skip these hooks; CI enforces the same checks.
- **pytest** provides unit, integration, and performance tests. The suite
  expects the default config, deterministic seeds, and optional DuckDB/Polars
  wheels.
- **Makefile** convenience targets:
  - `make lint`, `make fmt`, `make test`, `make e2e`.
- **Atomic writers** live under `src/data_needs_reporter/utils/io.py`. Always
  route JSON/CSV/Parquet writes through these helpers to avoid partial files.

## Development Workflow

1. Create a feature branch.
2. Implement changes with deterministic seeds in mind. Every generator and CLI
   command should accept overridden seeds and re-run cleanly.
3. Use the atomic write helpers whenever emitting files inside `data/`, `comms/`,
   or `reports/`.
4. Keep run-report and validation commands non-destructive: they should warn on
   coverage issues and only hard-fail in strict mode.
5. Update documentation (README, CONTRIBUTING, or `docs/` if applicable) when
   user-visible behaviour changes.

## Testing Expectations

- Run the fast suite locally:
  ```bash
  poetry run pytest
  ```
- For targeted runs:
  ```bash
  poetry run pytest tests/test_cli.py::test_run_report_generates_outputs
  poetry run pytest tests/test_perf.py         # skips automatically in CI
  ```
- Ensure pre-commit passes before pushing:
  ```bash
  pre-commit run --all-files
  ```
- When touching generators or metrics, add coverage in the mirrored test module
  (`tests/test_generators.py`, `tests/test_metrics.py`, etc.) and update any
  golden fixtures.

## Style & Conventions

- Python formatting is auto enforced by `black` (PEP 8, 4-space indentation).
- Imports follow `isort`'s Black profile; run `make fmt` after refactors.
- Lint issues are surfaced by `ruff`. Prefer addressing root causes over
  disabling rules.
- Keep functions and modules snake_case; Pydantic models stay PascalCase.
- Document complex flows with short docstrings or comments, but avoid verbose
  narration in obvious code.
- All writes to disk should be atomic (`write_json_atomic`, `write_parquet_atomic`,
  `write_csv_atomic`).

## Performance Budgets

`tests/test_perf.py` enforces local runtime budgets for the CLI commands. The
test skips automatically in CI, but contributors should run it when optimizing
or changing data volumes:

```bash
poetry run pytest tests/test_perf.py
```

If your change legitimately increases runtime, update both the test budgets and
README performance section with justification.

## Release Process

1. Ensure `poetry run pytest` and `pre-commit run --all-files` pass.
2. Update version numbers via the helper target:
   ```bash
   make bump VERSION=X.Y.Z
   ```
   Use `DRY_RUN=1` first if you want to preview the bump.
3. Regenerate distribution artifacts if needed:
   ```bash
   poetry build
   ```
4. Update README/CHANGELOG with user-facing changes and any new CLI flags.
5. Push the branch, open a PR referencing related issues, and let CI go green.
6. After approval, tag via `git tag vX.Y.Z` (done automatically by `make bump`)
   and push tags to the remote.

By following these guidelines we keep determinism, tests, and packaging healthy
for everyone. Happy shipping!
