# Repository Guidelines

## Project Structure & Module Organization
- CLI source lives in `src/data_needs_reporter/` with subpackages for config, utilities, generators, reporting, and evaluation.
- Synthetic assets persist under `data/`, `comms/`, and `reports/`; config defaults stay in `configs/default.yaml`.
- Tests mirror the runtime layout in `tests/` (e.g., `tests/test_generators.py`, `tests/test_cli.py`) and should track new modules 1:1.

## Build, Test, and Development Commands
- `poetry install` sets up the Python 3.11 environment with Typer, DuckDB, Polars, and test tooling.
- `poetry run dnr quickstart --no-llm` generates a fast, deterministic smoke run without LLM calls.
- `poetry run dnr gen-warehouse --archetype neobank --out data/neobank` and `poetry run dnr gen-comms --archetype neobank --out comms/neobank` materialize archetype data for manual inspection.
- `poetry run pytest` runs the full test suite; add `-k name` or `-m marker` for focused runs.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, snake_case for functions and modules, PascalCase for classes, and UPPER_SNAKE for constants.
- Prefer type hints and dataclasses/Pydantic models for structured data.
- Keep CLI command names short (`dnr run-report`), and align module names with their domain (`generate/warehouse.py`, `report/metrics.py`).

## Testing Guidelines
- Tests use `pytest`; name files `test_*.py` and functions `test_*` describing behavior (e.g., `test_budget_respects_cap`).
- Add fixtures for synthetic data in `tests/conftest.py` when shared setup is needed.
- Aim to cover defect injection, metrics accuracy, and CLI flows (`quickstart --no-llm`) before merging.

## Commit & Pull Request Guidelines
- Write imperative commit messages summarizing intent (e.g., `Add sampling guard for LLM budget`).
- Squash fixup noise before review; keep commits scoped to a single concern.
- Pull requests should include a concise summary, testing notes (`poetry run pytest`), and any configuration/environment caveats (e.g., required `OPENAI_API_KEY`).
- Link to relevant issues or todo items when applicable, and attach sample output paths (such as `reports/neobank/data_health.json`) to contextualize changes.

## Progress Tracking
- After completing any code change, review `todo.md` and mark tasks that are now done so the team can track session-to-session progress.
