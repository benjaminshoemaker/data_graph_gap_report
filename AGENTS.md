# Repository Guidelines

## Project Structure & Module Organization
- Runtime code lives under `src/data_needs_reporter/`; subpackages map directly to the spec: `config/` for Pydantic models, `generate/` for warehouse+comms synth, `report/` for metrics, scoring, LLM flows, `utils/` for I/O, DuckDB, logging, and cost guards.
- Default configuration sits in `configs/default.yaml`; generated artifacts land in `data/`, `comms/`, `reports/`, and cached LLM responses live under `.cache/llm/`.
- Keep tests mirrored to modules (`tests/test_generators.py`, `tests/test_cli.py`, `tests/test_end_to_end.py`) so every new feature ships with coverage in the matching suite.

## Build, Test, and Development Commands
- Install the toolchain with `poetry install`; use `poetry run dnr --help` to confirm Typer wiring.
- Fast smoke runs: `poetry run dnr quickstart --no-llm --fast`; targeted flows: `poetry run dnr gen-warehouse --archetype neobank --dry-run` and `poetry run dnr gen-comms --archetype neobank --out comms/neobank`.
- Run quality gates locally via `pre-commit run --all-files` and the full suite with `poetry run pytest`; execute long-path checks using `poetry run pytest tests/test_end_to_end.py`.

## Coding Style & Naming Conventions
- Follow Black-formatted, 4-space indentation and snake_case functions/modules; Pydantic models stay PascalCase and constants use UPPER_SNAKE.
- Pre-commit enforces `black`, `ruff`, and `isort`; install hooks once with `pre-commit install`.
- Thread seeds from config helpers (e.g., `config.warehouse.seed`) into RNG utilities to keep outputs reproducible per spec.

## Testing Guidelines
- Use `pytest` naming (`test_*`) with behavior-focused assertions; stash shared fixtures in `tests/conftest.py` and gate optional deps with `pytest.importorskip("polars")`.
- When adjusting generators or defect injectors, extend `tests/test_generators.py` and `tests/test_metrics.py`; CLI flows belong in `tests/test_cli.py` or `tests/test_end_to_end.py`.
- Record example command output paths (such as `reports/neobank/exec_summary.json`) when debugging failures.

## Commit & Pull Request Guidelines
- Prefer single-purpose, imperative commits (e.g., `Add sample-to-fit allocator guard`) and run `pre-commit run --all-files && poetry run pytest` before pushing.
- PRs should outline affected CLI commands, config knobs, and include testing notes; link any spec paragraphs or `todo.md` items that motivated the change.

## Progress Tracking
- After writing any code, open `todo.md`, check off all completed items, and call out remaining follow-ups so progress stays traceable between sessions.

## Configuration & Secrets
- Load overrides via `--config`, env vars, or flags in that precedence; keep credentials in `.env` (see `.env.example`) and export `OPENAI_API_KEY` only when running LLM-enabled commands.
