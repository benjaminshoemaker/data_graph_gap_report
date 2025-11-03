.PHONY: test lint fmt e2e

test:
	poetry run pytest

lint:
	poetry run ruff check .
	poetry run isort --check-only --profile black .

fmt:
	poetry run isort --profile black .
	poetry run black .

e2e:
	poetry run dnr quickstart --no-llm --fast
