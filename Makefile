.PHONY: test lint fmt e2e bump

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

bump:
ifndef VERSION
	$(error VERSION is required (example: make bump VERSION=0.2.0 [DRY_RUN=1]))
endif
	python3 scripts/bump_version.py $(if $(strip $(DRY_RUN)),--dry-run,) $(VERSION)
ifeq ($(strip $(DRY_RUN)),)
	git add pyproject.toml src/data_needs_reporter/__init__.py
	git commit -m "Bump version to $(VERSION)"
	git tag v$(VERSION)
else
	@echo "DRY RUN: git add pyproject.toml src/data_needs_reporter/__init__.py"
	@echo "DRY RUN: git commit -m \"Bump version to $(VERSION)\""
	@echo "DRY RUN: git tag v$(VERSION)"
endif
