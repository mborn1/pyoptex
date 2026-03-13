VENV := venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest
PRECOMMIT := $(VENV)/bin/pre-commit

.PHONY: lint format typecheck test test-cov build docs pre-commit all help

help:
	@echo "Available targets:"
	@echo "  lint        Run ruff linter over src/ and tests/"
	@echo "  format      Run ruff formatter over src/ and tests/"
	@echo "  typecheck   Run mypy type checker over src/pyoptex/"
	@echo "  test        Run pytest"
	@echo "  test-cov    Run pytest with coverage report"
	@echo "  build       Build distribution wheels"
	@echo "  docs        Build Sphinx HTML docs"
	@echo "  pre-commit  Run all pre-commit hooks on all files"
	@echo "  all         Run lint, typecheck, and test"

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/

typecheck:
	$(MYPY) src/pyoptex/

test:
	$(PYTEST)

test-cov:
	$(PYTEST) --cov=pyoptex --cov-report=term-missing

build:
	$(PYTHON) -m build

docs:
	$(MAKE) -C docs html

pre-commit:
	$(PRECOMMIT) run --all-files

all: lint typecheck test
