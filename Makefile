VENV := venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest
PRECOMMIT := $(VENV)/bin/pre-commit

# Match CI: single-threaded math and CPU feature caps for reproducible tests
TEST_ENV := \
	OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE \
	MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE \
	OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
	PYTHON_CPU_COUNT=1 VECLIB_MAXIMUM_THREADS=1 \
	NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR \
	OPENBLAS_CORETYPE=HASWELL

.PHONY: lint format typecheck test test-cov build docs pre-commit all help sync-citation check-citation

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
	@echo "  sync-citation Sync CITATION.cff version from __init__.py"
	@echo "  all         Run lint, typecheck, and test"

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/

typecheck:
	$(MYPY) src/pyoptex/

capture-references:
	$(TEST_ENV) python tests/_capture_references.py

test:
	$(TEST_ENV) $(PYTEST)

test-cov:
	$(TEST_ENV) $(PYTEST) --cov=pyoptex --cov-report=term-missing

build:
	$(PYTHON) -m build

docs:
	$(MAKE) -C docs html

pre-commit:
	$(PRECOMMIT) run --all-files

sync-citation:
	$(PYTHON) scripts/sync_citation_version.py

check-citation:
	$(PYTHON) scripts/sync_citation_version.py --check

all: lint typecheck test
