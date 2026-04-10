# Contributing to PyOptEx

Thanks for your interest in contributing! Whether it's a bug fix, new feature,
documentation improvement, or just a question, your help is welcome.

## Reporting issues

Use the [GitHub issue tracker](https://github.com/mborn1/pyoptex/issues) to
report bugs or request features. The issue templates will guide you through
providing the information we need.

## Development setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/mborn1/pyoptex.git
   cd pyoptex
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Install in editable mode with dev dependencies**

   ```bash
   venv/bin/pip install -e ".[dev]"
   ```

   This compiles the Cython extensions and installs all development tools
   (ruff, mypy, pytest, Sphinx, etc.).

## Verification checklist

Run these checks before submitting a pull request. All of them are available
as Makefile targets (the Makefile uses `venv/bin/` so activation is optional):

| Command           | What it does                       |
|-------------------|------------------------------------|
| `make lint`       | Ruff linter on `src/` and `tests/` |
| `make format`     | Ruff formatter (black-compatible)  |
| `make typecheck`  | mypy type checker                  |
| `make test`       | pytest with reproducible env vars  |
| `make all`        | lint + typecheck + test            |

Pre-commit hooks run the same checks automatically on `git commit`.

## Code style

- **Line length**: 120 characters (enforced by ruff).
- **Import order**: stdlib, third-party, first-party (`pyoptex`), separated by
  blank lines. Use `ruff check --fix` to auto-sort.
- **Type hints**: Add type hints to new public functions and methods.
- **Docstrings**: Google-style, consistent with existing code.
- **Numpy conventions**: Prefer vectorized operations over Python loops.

## Cython (.pyx files)

The performance-critical paths are in Cython. Only modify `.pyx` files if you
understand Cython. Key rules:

- Prefer typed memoryviews over `np.ndarray` for buffer access.
- Use `cdef` for C-only functions, `cpdef` for dual-callable functions.
- Annotate local variables for performance (`cdef int i`, `cdef double val`).
- Rebuild after editing: `venv/bin/pip install -e ".[dev]"` or
  `venv/bin/python setup.py build_ext --inplace`.

## Tests

- Add tests in `tests/` for new public API.
- Tests use seeded RNG (`set_seed(42)` via the `conftest.py` autouse fixture)
  and compare against reference JSON files in `tests/references/`.
- Always run tests via `make test`, which sets environment variables for
  reproducible numerics (single-threaded BLAS, CPU feature caps).
- To regenerate reference files after intentional changes:
  `make capture-references`.

## Examples

- Add standalone example scripts in `examples/` for new features.
- Examples should be runnable with `venv/bin/python examples/run_all.py`.

## Documentation

- Documentation lives in `docs/` and is built with Sphinx.
- Build locally: `make docs` (output in `docs/_build/html/`).
- Update or add `.rst` files in `docs/source/` for new features.

## Version and citation

- The version number lives only in `src/pyoptex/__init__.py`.
- `CITATION.cff` is kept in sync automatically. After a version bump, run
  `make sync-citation` and commit the result.

## Pull requests

- One logical change per PR.
- Follow the [PR template](.github/pull_request_template.md) checklist.
- If your change is user-facing, add an entry to `CHANGELOG.md`.

## License

By contributing, you agree that your contributions will be licensed under the
[BSD 3-Clause License](LICENSE).
