# Contributing

## Quick start

```bash
git clone https://github.com/LSU-ATHENA/Pynerve
cd pynerve

# Install Python package in editable mode (requires C++20, CMake >=3.20)
pip install -e ./python

# Install dev dependencies
pip install pytest pytest-cov pytest-xdist hypothesis ruff mypy pre-commit
```

### Using the Makefile

The project includes a `Makefile` with common development commands.
Run `make help` to see all available targets:

| Command | Description |
|---------|-------------|
| `make install` | Install the package in editable mode |
| `make dev` | Install dev dependencies |
| `make test` | Run all Python tests |
| `make test-quick` | Run fast tests (skip slow, cuda, distributed) |
| `make test-coverage` | Run tests with branch coverage report |
| `make lint` | Run ruff linter |
| `make format` | Check formatting |
| `make typecheck` | Run mypy type checking |
| `make quality` | Run lint + typecheck |
| `make ci-local` | Run CI-equivalent checks locally |
| `make clean` | Remove build artifacts |
| `make pre-commit` | Run all pre-commit hooks |

## System requirements

| Dependency | Minimum version | Notes |
|------------|---------------|-------|
| C++ compiler | C++20 | GCC >=11, Clang >=14, MSVC 2022 |
| CMake | 3.20 | Build system |
| Python | 3.10 | 3.10--3.13 supported |
| CUDA (optional) | 12.4 | For GPU acceleration |
| MPI (optional) | Any | For distributed computation |
| Ninja (recommended) | Any | Faster builds |
| ccache (recommended) | Any | Caching C++ compilations |

Optional system libraries: `liburing` (Linux I/O), `mimalloc` (fast allocator).

### Feature flags

The C++ build supports these CMake feature flags, passed via environment or
`scripts/build_*.sh`:

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_PH5` | ON | PH5 balanced engine |
| `ENABLE_PH6` | ON | PH6 high-precision engine |
| `ENABLE_GREEN_CONTEXTS` | ON | Green thread contexts |
| `ENABLE_CUDA` | OFF | CUDA GPU support |
| `ENABLE_MPI` | OFF | MPI distributed support |
| `ENABLE_EIGEN` | OFF | Eigen3 linear algebra |

## Running tests

```bash
# All Python tests
pytest tests/python/ -v

# Without C++ extension (fast, limited scope)
pytest tests/python/test_python_api.py -v
pytest tests/python/test_api_coverage.py -v

# With coverage
pytest tests/python/ --cov=python/pynerve --cov-branch --cov-report=term-missing

# A single test file
pytest tests/python/test_module_smoke.py -v

# Only fast tests (skip slow, cuda, distributed markers)
pytest tests/python/ -v -m "not slow and not cuda and not distributed"
```

Or use the Makefile shortcuts:
```bash
make test          # all tests
make test-quick    # fast subset
make test-coverage # with coverage
```

## Linting and type checking

```bash
# Python linting
ruff check python/pynerve/ tests/

# Format check
ruff format --check python/pynerve/ tests/

# Type checking
mypy python/pynerve/ --ignore-missing-imports

# C++ formatting
find tests/cpp/ -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=file
```

Or use the Makefile:
```bash
make lint
make format
make typecheck
make quality    # all three
```

## Pre-commit hooks (recommended)

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Or: `make pre-commit`

## Building C++ extension

```bash
pip install -e ./python  # builds in-place
# or for a clean build:
pip install -e ./python --no-build-isolation
```

## Pull request guidelines

- Include tests for new functionality
- Ensure all existing tests pass
- Run `make quality` before submitting
- Keep the API reference docs in `doc/reference/api_python.md` in sync
- Mark GPU/MPI/torch-dependent tests with the appropriate pytest marker

## API Stability Policy

Pynerve follows [Semantic Versioning](https://semver.org). Until 2.0, the following guarantees apply:

### Public API

These modules are covered by the stability guarantee — breaking changes require a major version bump:

- `pynerve.compute_persistence()` and top-level functions
- `pynerve.PersistenceResult` and data classes
- `pynerve.fast_ops` — vectorized NumPy-backed operations
- `pynerve.torch` — PyTorch interoperability

### Semi-Public API

These modules may receive additions but breaking changes will be announced one minor version in advance:

- `pynerve.nn` — neural network layers
- `pynerve.mapper` — Mapper algorithm
- `pynerve.diff` — differentiable topology (experimental)

### Internal API

Modules prefixed with `_` (e.g., `pynerve._compute_core`, `pynerve._validation`) and C++ bindings are internal and may change without notice.

### Deprecation Policy

- Deprecated features are marked with `DeprecationWarning` for at least one minor release before removal
- Breaking changes to public API trigger a major version bump
- Check the [CHANGELOG.md](CHANGELOG.md) for migration notes

## Project structure

```
python/pynerve/        # Python package source
  __init__.py          # Public API exports
  _compute_api.py      # compute_persistence and friends
  _fallback_classes.py # Pure-Python class definitions
  fast_ops.py          # NumPy-backed vectorized operations
  torch/               # PyTorch sub-package
  nn/                  # Neural network sub-package
  ...
tests/
  python/              # Python test suite (pytest)
  cpp/                 # C++ regression tests
doc/                   # Markdown documentation
```
