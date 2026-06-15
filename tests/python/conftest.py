from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
TOOLS_ROOT = ROOT / "tools"
# Prefer installed nerve package over local checkout to detect import bugs.
# Only add paths if nerve is not already importable or we are in a CI/dev setting.
_nerve_installed = False
try:
    import pynerve  # noqa: F401

    _nerve_installed = True
except ImportError:
    pass
if not _nerve_installed and str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


def _nerve_core_available() -> bool:
    """Check whether nerve_internal (C++ extension) is available."""
    try:
        import pynerve

        return getattr(pynerve, "_core", None) is not None
    except ImportError:
        return False


@pytest.fixture(scope="session")
def nerve_root() -> Path:
    """Project root directory."""
    return ROOT


@pytest.fixture(scope="session")
def nerve_core() -> None:
    """Skip test if nerve_internal (C++ extension) is not available."""
    if not _nerve_core_available():
        pytest.skip(
            "nerve_internal C++ extension not available; "
            "build with 'pip install -e .' or 'make build-python'"
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--nerve-shard-index",
        type=int,
        default=0,
        help="Shard index for parallel test execution (0-based). "
        "Use with --nerve-shard-count to split tests across workers: "
        "pytest --nerve-shard-index=0 --nerve-shard-count=4 & "
        "pytest --nerve-shard-index=1 --nerve-shard-count=4 & ...",
    )
    parser.addoption(
        "--nerve-shard-count",
        type=int,
        default=1,
        help="Total shard count for parallel test execution. "
        "Tests are divided evenly across shards. "
        "Also configurable via NERVE_SHARD_INDEX and NERVE_SHARD_COUNT env vars.",
    )


def pytest_configure(config: pytest.Config) -> None:
    for marker in (
        "fast",
        "slow",
        "generated",
        "cpu",
        "cuda",
        "xpu",
        "gradient",
        "autograd",
        "operators",
        "distributed",
        "performance",
        "quality",
        "torch",
    ):
        config.addinivalue_line("markers", f"{marker}: Nerve generated test category")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    for item in items:
        if not item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.fast)


try:
    from test_matrix import iter_cases as _iter_cases  # noqa: PLC0415
    from test_matrix import select_cases as _select_cases  # noqa: PLC0415
    from test_matrix import total_case_count as _total_case_count  # noqa: PLC0415

    sys.modules["nerve_test_matrix"] = sys.modules["test_matrix"]  # compatibility shim
except ImportError:
    _iter_cases = lambda *_, **__: []  # noqa: E731
    _select_cases = lambda cases, *_, **__: []  # noqa: E731
    _total_case_count = lambda *_, **__: 0  # noqa: E731


@pytest.fixture(scope="session")  # noqa: N802
def generated_cases(request):  # noqa: N803
    shard_count = request.config.getoption("--nerve-shard-count")
    cases = _select_cases(
        _iter_cases(include_inactive=False),
        request.config.getoption("--nerve-shard-index"),
        shard_count,
    )
    min_expected = max(1, _total_case_count(include_inactive=False) // (shard_count * 2))
    assert len(cases) >= min_expected
    return cases


@pytest.fixture(scope="session")
def torch():
    """Skip test if torch is not available."""
    pytest.importorskip("torch")
    import torch as _torch

    return _torch
