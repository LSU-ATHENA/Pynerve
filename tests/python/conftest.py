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
    parser.addoption(
        "--flaky-report",
        type=str,
        default=None,
        help="Write flaky test results to this JSON file for CI tracking. "
        "Tests marked @pytest.mark.flaky are tracked; failures are recorded "
        "with test name, nodeid, and failure reason.",
    )


def pytest_configure(config: pytest.Config) -> None:
    for marker in (
        "fast",
        "slow",
        "flaky",
        "generated",
        "cpu",
        "cuda",
        "gpu_smoke",
        "gpu_slow",
        "gpu_multi",
        "gpu_benchmark",
        "sm75",
        "sm80",
        "sm90",
        "gradient",
        "autograd",
        "operators",
        "distributed",
        "performance",
        "quality",
        "torch",
        "nerve_extras",
    ):
        config.addinivalue_line("markers", f"{marker}: Nerve generated test category")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    for item in items:
        if not item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.fast)

    # Auto-skip SM capability markers when GPU isn't capable enough.
    # sm75 = Turing (RTX 20xx, T4), sm80 = Ampere (A100, RTX 30xx),
    # sm90 = Hopper (H100). Tests auto-skip if the GPU is below the required tier.
    cc = _cuda_compute_capability()
    if cc is not None:
        cc_value = cc[0] * 10 + cc[1]  # e.g. (8, 6) → 86
        for item in items:
            _apply_sm_skip(item, cc_value)


def _apply_sm_skip(item: pytest.Item, cc_value: int) -> None:
    """Skip a test if the GPU compute capability is below the required SM tier."""
    sm_requirements = {
        "sm75": 75,
        "sm80": 80,
        "sm90": 90,
    }
    for marker_name, min_cc in sm_requirements.items():
        if item.get_closest_marker(marker_name) and cc_value < min_cc:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"Requires SM {min_cc // 10}.{min_cc % 10}+ "
                    f"(current: SM {cc_value // 10}.{cc_value % 10})"
                )
            )
            return


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


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _cuda_compute_capability() -> tuple[int, int] | None:
    if not _cuda_available():
        return None
    import torch

    return torch.cuda.get_device_capability(0)


def _cuda_device_count() -> int:
    if not _cuda_available():
        return 0
    import torch

    return torch.cuda.device_count()


@pytest.fixture(scope="session")
def torch():
    pytest.importorskip("torch")
    import torch as _torch

    return _torch


@pytest.fixture(scope="session")
def cuda_device():
    if not _cuda_available():
        pytest.skip("CUDA device not available")
    import torch

    return torch.cuda


@pytest.fixture(scope="session")
def multi_gpu():
    count = _cuda_device_count()
    if count < 2:
        pytest.skip(f"Multi-GPU test requires at least 2 CUDA devices, found {count}")
    import torch

    return torch.cuda


# ---------------------------------------------------------------------------
# Flaky test tracking — CI integration for GPU test reliability monitoring
# ---------------------------------------------------------------------------

_flaky_results: list[dict[str, object]] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """Track test outcomes for flaky test reporting.

    Tests marked @pytest.mark.flaky have their pass/fail status recorded.
    Results are written to the --flaky-report JSON file at session end.
    """
    outcome = yield
    report = outcome.get_result()

    if item.get_closest_marker("flaky") and report.when == "call":
        _flaky_results.append(
            {
                "nodeid": item.nodeid,
                "name": item.name,
                "outcome": report.outcome,
                "duration": round(report.duration, 4),
                "message": (
                    report.longreprtext[:500]
                    if hasattr(report, "longreprtext") and report.longreprtext
                    else ""
                ),
            }
        )


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write flaky test report if --flaky-report was specified."""
    report_path = session.config.getoption("--flaky-report", default=None)
    if report_path and _flaky_results:
        import json

        failures = [r for r in _flaky_results if r["outcome"] == "failed"]
        report = {
            "total_flaky_tests": len(_flaky_results),
            "failures": len(failures),
            "pass_rate": (
                round((len(_flaky_results) - len(failures)) / len(_flaky_results) * 100, 1)
                if _flaky_results
                else 100.0
            ),
            "results": _flaky_results,
        }
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
