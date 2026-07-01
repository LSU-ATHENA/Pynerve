from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _source_path(*parts: str) -> Path:
    """Resolve a source path, with a friendly message if the file is missing."""
    path = ROOT.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"Source file not found: {path}. Skipping test.")
    return path


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"nerve_{name}", ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"could not load tool module {name}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TOOL_CACHE: dict[str, Any] = {}


def _get_tool(name: str):
    if name not in _TOOL_CACHE:
        _TOOL_CACHE[name] = _load_tool(name)
    return _TOOL_CACHE[name]


def run_tests():
    return _get_tool("run_tests")


def backend_checks():
    return _get_tool("backend_checks")


@pytest.mark.quality
def test_mpi_backend_requires_cuda_aware_runtime_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available_tool(name: str) -> str:
        return f"/usr/bin/{name}"

    class ProbeResult:
        returncode = 0
        stdout = "mpi_built_with_cuda_support:value:false"
        stderr = ""

    monkeypatch.setattr(backend_checks().shutil, "which", available_tool)
    monkeypatch.setattr(backend_checks().subprocess, "run", lambda *_args, **_kwargs: ProbeResult())

    assert (
        backend_checks().check_mpi(
            "build",
            required=True,
            require_cuda_aware_mpi=True,
        )
        == 1
    ), (
        f"expected 1, got {backend_checks().check_mpi('build', required=True, require_cuda_aware_mpi=True)}"
    )


@pytest.mark.quality
def test_mpi_backend_passes_cuda_aware_request_to_ctest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available_tool(name: str) -> str:
        return f"/usr/bin/{name}"

    class ProbeResult:
        returncode = 0
        stdout = "mpi_built_with_cuda_support:value:true"
        stderr = ""

    class InventoryResult:
        returncode = 0
        stdout = "Total Tests: 1"
        stderr = ""

    def recording_run(command, **_kwargs):
        if command[:2] == ["ompi_info", "--parsable"]:
            return ProbeResult()
        if command[:4] == ["ctest", "--test-dir", "build", "-N"]:
            return InventoryResult()
        raise AssertionError(command)

    def recording_call(command, *, cwd, env) -> int:
        assert command[:4] == ["ctest", "--test-dir", "build", "-L"], (
            f"expected ctest -L command, got {command}"
        )
        assert cwd == backend_checks().ROOT, f"expected cwd {backend_checks().ROOT}, got {cwd}"
        assert env["NERVE_TEST_CUDA_AWARE_MPI"] == "1", (
            f"expected CUDA_AWARE_MPI=1, got {env.get('NERVE_TEST_CUDA_AWARE_MPI')}"
        )
        return 0

    monkeypatch.setattr(backend_checks().shutil, "which", available_tool)
    monkeypatch.setattr(backend_checks().subprocess, "run", recording_run)
    monkeypatch.setattr(backend_checks().subprocess, "call", recording_call)

    assert (
        backend_checks().check_mpi(
            "build",
            required=True,
            require_cuda_aware_mpi=True,
        )
        == 0
    ), (
        f"expected 0, got {backend_checks().check_mpi('build', required=True, require_cuda_aware_mpi=True)}"
    )


@pytest.mark.quality
def test_mpi_backend_required_check_fails_without_registered_distributed_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available_tool(name: str) -> str:
        return f"/usr/bin/{name}"

    class InventoryResult:
        returncode = 0
        stdout = "Total Tests: 0"
        stderr = ""

    monkeypatch.setattr(backend_checks().shutil, "which", available_tool)
    monkeypatch.setattr(
        backend_checks().subprocess, "run", lambda *_args, **_kwargs: InventoryResult()
    )

    assert backend_checks().check_mpi("build", required=True) == 1, (
        f"expected 1, got {backend_checks().check_mpi('build', required=True)}"
    )
    assert backend_checks().check_mpi("build", required=False) == 0, (
        f"expected 0, got {backend_checks().check_mpi('build', required=False)}"
    )
