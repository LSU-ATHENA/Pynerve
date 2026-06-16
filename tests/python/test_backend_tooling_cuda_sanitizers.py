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


def cuda_launch_audit():
    return _get_tool("cuda_launch_audit")


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.quality
@pytest.mark.skipif(not _torch_cuda_available(), reason="torch with CUDA not available")
@pytest.mark.skipif(not _torch_cuda_available(), reason="torch with CUDA not available")
def test_cuda_backend_runs_memory_race_and_sync_sanitizers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def available_tool(name: str) -> str:
        return f"/usr/bin/{name}"

    class VersionResult:
        stdout = "Cuda compilation tools, release 12.4, V12.4.0"
        stderr = ""

    class InventoryResult:
        returncode = 0
        stdout = "Total Tests: 1"
        stderr = ""

    def recording_run(command, **_kwargs):
        if command == ["nvcc", "--version"]:
            return VersionResult()
        if command[:4] == ["ctest", "--test-dir", "build", "-N"]:
            return InventoryResult()
        raise AssertionError(command)

    def recording_backend_run(command, *, env=None) -> int:
        del env
        commands.append(command)
        return 0

    monkeypatch.setattr(backend_checks().shutil, "which", available_tool)
    monkeypatch.setattr(backend_checks().subprocess, "run", recording_run)
    monkeypatch.setattr(backend_checks(), "_run", recording_backend_run)

    assert backend_checks().check_cuda("build", required=True) == 0, (
        f"expected 0, got {backend_checks().check_cuda('build', required=True)}"
    )
    assert [command[2] for command in commands] == ["memcheck", "racecheck", "synccheck"], (
        f"expected sanitzers, got {[command[2] for command in commands]}"
    )
    assert all("--target-processes" in command for command in commands), (
        "all cuda commands must have --target-processes"
    )
    assert all(command[command.index("--target-processes") + 1] == "all" for command in commands), (
        "all --target-processes must be 'all'"
    )
    assert all(command[command.index("-L") + 1] == "cuda-hardware" for command in commands), (
        "all -L must be 'cuda-hardware'"
    )
    assert all("--error-exitcode" in command for command in commands), (
        "all cuda commands must have --error-exitcode"
    )


@pytest.mark.quality
def test_cuda_backend_required_check_fails_without_hardware_ctest_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available_tool(name: str) -> str:
        return f"/usr/bin/{name}"

    class VersionResult:
        stdout = "Cuda compilation tools, release 12.4, V12.4.0"
        stderr = ""

    class InventoryResult:
        returncode = 0
        stdout = "Total Tests: 0"
        stderr = ""

    def recording_run(command, **_kwargs):
        if command == ["nvcc", "--version"]:
            return VersionResult()
        if command[:4] == ["ctest", "--test-dir", "build", "-N"]:
            return InventoryResult()
        raise AssertionError(command)

    monkeypatch.setattr(backend_checks().shutil, "which", available_tool)
    monkeypatch.setattr(backend_checks().subprocess, "run", recording_run)

    assert backend_checks().check_cuda("build", required=True) == 1, (
        f"expected 1, got {backend_checks().check_cuda('build', required=True)}"
    )
    assert backend_checks().check_cuda("build", required=False) == 0, (
        f"expected 0, got {backend_checks().check_cuda('build', required=False)}"
    )


@pytest.mark.quality
def test_cuda_backend_wrong_version_is_optional_without_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available_tool(name: str) -> str:
        return f"/usr/bin/{name}"

    class VersionResult:
        returncode = 0
        stdout = "Cuda compilation tools, release 12.3, V12.3.0"
        stderr = ""

        monkeypatch.setattr(backend_checks().shutil, "which", available_tool)
        monkeypatch.setattr(
            backend_checks().subprocess, "run", lambda *_args, **_kwargs: VersionResult()
        )

    assert backend_checks().check_cuda("build", required=False) == 0, (
        f"expected 0, got {backend_checks().check_cuda('build', required=False)}"
    )
    assert backend_checks().check_cuda("build", required=True) == 1, (
        f"expected 1, got {backend_checks().check_cuda('build', required=True)}"
    )
