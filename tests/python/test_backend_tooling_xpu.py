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


@pytest.mark.quality
def test_xpu_backend_check_is_optional_without_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    class ProbeResult:
        returncode = 1

    def missing_probe(*_args, **_kwargs) -> ProbeResult:
        return ProbeResult()

    monkeypatch.setattr(backend_checks().subprocess, "run", missing_probe)

    assert backend_checks().check_xpu(required=False) == 0, (
        f"expected 0, got {backend_checks().check_xpu(required=False)}"
    )
    assert backend_checks().check_xpu(required=True) == 1, (
        f"expected 1, got {backend_checks().check_xpu(required=True)}"
    )
