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
def test_runner_detects_missing_accelerator_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NERVE_TEST_CUDA", raising=False)

    assert run_tests()._accelerator_missing(["cuda"]) == "cuda", "expected cuda to be missing"
    assert run_tests()._accelerator_missing(["cpu", "generated"]) is None, (
        "expected cpu/generated not to be missing"
    )
    assert run_tests()._pytest_marker_expression(["cpu", "generated"]) == "cpu or generated", (
        f"unexpected marker expression: {run_tests()._pytest_marker_expression(['cpu', 'generated'])}"
    )
    assert run_tests()._available_labels_for_environment(["cuda", "generated", "distributed"]) == (
        ["generated", "distributed"],
        ["cuda"],
    ), (
        f"unexpected labels: {run_tests()._available_labels_for_environment(['cuda', 'generated', 'distributed'])}"
    )


@pytest.mark.quality
def test_runner_changed_path_labels_cover_nested_cuda_and_distributed_sources() -> None:
    labels = run_tests()._labels_for_paths(
        [
            "src/graphs/attention_gpu.cu",
            "src/persistence/distributed/mpi_distributed_ph.cpp",
            "python/nerve/torch/__init__.py",
            "scripts/hpc_cuda_mpi_verify.sbatch",
        ]
    )

    assert labels == ["cuda", "distributed", "generated", "gradient", "python"], (
        f"expected labels ['cuda', 'distributed', 'generated', 'gradient', 'python'], got {labels}"
    )
