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
def test_cuda_launch_audit_requires_post_launch_status_check(tmp_path: Path) -> None:
    guarded = tmp_path / "guarded.cu"
    guarded.write_text(
        """
__global__ void kernel() {}
void launch() {
    kernel<<<1, 32>>>();
    GPU_CHECK(cudaPeekAtLastError());
}
""",
        encoding="utf-8",
    )
    sync_guarded = tmp_path / "sync_guarded.cu"
    sync_guarded.write_text(
        """
__global__ void kernel() {}
void launch(cudaStream_t stream, cudaEvent_t done) {
    kernel<<<1, 32, 0, stream>>>();
    cudaStreamSynchronize(stream);
    cudaEventSynchronize(done);
}
""",
        encoding="utf-8",
    )
    unguarded = tmp_path / "unguarded.cu"
    unguarded.write_text(
        """
__global__ void kernel() {}
void launch() {
    kernel<<<1, 32>>>();
    return;
}
""",
        encoding="utf-8",
    )

    assert cuda_launch_audit().iter_findings([guarded]) == [], (
        f"expected no findings for guarded, got {cuda_launch_audit().iter_findings([guarded])}"
    )
    assert cuda_launch_audit().iter_findings([sync_guarded]) == [], (
        f"expected no findings for sync_guarded, got {cuda_launch_audit().iter_findings([sync_guarded])}"
    )
    findings = cuda_launch_audit().iter_findings([unguarded])
    assert len(findings) == 1, f"expected 1 finding for unguarded, got {len(findings)}"
    assert findings[0].check == "cuda-launch-audit", (
        f"expected 'cuda-launch-audit', got {findings[0].check}"
    )


@pytest.mark.quality
def test_cuda_launch_audit_rejects_sample_grid_for_element_indexed_kernel(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid_grid.cu"
    invalid.write_text(
        """
__global__ void geometricTransformKernel(float*, float*, int, int, float, float) {}
void launch() {
    geometricTransformKernel<<<sample_blocks, BLOCK_SIZE>>>(
        nullptr, nullptr, num_samples, feature_dim, rotation, scale);
    GPU_CHECK(cudaPeekAtLastError());
}
""",
        encoding="utf-8",
    )
    valid = tmp_path / "valid_grid.cu"
    valid.write_text(
        """
__global__ void geometricTransformKernel(float*, float*, int, int, float, float) {}
void launch() {
    geometricTransformKernel<<<blocks, BLOCK_SIZE>>>(
        nullptr, nullptr, num_samples, feature_dim, rotation, scale);
    GPU_CHECK(cudaPeekAtLastError());
}
""",
        encoding="utf-8",
    )

    findings = cuda_launch_audit().iter_findings([invalid])
    assert len(findings) == 1, f"expected 1 finding for invalid grid, got {len(findings)}"
    assert findings[0].check == "cuda-launch-grid-contract", (
        f"expected 'cuda-launch-grid-contract', got {findings[0].check}"
    )
    assert cuda_launch_audit().iter_findings([valid]) == [], (
        f"expected no findings for valid, got {cuda_launch_audit().iter_findings([valid])}"
    )


@pytest.mark.quality
def test_cuda_launch_audit_scope_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    launch = src / "kernel.cu"
    launch.write_text(
        """
__global__ void kernel() {}
void launch() {
    kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
}
""",
        encoding="utf-8",
    )

    audit = cuda_launch_audit()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(
        audit, "SOURCE_GROUPS_PATH", tmp_path / "src" / "cmake" / "source_groups.cmake"
    )
    monkeypatch.setattr(audit, "MANDATORY_CUDA_SOURCES", ())
    assert audit.all_launch_sources(src) == [launch], (
        f"expected [launch], got {audit.all_launch_sources(src)}"
    )
    assert audit.audit_sources(audit.ALL_SOURCE_SCOPE) == [launch], (
        f"expected [launch], got {audit.audit_sources(audit.ALL_SOURCE_SCOPE)}"
    )
    assert audit.audit_sources(audit.CONFIGURED_SOURCE_SCOPE) == [], (
        f"expected [], got {audit.audit_sources(audit.CONFIGURED_SOURCE_SCOPE)}"
    )

    coverage = audit.coverage_findings([], [launch])
    assert len(coverage) == 1, f"expected 1 coverage finding, got {len(coverage)}"
    assert coverage[0].check == "cuda-launch-coverage", (
        f"expected 'cuda-launch-coverage', got {coverage[0].check}"
    )
    assert audit.coverage_findings([launch], [launch]) == [], (
        f"expected no coverage findings, got {audit.coverage_findings([launch], [launch])}"
    )


@pytest.mark.quality
def test_cuda_launch_audit_follows_configured_wrapper_includes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    cmake = src / "cmake"
    cmake.mkdir(parents=True)
    wrapper = src / "wrapper.cu"
    wrapper.write_text('#include "kernel_api.inl"\n', encoding="utf-8")
    included = src / "kernel_api.inl"
    included.write_text(
        """
__global__ void kernel() {}
void launch() {
    kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
}
""",
        encoding="utf-8",
    )
    (cmake / "source_groups.cmake").write_text(
        """
set(NERVE_CUDA_SOURCES
    wrapper.cu
)
""",
        encoding="utf-8",
    )

    audit = cuda_launch_audit()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "SOURCE_GROUPS_PATH", cmake / "source_groups.cmake")
    monkeypatch.setattr(audit, "MANDATORY_CUDA_SOURCES", ())

    assert audit.audit_sources(audit.CONFIGURED_SOURCE_SCOPE) == [
        wrapper,
        included.resolve(),
    ], f"unexpected audit sources: {audit.audit_sources(audit.CONFIGURED_SOURCE_SCOPE)}"
    assert audit.iter_findings([included]) == [], (
        f"expected no findings, got {audit.iter_findings([included])}"
    )
