#!/usr/bin/env python3
"""Audit CUDA kernel launches for explicit launch-status checks."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GROUPS_PATH = ROOT / "src" / "cmake" / "source_groups.cmake"
DEFAULT_SOURCE_GROUPS = (
    "NERVE_CORE_SOURCES",
    "NERVE_CUDA_SOURCES",
    "NERVE_CUDA_EXTENDED_SOURCES",
    "NERVE_PYTORCH_SOURCES",
    "NERVE_CUDNN_SOURCES",
    "NERVE_EXPERIMENTAL_TUNING_SOURCES",
)
ALL_SOURCE_SCOPE = "all"
CONFIGURED_SOURCE_SCOPE = "configured"
MANDATORY_CUDA_SOURCES = (
    ROOT / "src" / "include" / "nerve" / "persistence" / "accelerated" / "gpu_apparent_pairs.hpp",
    ROOT / "src" / "include" / "nerve" / "persistence" / "cuda" / "cuda_error_handling.hpp",
    ROOT / "src" / "persistence" / "cuda" / "cluster_16_block.cu",
    ROOT / "src" / "persistence" / "cuda" / "cluster_distributed_l2.cu",
    ROOT / "src" / "persistence" / "cuda" / "cluster_tma_multicast.cu",
    ROOT / "src" / "cuda" / "kernels" / "bottleneck_distance.cu",
    ROOT / "src" / "cuda" / "kernels" / "distance_fasted.cu",
    ROOT / "src" / "cuda" / "kernels" / "distance_kernels_tensorcore.cuh",
    ROOT / "src" / "cuda" / "kernels" / "distance_tedjoin.cu",
    ROOT / "src" / "cuda" / "kernels" / "gpu_persistence_launcher.cu",
    ROOT / "src" / "cuda" / "kernels" / "mapper_gpu.cu",
    ROOT / "src" / "cuda" / "kernels" / "persistence_image.cu",
    ROOT / "src" / "cuda" / "kernels" / "specseq_reduction.cu",
    ROOT / "src" / "cuda" / "kernels" / "wasserstein_distance.cu",
    ROOT / "src" / "persistence" / "cuda" / "kernel_hypha_scan.cu",
)
CUDA_SOURCE_SUFFIXES = (".cu", ".cuh", ".cpp", ".hpp", ".h", ".inl")
LAUNCH_GUARD_TOKENS = (
    "cudaGetLastError",
    "cudaPeekAtLastError",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "CUDA_KERNEL_CHECK",
    "GPU_CHECK",
    "CUDA_CHECK",
    "checkCuda",
    "check_cuda_operation",
    "validateKernelLaunch",
    "cuda_check_kernel",
)
KERNEL_GRID_CONTRACTS = {
    "geometricTransformKernel": {
        "disallowed_grid": "sample_blocks",
        "message": "geometric transform is element-indexed and must launch with an element-count grid",
    },
}


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    message: str


def _strip_comments(text: str) -> str:
    result: list[str] = []
    in_block = False
    index = 0
    while index < len(text):
        if in_block:
            if text.startswith("*/", index):
                in_block = False
                index += 2
            else:
                if text[index] == "\n":
                    result.append("\n")
                index += 1
            continue
        if text.startswith("/*", index):
            in_block = True
            index += 2
            continue
        if text.startswith("//", index):
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        result.append(text[index])
        index += 1
    return "".join(result)


def _cmake_group_sources(group: str) -> list[Path]:
    if not SOURCE_GROUPS_PATH.exists():
        return []
    text = SOURCE_GROUPS_PATH.read_text(encoding="utf-8")
    match = re.search(rf"set\(\s*{re.escape(group)}(?P<body>.*?)\n\)", text, re.DOTALL)
    if match is None:
        return []
    sources: list[Path] = []
    for line in match.group("body").splitlines():
        source = line.split("#", 1)[0].strip()
        if not source or source.startswith("$"):
            continue
        sources.append(ROOT / "src" / source)
    return sources


def _local_includes(path: Path) -> list[Path]:
    if not path.exists() or not path.is_file():
        return []
    includes: list[Path] = []
    text = _strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
    for include in re.findall(r'#include\s+"([^"]+)"', text):
        candidate = (path.parent / include).resolve()
        if candidate.exists() and candidate.is_file() and candidate.suffix in CUDA_SOURCE_SUFFIXES:
            includes.append(candidate)
    return includes


def included_launch_sources(paths: Iterable[Path]) -> list[Path]:
    sources: list[Path] = []
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        for include in _local_includes(path):
            if include in seen:
                continue
            seen.add(include)
            if "<<<" in include.read_text(encoding="utf-8", errors="ignore"):
                sources.append(include)
            visit(include)

    for path in paths:
        visit(path.resolve())
    return sources


def default_audit_sources(source_groups: Iterable[str] = DEFAULT_SOURCE_GROUPS) -> list[Path]:
    sources: list[Path] = []
    seen: set[Path] = set()
    for group in source_groups:
        for path in _cmake_group_sources(group):
            if path in seen or path.suffix not in CUDA_SOURCE_SUFFIXES:
                continue
            seen.add(path)
            sources.append(path)
    for path in MANDATORY_CUDA_SOURCES:
        if path in seen or path.suffix not in CUDA_SOURCE_SUFFIXES:
            continue
        seen.add(path)
        sources.append(path)
    for path in included_launch_sources(sources):
        if path in seen or path.suffix not in CUDA_SOURCE_SUFFIXES:
            continue
        seen.add(path)
        sources.append(path)
    return sources


def _launch_sources_under(root: Path) -> list[Path]:
    sources: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in CUDA_SOURCE_SUFFIXES:
            continue
        if "<<<" not in path.read_text(encoding="utf-8", errors="ignore"):
            continue
        sources.append(path)
    return sources


def all_launch_sources(root: Path | None = None) -> list[Path]:
    return _launch_sources_under(root or ROOT / "src")


def audit_sources(scope: str, source_groups: Iterable[str] = DEFAULT_SOURCE_GROUPS) -> list[Path]:
    if scope == CONFIGURED_SOURCE_SCOPE:
        return default_audit_sources(source_groups)
    if scope == ALL_SOURCE_SCOPE:
        return all_launch_sources()
    raise ValueError(f"unsupported CUDA launch audit scope: {scope}")


def coverage_findings(
    audited_sources: Iterable[Path] | None = None,
    launch_sources: Iterable[Path] | None = None,
) -> list[Finding]:
    audited = {
        path.resolve()
        for path in (audited_sources if audited_sources is not None else default_audit_sources())
        if path.suffix in CUDA_SOURCE_SUFFIXES
    }
    findings: list[Finding] = []
    sources = launch_sources if launch_sources is not None else all_launch_sources()
    for path in sources:
        resolved = path.resolve()
        if resolved in audited:
            continue
        rel = (
            resolved.relative_to(ROOT).as_posix()
            if resolved.is_relative_to(ROOT)
            else str(resolved)
        )
        findings.append(
            Finding(
                "cuda-launch-coverage",
                rel,
                "CUDA launch source is outside the configured launch audit scope; "
                "run tools/cuda_launch_audit.py --scope all to audit it",
            )
        )
    return findings


def _launch_end(lines: list[str], start: int) -> int:
    for index in range(start, min(len(lines), start + 20)):
        if ";" in lines[index]:
            return index
    return start


def _has_guard(lines: list[str], start: int, guard_window: int) -> bool:
    end = min(len(lines), start + guard_window + 1)
    return any(token in "\n".join(lines[start:end]) for token in LAUNCH_GUARD_TOKENS)


def iter_findings(paths: Iterable[Path], guard_window: int = 20) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if not path.exists() or not path.is_file() or path.suffix not in CUDA_SOURCE_SUFFIXES:
            continue
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        lines = _strip_comments(path.read_text(encoding="utf-8", errors="ignore")).splitlines()
        for index, line in enumerate(lines):
            if "<<<" not in line:
                continue
            launch_end = _launch_end(lines, index)
            launch_text = "\n".join(lines[index : launch_end + 1])
            for kernel, contract in KERNEL_GRID_CONTRACTS.items():
                match = re.search(rf"\b{re.escape(kernel)}\s*<<<\s*(?P<grid>[^,>]+)", launch_text)
                if match is not None and match.group("grid").strip() == contract["disallowed_grid"]:
                    findings.append(
                        Finding(
                            "cuda-launch-grid-contract",
                            rel,
                            f"line {index + 1}: {contract['message']}: {launch_text.strip()}",
                        )
                    )
            if _has_guard(lines, launch_end, guard_window):
                continue
            snippet = line.strip()
            if not snippet and index > 0:
                snippet = lines[index - 1].strip()
            if not snippet:
                snippet = "<launch>"
            findings.append(
                Finding(
                    "cuda-launch-audit",
                    rel,
                    f"line {index + 1}: CUDA launch is not followed by a launch-status check "
                    f"within {guard_window} lines: {snippet}",
                )
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--guard-window", type=int, default=20)
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="report launch sources outside the selected audit scope instead of per-launch guard findings",
    )
    parser.add_argument(
        "--scope",
        choices=(CONFIGURED_SOURCE_SCOPE, ALL_SOURCE_SCOPE),
        default=CONFIGURED_SOURCE_SCOPE,
    )
    parser.add_argument("--source-group", action="append", default=[])
    parser.add_argument("--path", action="append", type=Path, default=[])
    args = parser.parse_args()

    sources = list(args.path)
    if not sources:
        groups = tuple(args.source_group) if args.source_group else DEFAULT_SOURCE_GROUPS
        sources = audit_sources(args.scope, groups)

    normalized_sources = [path if path.is_absolute() else ROOT / path for path in sources]
    if args.coverage:
        findings = coverage_findings(normalized_sources, all_launch_sources())
    else:
        findings = iter_findings(normalized_sources, args.guard_window)
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    else:
        for finding in findings:
            print(f"{finding.check}: {finding.path}: {finding.message}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
