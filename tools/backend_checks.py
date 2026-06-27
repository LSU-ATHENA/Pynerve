#!/usr/bin/env python3
"""Backend verification helpers for CUDA and MPI test jobs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUTHY = {"1", "on", "true", "yes"}
CUDA_SANITIZER_TOOLS = ("memcheck", "racecheck", "synccheck")


def _run(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


def check_cuda(build_dir: str, required: bool) -> int:
    if shutil.which("nvcc") is None:
        print("nvcc is missing", file=sys.stderr)
        return 1 if required else 0
    version = subprocess.run(["nvcc", "--version"], text=True, capture_output=True, check=False)
    if "release 12" not in version.stdout + version.stderr:
        print(
            f"CUDA backend requires at least CUDA 12 (found {version.stdout.strip()})",
            file=sys.stderr,
        )
        return 1 if required else 0
    if shutil.which("compute-sanitizer") is None:
        print("compute-sanitizer is missing", file=sys.stderr)
        return 1 if required else 0
    cuda_hardware_count = _ctest_label_count(build_dir, "cuda-hardware")
    if cuda_hardware_count is None:
        print("could not inventory CUDA hardware CTest coverage", file=sys.stderr)
        return 1 if required else 0
    if cuda_hardware_count == 0:
        print("no CUDA hardware CTest coverage is registered", file=sys.stderr)
        return 1 if required else 0
    for sanitizer_tool in CUDA_SANITIZER_TOOLS:
        result = _run(
            [
                "compute-sanitizer",
                "--tool",
                sanitizer_tool,
                "--target-processes",
                "all",
                "--error-exitcode",
                "1",
                "ctest",
                "--test-dir",
                build_dir,
                "-L",
                "cuda-hardware",
                "--output-on-failure",
            ]
        )
        if result != 0:
            return result
    return 0


def _openmpi_cuda_support_status() -> tuple[bool | None, str]:
    if shutil.which("ompi_info") is None:
        return None, "ompi_info is missing; cannot confirm CUDA-aware Open MPI support"
    probe = subprocess.run(
        ["ompi_info", "--parsable", "--all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return None, "ompi_info failed; cannot confirm CUDA-aware Open MPI support"
    text = (probe.stdout + probe.stderr).lower()
    true_markers = (
        "mpi_built_with_cuda_support:value:true",
        "opal_built_with_cuda_support:value:true",
    )
    false_markers = (
        "mpi_built_with_cuda_support:value:false",
        "opal_built_with_cuda_support:value:false",
    )
    if any(marker in text for marker in true_markers):
        return True, "Open MPI reports CUDA-aware support"
    if any(marker in text for marker in false_markers):
        return False, "Open MPI reports CUDA-aware support is disabled"
    return None, "Open MPI CUDA-aware support marker was not found"


def _ctest_label_count(build_dir: str, label: str) -> int | None:
    probe = subprocess.run(
        ["ctest", "--test-dir", build_dir, "-N", "-L", label],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    match = re.search(r"Total Tests:\s*(\d+)", probe.stdout + probe.stderr)
    if match is None:
        return None
    return int(match.group(1))


def check_mpi(build_dir: str, required: bool, require_cuda_aware_mpi: bool = False) -> int:
    if shutil.which("mpirun") is None:
        print("mpirun is missing", file=sys.stderr)
        return 1 if required else 0
    env = os.environ.copy()
    cuda_aware_requested = (
        require_cuda_aware_mpi or env.get("NERVE_TEST_CUDA_AWARE_MPI", "").lower() in TRUTHY
    )
    if cuda_aware_requested:
        supported, message = _openmpi_cuda_support_status()
        if supported is not True:
            print(message, file=sys.stderr)
            if require_cuda_aware_mpi:
                return 1
        env["NERVE_TEST_CUDA_AWARE_MPI"] = "1"
    distributed_count = _ctest_label_count(build_dir, "distributed")
    if distributed_count is None:
        print("could not inventory distributed CTest coverage", file=sys.stderr)
        return 1 if required else 0
    if distributed_count == 0:
        print("no distributed CTest coverage is registered", file=sys.stderr)
        return 1 if required else 0
    return subprocess.call(
        ["ctest", "--test-dir", build_dir, "-L", "distributed", "--output-on-failure"],
        cwd=ROOT,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cuda", "mpi"), required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--required", action="store_true")
    parser.add_argument("--require-cuda-aware-mpi", action="store_true")
    args = parser.parse_args()
    if args.backend == "cuda":
        return check_cuda(args.build_dir, args.required)

    return check_mpi(args.build_dir, args.required, args.require_cuda_aware_mpi)


if __name__ == "__main__":
    raise SystemExit(main())
