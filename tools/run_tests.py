#!/usr/bin/env python3
"""Shard-aware test runner for local and CI use."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUTHY = {"1", "on", "true", "yes"}


def _run(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


def _labels_for_paths(paths: list[str]) -> list[str]:
    labels: set[str] = set()
    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        lower_path = path.lower()
        if (
            lower_path.endswith((".cu", ".cuh"))
            or "/cuda/" in lower_path
            or "cuda" in lower_path
            or "/gpu/" in lower_path
            or "_gpu" in lower_path
        ):
            labels.add("cuda")
        if "xpu" in lower_path:
            labels.add("xpu")
        if "/distributed/" in lower_path or lower_path.startswith("src/distributed/"):
            labels.add("distributed")
        if "autodiff" in lower_path or "torch" in lower_path or "diff" in lower_path:
            labels.add("gradient")
        if lower_path.startswith("python/"):
            labels.add("python")
        if lower_path.startswith(("src/", "tests/", "tools/", "scripts/", ".github/")):
            labels.add("generated")
    return sorted(labels or {"generated"})


def _labels_for_changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ["generated"]
    return _labels_for_paths(result.stdout.splitlines())


def _accelerator_missing(labels: list[str]) -> str | None:
    missing = _missing_accelerators(labels)
    return missing[0] if missing else None


def _missing_accelerators(labels: list[str]) -> list[str]:
    label_set = set(labels)
    missing: list[str] = []
    if "cuda" in label_set and os.environ.get("NERVE_TEST_CUDA", "").lower() not in TRUTHY:
        missing.append("cuda")
    if "xpu" in label_set and os.environ.get("NERVE_TEST_XPU", "").lower() not in TRUTHY:
        missing.append("xpu")
    return missing


def _available_labels_for_environment(labels: list[str]) -> tuple[list[str], list[str]]:
    missing = set(_missing_accelerators(labels))
    if not missing:
        return labels, []
    return [label for label in labels if label not in missing], sorted(missing)


def _pytest_marker_expression(labels: list[str]) -> str:
    return " or ".join(labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument(
        "--shard-index", type=int, default=int(os.environ.get("NERVE_SHARD_INDEX", "0"))
    )
    parser.add_argument(
        "--shard-count", type=int, default=int(os.environ.get("NERVE_SHARD_COUNT", "1"))
    )
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--ctest", action="store_true")
    parser.add_argument(
        "--retries", type=int, default=int(os.environ.get("NERVE_TEST_RETRIES", "0"))
    )
    parser.add_argument("--changed-from")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--require-hardware", action="store_true")
    args = parser.parse_args()

    if not args.label:
        changed_labels: set[str] = set()
        if args.changed_file:
            changed_labels.update(_labels_for_paths(args.changed_file))
        if args.changed_from:
            changed_labels.update(_labels_for_changed_files(args.changed_from))
        if changed_labels:
            args.label = sorted(changed_labels)

    selected_labels, missing_accelerators = _available_labels_for_environment(args.label)
    if missing_accelerators and args.require_hardware:
        print(
            "test selection requires missing hardware: " + ", ".join(missing_accelerators),
            file=sys.stderr,
        )
        return 1
    if missing_accelerators:
        if not selected_labels:
            print(
                "skipping accelerator-only test selection because the backend is not enabled: "
                + ", ".join(missing_accelerators),
                flush=True,
            )
            return 0
        print(
            "omitting missing accelerator labels: " + ", ".join(missing_accelerators),
            flush=True,
        )
        args.label = selected_labels

    labels = "|".join(args.label)
    commands: list[list[str]] = []
    if args.ctest or not args.python:
        ctest = [
            "ctest",
            "--test-dir",
            args.build_dir,
            "--output-on-failure",
            "--parallel",
            os.environ.get("CTEST_PARALLEL_LEVEL", "2"),
        ]
        if labels:
            ctest.extend(["-L", labels])
        commands.append(ctest)
    if args.python or not args.ctest:
        pytest = [
            sys.executable,
            "-m",
            "pytest",
            "tests/python",
            "-q",
            f"--nerve-shard-index={args.shard_index}",
            f"--nerve-shard-count={args.shard_count}",
        ]
        if args.label:
            pytest.extend(["-m", _pytest_marker_expression(args.label)])
        commands.append(pytest)

    for command in commands:
        attempts = args.retries + 1
        for attempt in range(1, attempts + 1):
            result = _run(command)
            if result == 0:
                break
            if attempt == attempts:
                return result
            print(f"retrying failed command, attempt {attempt + 1}/{attempts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
