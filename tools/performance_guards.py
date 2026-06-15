#!/usr/bin/env python3
"""Validate generated performance-risk coverage without running benchmarks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PERFORMANCE_LABELS = {
    "simplex-explosion",
    "boundary-reduction",
    "dense-distance",
    "cache-locality",
    "sparse-irregular",
    "warp-divergence",
}


@dataclass(frozen=True)
class GuardSummary:
    total_cases: int
    performance_cases: int
    max_simplex_upper_bound: int
    max_boundary_entry_upper_bound: int
    labels: dict[str, int]


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    message: str


def _load_test_matrix() -> ModuleType:
    path = ROOT / "tools" / "test_matrix.py"
    spec = importlib.util.spec_from_file_location("nerve_test_matrix_perf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test matrix module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize() -> GuardSummary:
    matrix = _load_test_matrix()
    cases = list(matrix.iter_cases(include_inactive=True))
    label_counts = {label: 0 for label in REQUIRED_PERFORMANCE_LABELS}
    performance_cases = 0
    max_simplex = 0
    max_boundary = 0
    for case in cases:
        case_labels = set(case.labels)
        if "performance" in case_labels:
            performance_cases += 1
        for label in REQUIRED_PERFORMANCE_LABELS:
            if label in case_labels:
                label_counts[label] += 1
        max_simplex = max(
            max_simplex,
            matrix.simplex_upper_bound(case.shape[0], case.topological_dimension),
        )
        max_boundary = max(
            max_boundary,
            matrix.boundary_entry_upper_bound(case.shape[0], case.topological_dimension),
        )
    return GuardSummary(
        total_cases=len(cases),
        performance_cases=performance_cases,
        max_simplex_upper_bound=max_simplex,
        max_boundary_entry_upper_bound=max_boundary,
        labels=label_counts,
    )


def check() -> list[Finding]:
    summary = summarize()
    findings: list[Finding] = []
    missing_labels = sorted(label for label, count in summary.labels.items() if count == 0)
    if missing_labels:
        findings.append(
            Finding(
                "performance-guards",
                "tools/test_matrix.py",
                f"missing generated performance labels: {missing_labels}",
            )
        )
    if summary.performance_cases < 10_000:
        findings.append(
            Finding(
                "performance-guards",
                "tools/test_matrix.py",
                f"too few generated performance-risk cases: {summary.performance_cases}",
            )
        )
    if summary.max_simplex_upper_bound < 10_000_000:
        findings.append(
            Finding(
                "performance-guards",
                "tools/test_matrix.py",
                f"simplex explosion coverage too small: {summary.max_simplex_upper_bound}",
            )
        )
    if summary.max_boundary_entry_upper_bound < 100_000_000:
        findings.append(
            Finding(
                "performance-guards",
                "tools/test_matrix.py",
                f"boundary reduction coverage too small: {summary.max_boundary_entry_upper_bound}",
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    findings = check()
    if args.json:
        payload = {
            "summary": asdict(summarize()),
            "findings": [asdict(finding) for finding in findings],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for finding in findings:
            print(f"{finding.check}: {finding.path}: {finding.message}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
