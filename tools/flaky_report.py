#!/usr/bin/env python3
"""Flaky test report generator for pynerve CI.

Reads the JSON report produced by pytest --flaky-report and generates
a human-readable summary with statistics and suggested actions.

Usage:
    # Generate flaky test report from pytest output:
    python tools/flaky_report.py build/flaky-report.json

    # Check if flaky rate exceeds threshold (default 5%):
    python tools/flaky_report.py --fail-above 5 build/flaky-report.json

Exit codes:
    0 — flaky rate within threshold
    1 — flaky rate exceeds threshold or report not found
    2 — CI flag set and failures detected (soft-fail for awareness)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _flaky_rate(report: dict) -> float:
    """Percentage of flaky-marked tests that failed."""
    total = report.get("total_flaky_tests", 0)
    failures = report.get("failures", 0)
    if total == 0:
        return 0.0
    return round(failures / total * 100, 1)


def print_report(report_path: str) -> dict:
    """Print human-readable flaky test summary. Returns parsed report dict."""
    path = Path(report_path)
    if not path.exists():
        print(f"Report file not found: {path}", file=sys.stderr)
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    total = data.get("total_flaky_tests", 0)
    failures = data.get("failures", 0)
    rate = _flaky_rate(data)

    print("Flaky Test Report")
    print(f"  File:            {path}")
    print(f"  Total tracked:   {total}")
    print(f"  Failures:        {failures}")
    print(f"  Flaky rate:      {rate}%")
    print(f"  Pass rate:       {data.get('pass_rate', 100.0)}%")

    results = data.get("results", [])
    failed = [r for r in results if r.get("outcome") == "failed"]

    if failed:
        print(f"\n  Failed tests ({len(failed)}):")
        for r in failed:
            name = r.get("name", r.get("nodeid", "unknown"))
            duration = r.get("duration", 0)
            msg = r.get("message", "")[:120]
            print(f"    - {name} ({duration:.2f}s)")
            if msg:
                print(f"      {msg}")

    passed = len(results) - len(failed)
    if passed > 0:
        print(f"\n  Passed tests ({passed}):")
        for r in results:
            if r.get("outcome") == "passed":
                name = r.get("name", r.get("nodeid", "unknown"))
                duration = r.get("duration", 0)
                print(f"    - {name} ({duration:.2f}s)")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flaky test report generator for pynerve CI"
    )
    parser.add_argument("report_path", help="Path to pytest --flaky-report JSON file")
    parser.add_argument(
        "--fail-above",
        type=float,
        default=float(os.environ.get("NERVE_FLAKY_THRESHOLD_PCT", "5.0")),
        help="Exit with code 1 if flaky rate exceeds this percentage (default: 5.0)",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        default=os.environ.get("CI", "").lower() in ("1", "true", "yes"),
        help="CI mode: exits with code 2 on any failures (soft-fail for awareness)",
    )
    args = parser.parse_args()

    data = print_report(args.report_path)
    if not data:
        return 1

    rate = _flaky_rate(data)
    failures = data.get("failures", 0)

    if args.ci_mode and failures > 0:
        print(
            f"\nCI mode: {failures} flaky test failures detected. "
            "This is a soft-fail for awareness — "
            "the CI pipeline should not be blocked by pre-existing flaky tests.",
            file=sys.stderr,
        )
        return 2

    if rate > args.fail_above:
        print(
            f"\nFlaky rate {rate}% exceeds threshold {args.fail_above}%",
            file=sys.stderr,
        )
        return 1

    print(f"\nFlaky rate {rate}% within threshold {args.fail_above}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
