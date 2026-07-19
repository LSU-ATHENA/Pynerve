#!/usr/bin/env python3
"""GPU benchmark regression detection for pynerve CI.

Compares benchmark results against a stored baseline and flags
performance regressions exceeding the configured threshold (default 5%).

Usage:
    # Record a new baseline (run once on a known-good commit):
    python tools/benchmark_regression.py --record --build-dir build

    # Check for regressions in CI (compares against baseline):
    python tools/benchmark_regression.py --check --build-dir build

    # Generate a human-readable report:
    python tools/benchmark_regression.py --report benchmark-report.json

Baseline is stored in benchmarks/baseline.json.
Results are written to benchmarks/benchmark-results.json.

The script runs ctest with the "benchmark" label, extracts timing from
CTest's Test.xml output, and compares wall-clock times.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "benchmarks" / "baseline.json"
RESULTS_PATH = ROOT / "benchmarks" / "benchmark-results.json"
REGRESSION_THRESHOLD_PCT = float(
    os.environ.get("NERVE_BENCHMARK_REGRESSION_THRESHOLD", "5.0")
)


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, **kwargs)


def _parse_ctest_timing(build_dir: str) -> dict[str, float]:
    """Extract test name → wall-clock time (seconds) from CTest Test.xml."""
    tag_file = Path(build_dir) / "Testing" / "TAG"
    if not tag_file.exists():
        print(f"TAG file not found at {tag_file}; running ctest first...", flush=True)
        result = _run(
            [
                "ctest",
                "--test-dir",
                build_dir,
                "-L",
                "benchmark",
                "--output-on-failure",
            ]
        )
        if result.returncode != 0:
            print("ctest benchmark run failed", file=sys.stderr)
            return {}

    # Find the most recent Test.xml
    tag_content = tag_file.read_text(encoding="utf-8").strip()
    test_xml = Path(build_dir) / "Testing" / tag_content / "Test.xml"
    if not test_xml.exists():
        print(f"Test.xml not found at {test_xml}", file=sys.stderr)
        return {}

    tree = ET.parse(str(test_xml))
    root = tree.getroot()
    timing: dict[str, float] = {}
    for test_el in root.findall(".//Test"):
        name = test_el.findtext("Name", "")
        status = test_el.get("Status", "")
        if status == "passed":
            named_measurements = test_el.find("Results/NamedMeasurement")
            if named_measurements is not None:
                value = named_measurements.findtext("Value", "0")
                try:
                    timing[name] = float(value)
                except ValueError:
                    pass
            else:
                # Fall back to Execution Time from the test element
                exec_time = test_el.findtext(
                    "Results/Measurement/Value", "0"
                )
                try:
                    timing[name] = float(exec_time)
                except ValueError:
                    timing[name] = 0.0

    return timing


def record_baseline(build_dir: str) -> int:
    """Run benchmarks and record results as the new baseline."""
    timing = _parse_ctest_timing(build_dir)
    if not timing:
        print("No benchmark timing data found", file=sys.stderr)
        return 1

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = {
        "version": 1,
        "threshold_pct": REGRESSION_THRESHOLD_PCT,
        "git_sha": _run(
            ["git", "rev-parse", "HEAD"], check=False
        ).stdout.strip()[:12],
        "tests": timing,
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Baseline recorded: {len(timing)} benchmarks → {BASELINE_PATH}")
    return 0


def check_regressions(build_dir: str) -> int:
    """Compare current benchmark results against baseline, flag regressions."""
    timing = _parse_ctest_timing(build_dir)
    if not timing:
        print("No benchmark timing data found", file=sys.stderr)
        return 1

    if not BASELINE_PATH.exists():
        print(
            f"No baseline found at {BASELINE_PATH}. "
            "Run with --record to create one.",
            file=sys.stderr,
        )
        return 1

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_tests: dict[str, float] = baseline.get("tests", {})
    threshold_pct = baseline.get("threshold_pct", REGRESSION_THRESHOLD_PCT)

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    new_tests: list[str] = []
    missing_tests: list[str] = []

    for name, current_time in timing.items():
        if name not in baseline_tests:
            new_tests.append(name)
            continue

        baseline_time = baseline_tests[name]
        if baseline_time <= 0:
            continue

        ratio = current_time / baseline_time
        pct_change = round((ratio - 1.0) * 100, 2)

        entry = {
            "name": name,
            "baseline_s": round(baseline_time, 4),
            "current_s": round(current_time, 4),
            "pct_change": pct_change,
        }

        if pct_change > threshold_pct:
            regressions.append(entry)
        elif pct_change < -threshold_pct:
            improvements.append(entry)

    for name in baseline_tests:
        if name not in timing:
            missing_tests.append(name)

    # Write results
    report = {
        "version": 1,
        "git_sha": _run(
            ["git", "rev-parse", "HEAD"], check=False
        ).stdout.strip()[:12],
        "baseline_sha": baseline.get("git_sha", "unknown"),
        "threshold_pct": threshold_pct,
        "summary": {
            "total": len(timing),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "new_tests": len(new_tests),
            "missing_tests": len(missing_tests),
        },
        "regressions": regressions,
        "improvements": improvements,
        "new_tests": new_tests,
        "missing_tests": missing_tests,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n=== Benchmark Regression Report ===")
    print(f"Tests:          {len(timing)}")
    print(f"Regressions:    {len(regressions)}  (>{threshold_pct}% slower)")
    print(f"Improvements:   {len(improvements)}  (>{threshold_pct}% faster)")
    print(f"New:            {len(new_tests)}")
    print(f"Missing:        {len(missing_tests)}")

    if regressions:
        print(f"\nREGRESSIONS (> {threshold_pct}%):")
        for r in sorted(regressions, key=lambda x: x["pct_change"], reverse=True):
            print(
                f"  {r['name']}: "
                f"{r['baseline_s']:.3f}s → {r['current_s']:.3f}s "
                f"(+{r['pct_change']:.1f}%)"
            )

    if improvements:
        print(f"\nIMPROVEMENTS (> {threshold_pct}%):")
        for imp in sorted(improvements, key=lambda x: x["pct_change"]):
            print(
                f"  {imp['name']}: "
                f"{imp['baseline_s']:.3f}s → {imp['current_s']:.3f}s "
                f"({imp['pct_change']:.1f}%)"
            )

    return 1 if regressions else 0


def generate_report(report_path: str) -> int:
    """Print a human-readable summary from a benchmark results JSON file."""
    path = Path(report_path)
    if not path.exists():
        print(f"Report file not found: {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    print(f"Benchmark Report ({data.get('git_sha', 'unknown')[:8]})")
    print(f"  Baseline:  {data.get('baseline_sha', 'unknown')[:8]}")
    print(f"  Threshold: {data.get('threshold_pct', 5.0)}%")
    print(f"  Tests:     {summary.get('total', 0)}")
    print(f"  Regressions:  {summary.get('regressions', 0)}")
    print(f"  Improvements: {summary.get('improvements', 0)}")

    regressions = data.get("regressions", [])
    if regressions:
        print("\nRegressions:")
        for r in sorted(regressions, key=lambda x: float(x["pct_change"]), reverse=True):
            print(
                f"  {r['name']}: "
                f"{r['baseline_s']:.3f}s → {r['current_s']:.3f}s "
                f"(+{r['pct_change']}%)"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPU benchmark regression detection for pynerve CI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--record",
        action="store_true",
        help="Run benchmarks and record as new baseline",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Run benchmarks and check for regressions against baseline",
    )
    group.add_argument(
        "--report",
        type=str,
        metavar="FILE",
        help="Generate human-readable summary from a benchmark results JSON file",
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="CMake build directory (default: build)",
    )
    args = parser.parse_args()

    if args.report:
        return generate_report(args.report)
    if args.record:
        return record_baseline(args.build_dir)
    return check_regressions(args.build_dir)


if __name__ == "__main__":
    raise SystemExit(main())
