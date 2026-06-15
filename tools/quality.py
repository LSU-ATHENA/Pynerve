# ruff: noqa: F401 -- imports register checks via quality framework side effects
#!/usr/bin/env python3
"""Repository quality check CLI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_missing_checks: list[str] = []

try:
    import ast as _ast

    from quality_checks import (
        CHECK_GROUPS,
        CHECKS,
        generate_manifest,
    )
    from quality_checks import (
        build_contracts as _build_contracts,
    )
    from quality_checks.binding_contracts import (
        _check_torch_operator_schema_text,
        check_algorithm_bindings_schema,
        check_binding_smoke_contract,
        check_operator_schema,
        check_pybind_schema,
        check_torch_bindings_schema,
    )
    from quality_checks.build_contracts import (
        check_build_install_contract,
        check_ci_contract,
        check_ctest_contract,
        check_performance_guard_contract,
        check_static_analysis_contract,
        check_test_matrix_contract,
    )
    from quality_checks.common import (
        ROOT,
        SCRIPTS_ROOT,
        Finding,
        _lazy_submodules,
        _load_tool_module,
    )
    from quality_checks.import_api import (
        _detect_cycles,
        _is_nerve_package_import,
        _resolve_import_from,
        check_import_graph,
        check_public_api,
    )
    from quality_checks.static_text import check_static_text

    ast = _ast
except ImportError as _exc:
    _missing_checks.append(f"quality_checks package: {_exc}")
    CHECK_GROUPS = {}
    CHECKS = {}
    _build_contracts = None

    def generate_manifest(*_a: object, **_kw: object) -> None:
        pass

    Finding = dict
    ROOT = Path.cwd()
    SCRIPTS_ROOT = ROOT
    _load_tool_module = None


def _sync_build_contract_globals() -> None:
    if _build_contracts is not None:
        _build_contracts._load_tool_module = _load_tool_module
        _build_contracts.ROOT = ROOT
        _build_contracts.SCRIPTS_ROOT = SCRIPTS_ROOT


def check_cuda_launch_contract() -> list:
    _sync_build_contract_globals()
    if _build_contracts is not None:
        return _build_contracts.check_cuda_launch_contract()
    return []


def check_script_syntax_contract() -> list:
    _sync_build_contract_globals()
    if _build_contracts is not None:
        return _build_contracts.check_script_syntax_contract()
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        choices=("all", *CHECK_GROUPS.keys(), *CHECKS.keys()),
        default="all",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    if _missing_checks:
        print(f"Warning: {len(_missing_checks)} check modules not available:", file=sys.stderr)
        for msg in _missing_checks:
            print(f"  - {msg}", file=sys.stderr)

    if args.manifest_output:
        generate_manifest(args.manifest_output)

    if not CHECKS:
        print("No quality checks available. Install the quality_checks package.", file=sys.stderr)
        return 1

    if args.check == "all":
        selected = CHECKS
    elif args.check in CHECK_GROUPS:
        selected = {name: CHECKS[name] for name in CHECK_GROUPS[args.check]}
    else:
        selected = {args.check: CHECKS[args.check]}
    findings = [finding for check in selected.values() for finding in check()]

    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    else:
        for finding in findings:
            print(f"{finding.check}: {finding.path}: {finding.message}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
