"""Grouped repository quality checks."""

from __future__ import annotations

from .binding_contracts import (
    check_algorithm_bindings_schema,
    check_binding_smoke_contract,
    check_operator_schema,
    check_pybind_schema,
    check_torch_bindings_schema,
)
from .build_contracts import (
    check_build_install_contract,
    check_ci_contract,
    check_ctest_contract,
    check_performance_guard_contract,
    check_static_analysis_contract,
    check_test_matrix_contract,
)
from .import_api import check_import_graph, check_public_api
from .manifest import generate_manifest
from .static_text import check_static_text

CHECKS = {
    "imports": check_import_graph,
    "api": check_public_api,
    "operators": check_operator_schema,
    "pybind": check_pybind_schema,
    "algorithm-bindings": check_algorithm_bindings_schema,
    "torch-bindings": check_torch_bindings_schema,
    "binding-smoke": check_binding_smoke_contract,
    "static": check_static_text,
    "matrix": check_test_matrix_contract,
    "ctest": check_ctest_contract,
    "build-install": check_build_install_contract,
    "static-analysis": check_static_analysis_contract,
    "performance-guards": check_performance_guard_contract,
    "ci": check_ci_contract,
}

CHECK_GROUPS = {
    "import-api": ("imports", "api"),
    "build-contracts": (
        "matrix",
        "ctest",
        "build-install",
        "static-analysis",
        "performance-guards",
        "ci",
    ),
    "binding-contracts": (
        "operators",
        "pybind",
        "algorithm-bindings",
        "torch-bindings",
        "binding-smoke",
    ),
    "static-text": ("static",),
}

__all__ = ["CHECKS", "CHECK_GROUPS", "generate_manifest"]
