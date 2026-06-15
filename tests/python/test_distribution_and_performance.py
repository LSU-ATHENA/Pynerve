from __future__ import annotations

import math
import os
import sys

import pytest

TRUTHY = {"1", "on", "true", "yes"}


@pytest.mark.distributed
def test_mpi_is_optional_for_base_install() -> None:
    assert os.environ.get("NERVE_REQUIRE_MPI", "0") in {"0", "false", "False", ""}, (
        f"NERVE_REQUIRE_MPI should not be truthy for base install, got {os.environ.get('NERVE_REQUIRE_MPI', '0')}"
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("xpu", marks=pytest.mark.xpu),
    ],
)
def test_multi_device_labels_are_selectable(generated_cases, backend: str) -> None:
    selected = [case for case in generated_cases if case.backend == backend]
    enabled = (
        backend == "cpu" or os.environ.get(f"NERVE_TEST_{backend.upper()}", "").lower() in TRUTHY
    )
    if enabled:
        assert selected, f"expected selected cases for backend {backend}"
    else:
        assert not selected, f"expected no selected cases for backend {backend}"


@pytest.mark.performance
def test_generated_case_loop_has_linear_sanity_guard(generated_cases) -> None:
    total = 0.0
    for case in generated_cases[:4096]:
        rows, cols = case.shape
        total += math.sqrt(rows * cols + case.seed)
    assert total > 0.0, "computed total should be positive"


@pytest.mark.performance
def test_generated_cases_expose_performance_risk_labels() -> None:
    matrix = sys.modules["nerve_test_matrix"]
    labels = {label for case in matrix.iter_cases(include_inactive=True) for label in case.labels}
    assert {
        "simplex-explosion",
        "boundary-reduction",
        "dense-distance",
        "cache-locality",
        "sparse-irregular",
        "warp-divergence",
    }.issubset(labels), "expected all performance risk labels to be present in the test matrix"
