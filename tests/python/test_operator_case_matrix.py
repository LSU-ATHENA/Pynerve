from __future__ import annotations

import math
import random

import pytest
import test_matrix


@pytest.mark.generated
def test_matrix_defines_tens_of_thousands_of_cases() -> None:
    cases = list(test_matrix.iter_cases(include_inactive=True))
    assert len(cases) == 92160, f"expected 92160 cases, got {len(cases)}"
    assert test_matrix.total_case_count(include_inactive=True) == len(cases), (
        f"total_case_count returned {test_matrix.total_case_count(include_inactive=True)}, expected {len(cases)}"
    )
    assert len({case.id for case in cases}) == len(cases), (
        f"unique ids ({len({case.id for case in cases})}) != total cases ({len(cases)})"
    )
    assert {"cpu", "cuda"} == {case.backend for case in cases}, (
        f"backends mismatch: expected cpu/cuda, got {sorted({case.backend for case in cases})}"
    )
    assert {"float64", "float32", "float16", "bfloat16", "float8_e4m3", "float8_e5m2"} == {
        case.dtype for case in cases
    }, f"dtypes mismatch: expected 6 float types, got {sorted({case.dtype for case in cases})}"
    assert {0, 1, 2, 3, 4, 5, 6, 8} == {case.topological_dimension for case in cases}, (
        f"topological dimension mismatch: expected 0-6,8, got {sorted({case.topological_dimension for case in cases})}"
    )
    assert {2, 3, 5} == {case.coefficient_field for case in cases}, (
        f"coefficient field mismatch: expected 2,3,5, got {sorted({case.coefficient_field for case in cases})}"
    )
    assert {"dense", "sparse", "clustered", "adversarial"} == {case.sparsity for case in cases}, (
        f"sparsity mismatch: expected dense/sparse/clustered/adversarial, got {sorted({case.sparsity for case in cases})}"
    )
    assert any("float8" in case.labels for case in cases), "no cases have float8 in labels"


@pytest.mark.generated
@pytest.mark.cpu
@pytest.mark.operators
def test_generated_cpu_cases_are_numerically_stable(generated_cases) -> None:
    for case in generated_cases[:2048]:
        rng = random.Random(case.seed)
        rows, cols = case.shape
        values = [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]
        norms = [math.sqrt(sum(value * value for value in row)) for row in values]
        assert all(math.isfinite(norm) for norm in norms), "some norms are non-finite"
        assert all(norm >= 0.0 for norm in norms), "some norms are negative"


@pytest.mark.generated
@pytest.mark.operators
def test_generated_cases_cover_tda_sensitivity_scenarios(generated_cases) -> None:
    assert {case.topological_dimension for case in generated_cases} == {0, 1, 2, 3, 4, 5, 6, 8}, (
        f"topological dimension mismatch: expected 0-6,8, got {sorted({case.topological_dimension for case in generated_cases})}"
    )
    assert {case.coefficient_field for case in generated_cases} == {2, 3, 5}, (
        f"coefficient field mismatch: expected 2,3,5, got {sorted({case.coefficient_field for case in generated_cases})}"
    )
    assert {case.filtration_order for case in generated_cases} == {
        "lexicographic",
        "value_stable",
        "diameter_stable",
        "colexicographic",
    }, (
        f"filtration order mismatch: got {sorted({case.filtration_order for case in generated_cases})}"
    )
    assert {"gaussian", "grid", "circle", "near_tie", "sphere", "torus", "mixture"} == {
        case.distribution for case in generated_cases
    }, f"distribution mismatch: got {sorted({case.distribution for case in generated_cases})}"
    assert all("operators" in case.labels for case in generated_cases[:1024]), (
        "some generated cases lack 'operators' label"
    )


@pytest.mark.generated
@pytest.mark.gradient
@pytest.mark.autograd
def test_generated_finite_difference_gradient_cases(generated_cases) -> None:
    target = min(2048, sum(1 for case in generated_cases if case.autograd == "backward"))
    assert target > 0, "no backward autograd cases found"
    checked = 0
    for case in generated_cases:
        if case.autograd != "backward":
            continue
        x = (case.seed % 23) / 7.0 - 1.0
        eps = 1.0e-4
        forward = (x + eps) * (x + eps)
        backward = (x - eps) * (x - eps)
        numerical_grad = (forward - backward) / (2.0 * eps)
        assert abs(numerical_grad - 2.0 * x) < 1.0e-8, (
            f"numerical gradient {numerical_grad} differs from analytic {2.0 * x} by more than 1e-8"
        )
        checked += 1
        if checked >= target:
            break
    assert checked == target, f"only checked {checked} of {target} backward autograd cases"


@pytest.mark.generated
def test_shard_selection_is_stable_and_disjoint() -> None:
    full = {case.id for case in test_matrix.iter_cases(include_inactive=True)}
    shards = [
        {case.id for case in test_matrix.select_cases(test_matrix.iter_cases(True), index, 8)}
        for index in range(8)
    ]
    assert set.union(*shards) == full, (
        f"shard union ({len(set.union(*shards))}) does not cover full set ({len(full)})"
    )
    for index, shard in enumerate(shards):
        others = set.union(*(other for pos, other in enumerate(shards) if pos != index))
        assert shard.isdisjoint(others), f"shard {index} overlaps with other shards"
