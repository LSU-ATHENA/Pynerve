#!/usr/bin/env python3
"""Generate deterministic Nerve test cases and CTest shard metadata."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

CPU_BACKENDS = ("cpu",)
ACCEL_BACKENDS = ("cuda", "xpu")
DTYPES = ("float64", "float32", "float16", "bfloat16", "float8_e4m3", "float8_e5m2")
AUTOGRAD_MODES = ("forward", "backward")
SHAPES = ((1, 2), (2, 3), (4, 4), (8, 3), (16, 8), (32, 16))
SEEDS = tuple(range(4))
OPERATORS = (
    "pairwise_distance",
    "vietoris_rips_edges",
    "boundary_matrix",
    "persistence_pairs",
    "diagram_vectorize",
    "wasserstein_distance",
    "bottleneck_distance",
    "mapper_graph",
    "spectral_laplacian",
    "topology_loss",
)


@dataclass(frozen=True)
class Scenario:
    id: str
    topological_dimension: int
    coefficient_field: int
    filtration_order: str
    sparsity: str
    distribution: str


TDA_SCENARIOS = (
    Scenario("h0_dense_gaussian", 0, 2, "lexicographic", "dense", "gaussian"),
    Scenario("h0_sparse_grid", 0, 3, "value_stable", "sparse", "grid"),
    Scenario("h1_circle_sparse", 1, 2, "diameter_stable", "sparse", "circle"),
    Scenario("h1_near_tie", 1, 5, "colexicographic", "clustered", "near_tie"),
    Scenario("h2_sphere_dense", 2, 2, "lexicographic", "dense", "sphere"),
    Scenario("h2_torus_sparse", 2, 3, "value_stable", "sparse", "torus"),
    Scenario("h3_clustered", 3, 5, "colexicographic", "clustered", "mixture"),
    Scenario("h3_adversarial", 3, 2, "diameter_stable", "adversarial", "near_tie"),
    Scenario("h4_sparse_highdim", 4, 3, "lexicographic", "sparse", "gaussian"),
    Scenario("h4_clustered_highdim", 4, 5, "value_stable", "clustered", "mixture"),
    Scenario("h5_dense_highdim", 5, 2, "colexicographic", "dense", "sphere"),
    Scenario("h5_sparse_highdim", 5, 3, "diameter_stable", "sparse", "torus"),
    Scenario("h6_clustered_highdim", 6, 5, "lexicographic", "clustered", "near_tie"),
    Scenario("h6_sparse_highdim", 6, 2, "value_stable", "sparse", "grid"),
    Scenario("h8_sparse_extreme", 8, 3, "colexicographic", "sparse", "gaussian"),
    Scenario("h8_adversarial_extreme", 8, 5, "diameter_stable", "adversarial", "near_tie"),
)


@dataclass(frozen=True)
class Case:
    id: str
    operator: str
    backend: str
    dtype: str
    shape: tuple[int, int]
    autograd: str
    scenario: str
    topological_dimension: int
    coefficient_field: int
    filtration_order: str
    sparsity: str
    distribution: str
    seed: int
    labels: tuple[str, ...]


def _stable_id(parts: Iterable[object]) -> str:
    raw = "::".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _backend_available(backend: str) -> bool:
    if backend == "cpu":
        return True
    if backend == "cuda":
        return os.environ.get("NERVE_TEST_CUDA", "").lower() in {"1", "on", "true", "yes"}
    if backend == "xpu":
        return os.environ.get("NERVE_TEST_XPU", "").lower() in {"1", "on", "true", "yes"}
    return False


def simplex_upper_bound(n_points: int, max_dim: int) -> int:
    max_vertices = min(n_points, max_dim + 1)
    return sum(math.comb(n_points, size) for size in range(1, max_vertices + 1))


def boundary_entry_upper_bound(n_points: int, max_dim: int) -> int:
    max_vertices = min(n_points, max_dim + 1)
    return sum(size * math.comb(n_points, size) for size in range(2, max_vertices + 1))


def _performance_labels(
    operator: str,
    backend: str,
    dtype: str,
    shape: tuple[int, int],
    scenario: Scenario,
) -> tuple[str, ...]:
    n_points, _ = shape
    labels: list[str] = []
    simplex_bound = simplex_upper_bound(n_points, scenario.topological_dimension)
    boundary_bound = boundary_entry_upper_bound(n_points, scenario.topological_dimension)
    if simplex_bound >= 100_000:
        labels.extend(("performance", "simplex-explosion"))
    if operator in {"boundary_matrix", "persistence_pairs"} and boundary_bound >= 100_000:
        labels.extend(("performance", "boundary-reduction"))
    if operator == "pairwise_distance" and scenario.sparsity == "dense":
        labels.extend(("performance", "dense-distance", "cache-locality"))
    if scenario.sparsity in {"sparse", "adversarial"} and operator in {
        "vietoris_rips_edges",
        "boundary_matrix",
        "persistence_pairs",
    }:
        labels.extend(("performance", "sparse-irregular"))
    if backend == "cuda" and (
        scenario.sparsity in {"adversarial", "clustered"} or dtype.startswith("float8")
    ):
        labels.extend(("performance", "warp-divergence"))
    return tuple(dict.fromkeys(labels))


def total_case_count(include_inactive: bool = True) -> int:
    backend_count = len(CPU_BACKENDS)
    if include_inactive:
        backend_count += len(ACCEL_BACKENDS)
    else:
        backend_count += sum(1 for backend in ACCEL_BACKENDS if _backend_available(backend))
    return (
        len(OPERATORS)
        * backend_count
        * len(DTYPES)
        * len(SHAPES)
        * len(AUTOGRAD_MODES)
        * len(TDA_SCENARIOS)
        * len(SEEDS)
    )


def iter_cases(include_inactive: bool = True) -> Iterable[Case]:
    backends = CPU_BACKENDS + ACCEL_BACKENDS
    for op, backend, dtype, shape, autograd, scenario, seed in itertools.product(
        OPERATORS,
        backends,
        DTYPES,
        SHAPES,
        AUTOGRAD_MODES,
        TDA_SCENARIOS,
        SEEDS,
    ):
        if not include_inactive and not _backend_available(backend):
            continue
        labels = [
            "generated",
            "operators",
            "autograd",
            op,
            backend,
            dtype,
            autograd,
            scenario.id,
            f"dim-{scenario.topological_dimension}",
            f"field-{scenario.coefficient_field}",
            f"filtration-{scenario.filtration_order}",
            f"sparsity-{scenario.sparsity}",
            f"distribution-{scenario.distribution}",
        ]
        labels.extend(_performance_labels(op, backend, dtype, shape, scenario))
        if dtype.startswith("float8"):
            labels.append("float8")
        if backend in {"cuda", "xpu"}:
            labels.append("accelerator")
        if autograd == "backward":
            labels.append("gradient")
        yield Case(
            id=_stable_id((op, backend, dtype, shape, autograd, scenario.id, seed)),
            operator=op,
            backend=backend,
            dtype=dtype,
            shape=shape,
            autograd=autograd,
            scenario=scenario.id,
            topological_dimension=scenario.topological_dimension,
            coefficient_field=scenario.coefficient_field,
            filtration_order=scenario.filtration_order,
            sparsity=scenario.sparsity,
            distribution=scenario.distribution,
            seed=seed,
            labels=tuple(labels),
        )


def select_cases(cases: Iterable[Case], shard_index: int, shard_count: int) -> list[Case]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    selected: list[Case] = []
    for case in cases:
        bucket = int(case.id[:8], 16) % shard_count
        if bucket == shard_index:
            selected.append(case)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--dtype", action="append", default=[])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = select_cases(
        iter_cases(include_inactive=not args.available_only),
        args.shard_index,
        args.shard_count,
    )
    for label in args.label:
        cases = [case for case in cases if label in case.labels]
    for operator in args.operator:
        cases = [case for case in cases if case.operator == operator]
    for backend in args.backend:
        cases = [case for case in cases if case.backend == backend]
    for dtype in args.dtype:
        cases = [case for case in cases if case.dtype == dtype]
    for scenario in args.scenario:
        cases = [case for case in cases if case.scenario == scenario]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("limit must be non-negative")
        cases = cases[: args.limit]
    payload = [asdict(case) for case in cases]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
