#!/usr/bin/env python3
"""Smoke-test built Python extension modules against the source Python package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _expect_validation(name: str, call, fragment: str) -> None:
    try:
        call()
    except Exception as exc:
        if fragment not in str(exc):
            raise RuntimeError(f"{name} validation error lost expected context: {exc}") from exc
    else:
        raise RuntimeError(f"{name} accepted invalid input")


def _prepend_import_paths(build_dir: Path) -> None:
    build_python = build_dir / "python"
    if not build_python.exists():
        raise FileNotFoundError(f"missing Python build directory: {build_python}")
    sys.path[:0] = [str(build_python)]


def _check_core_api() -> None:
    import pynerve_internal as nerve

    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    required_keys = {"pairs", "betti_numbers", "diagnostics"}

    options = nerve.PersistenceOptions()
    options.max_dim = 1
    options.max_radius = 2.0

    for operation in (
        nerve.compute_persistence,
        nerve.compute_persistence_ph4,
        nerve.compute_persistence_ph5,
        nerve.compute_persistence_ph6,
        nerve.compute_persistence_cohomology,
    ):
        result = operation(points, options)
        if not required_keys.issubset(result):
            raise RuntimeError(f"{operation.__name__} returned unexpected keys: {sorted(result)}")
        if not result["pairs"]:
            raise RuntimeError(f"{operation.__name__} returned no persistence pairs")

    if not hasattr(nerve, "PH5PH6Engine") or hasattr(nerve, "PH5PH6Prototype"):
        raise RuntimeError("public PH5/PH6 engine API must not expose prototype naming")
    _ = nerve.PH5PH6Engine(nerve.PH5PH6Config())

    update_opts = nerve.PersistenceOptions()
    updated = nerve.update_persistence([("add", [0]), ("add", [1]), ("add", [0, 1])], update_opts)
    if not required_keys.issubset(updated):
        raise RuntimeError(f"unexpected incremental persistence result keys: {sorted(updated)}")

    _expect_validation(
        "core public nonfinite point",
        lambda: nerve.compute_persistence(
            np.asarray([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float64),
            options,
        ),
        "NaN or infinite",
    )
    invalid_options = nerve.PersistenceOptions()
    invalid_options.max_dim = 1
    invalid_options.max_radius = float("nan")
    _expect_validation(
        "core invalid max_radius override",
        lambda: nerve.compute_persistence(points, invalid_options),
        "max_radius",
    )
    invalid_tol_opts = nerve.PersistenceOptions()
    invalid_tol_opts.max_dim = 1
    invalid_tol_opts.max_radius = 2.0
    invalid_tol_opts.error_tolerance = float("inf")
    _expect_validation(
        "core invalid error_tolerance override",
        lambda: nerve.compute_persistence(points, invalid_tol_opts),
        "error_tolerance",
    )
    invalid_radius_opts = nerve.PersistenceOptions()
    invalid_radius_opts.max_radius = float("nan")
    _expect_validation(
        "core invalid cloned max_radius option",
        lambda: nerve.compute_persistence(points, invalid_radius_opts),
        "max_radius",
    )

    direct_options = nerve.PersistenceOptions()
    direct_options.max_dim = 1
    direct_options.max_radius = 2.0
    _expect_validation(
        "core internal nonfinite point",
        lambda: nerve.compute_persistence(
            np.asarray([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float64),
            direct_options,
        ),
        "NaN or infinite",
    )
    bad_direct_options = nerve.PersistenceOptions()
    bad_direct_options.max_radius = float("nan")
    _expect_validation(
        "core internal invalid max_radius option",
        lambda: nerve.compute_persistence(points, bad_direct_options),
        "max_radius",
    )
    bad_direct_options = nerve.PersistenceOptions()
    bad_direct_options.error_tolerance = float("nan")
    _expect_validation(
        "core internal invalid error_tolerance option",
        lambda: nerve.compute_persistence(points, bad_direct_options),
        "error_tolerance",
    )
    _expect_validation(
        "core invalid event simplex",
        lambda: nerve.update_persistence([("add", [])], nerve.PersistenceOptions()),
        "simplex",
    )
    engine = nerve.PH5PH6Engine(nerve.PH5PH6Config())
    _expect_validation(
        "PH5PH6 nonfinite point",
        lambda: engine.compute_persistence_cohomology(
            np.asarray([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float64), 1
        ),
        "NaN or infinite",
    )
    _expect_validation(
        "PH5PH6 zero max_dimension",
        lambda: engine.compute_persistence_cohomology(points, 0),
        "max_dimension",
    )
    _expect_validation(
        "PH5PH6 zero stability runs",
        lambda: engine.run_stability_test(points, 1, 0),
        "num_runs",
    )
    bad_config = nerve.PH5PH6Config()
    bad_config.numerical_tolerance = float("nan")
    _expect_validation(
        "PH5PH6 invalid numerical_tolerance",
        lambda: nerve.PH5PH6Engine(bad_config),
        "numerical_tolerance",
    )

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return
    torch_result = nerve.compute_persistence(
        torch.tensor(points, dtype=torch.float32),
        options,
    )
    if not required_keys.issubset(torch_result):
        raise RuntimeError(
            f"unexpected torch-input persistence result keys: {sorted(torch_result)}"
        )


def _check_algorithm_bindings() -> None:
    import algorithms_bindings  # type: ignore[import-not-found]

    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    distances = algorithms_bindings.pairwise_distances(points, 3, 2)
    if distances.shape != (3, 3):
        raise RuntimeError(f"unexpected pairwise distance shape: {distances.shape}")
    if not np.allclose(distances, distances.T):
        raise RuntimeError("pairwise distance matrix is not symmetric")

    knn_distances, knn_indices = algorithms_bindings.knn(points, 3, 2, 5)
    if knn_distances.shape != (3, 2) or knn_indices.shape != (3, 2):
        raise RuntimeError(
            "KNN output must use the clamped native neighbor count; "
            f"got {knn_distances.shape} and {knn_indices.shape}"
        )
    if not np.isfinite(knn_distances).all():
        raise RuntimeError("KNN distances contain non-finite values")
    _expect_validation(
        "algorithm pairwise nonfinite point",
        lambda: algorithms_bindings.pairwise_distances(
            np.asarray([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float32), 2, 2
        ),
        "finite values",
    )
    _expect_validation(
        "algorithm pairwise mismatched shape",
        lambda: algorithms_bindings.pairwise_distances(points, 4, 2),
        "rows * dim",
    )
    _expect_validation(
        "algorithm KNN zero dimension",
        lambda: algorithms_bindings.knn(points, 3, 0, 2),
        "dimension",
    )

    conv_config = algorithms_bindings.DiagramConvConfigDiagramConv1DF()
    conv = algorithms_bindings.DiagramConv1DF(conv_config)
    conv_out = conv.forward(
        np.asarray([[[0.0, 1.0, 0.0]]], dtype=np.float32),
        np.asarray([[[1.0]]], dtype=np.float32),
        1,
        1,
    )
    if conv_out.shape != (1, 1) or not np.isfinite(conv_out).all():
        raise RuntimeError(f"unexpected diagram convolution output: {conv_out}")
    _expect_validation(
        "algorithm diagram convolution nonfinite features",
        lambda: conv.forward(
            np.asarray([[[0.0, 1.0, 0.0]]], dtype=np.float32),
            np.asarray([[[np.nan]]], dtype=np.float32),
            1,
            1,
        ),
        "features",
    )
    _expect_validation(
        "algorithm diagram convolution zero batch",
        lambda: conv.forward(
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            0,
            0,
        ),
        "batch_size",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()

    _prepend_import_paths(args.build_dir.resolve())

    _check_core_api()
    _check_algorithm_bindings()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
