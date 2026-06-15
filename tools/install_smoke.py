#!/usr/bin/env python3
"""Smoke-test an installed Nerve Python prefix."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _prepend_install_prefix(prefix: Path) -> None:
    if not (prefix / "nerve" / "__init__.py").exists():
        raise FileNotFoundError(f"missing installed nerve package under {prefix}")
    sys.path[:0] = [str(prefix)]


def _check_core_install() -> None:
    import algorithms_bindings  # type: ignore[import-not-found]
    import pynerve_internal as _  # noqa: F811

    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    result = nerve.compute_persistence(points, max_dim=1, max_radius=2.0)
    for attr in ("pairs", "betti_numbers", "diagnostics"):
        if not hasattr(result, attr):
            raise RuntimeError(f"unexpected installed persistence result: missing {attr}")
    if not hasattr(nerve, "PH5PH6Engine") or hasattr(nerve, "PH5PH6Prototype"):
        raise RuntimeError("installed public PH5/PH6 engine API must not expose prototype naming")
    _ = nerve.PH5PH6Engine(nerve.PH5PH6Config())

    distances = algorithms_bindings.pairwise_distances(points.astype(np.float32), 3, 2)
    if distances.shape != (3, 3) or not np.isfinite(distances).all():
        raise RuntimeError(f"unexpected installed distance result shape: {distances.shape}")
    diagram = np.asarray([[0.0, 1.0, 0.0], [0.25, 0.75, 1.0]], dtype=np.float64)
    image = nerve.persistence_image(diagram, resolution=(8, 8), sigma=0.2)
    if image.shape != (8, 8) or not np.isfinite(image).all() or float(image.sum()) <= 0.0:
        raise RuntimeError(f"unexpected installed NumPy persistence image: {image.shape}")


def _check_torch_install(require_torch: bool) -> None:
    torch_spec = importlib.util.find_spec("torch")
    torch_ext_spec = importlib.util.find_spec("nerve_torch_internal")
    if not require_torch and (torch_spec is None or torch_ext_spec is None):
        return
    if torch_spec is None:
        raise RuntimeError("torch install smoke requires torch, but torch is missing")
    if torch_ext_spec is None:
        raise RuntimeError("torch install smoke requires nerve_torch_internal")

    import pynerve.torch as tda_torch  # type: ignore[import-not-found]

    import torch  # type: ignore[import-not-found]

    points = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    diagram = tda_torch.vr_persistence(points, max_dim=1, max_radius=2.0)
    image = tda_torch.persistence_image(diagram, resolution=(8, 8), sigma=0.1)
    if image.shape != (1, 8, 8) or image.dtype != points.dtype:
        raise RuntimeError(
            f"unexpected installed torch persistence image: {image.shape}, {image.dtype}"
        )
    op_distances = torch.ops.nerve.filtration_distance_matrix(points[0], "euclidean")
    if op_distances.shape != (3, 3) or op_distances.dtype != torch.float64:
        raise RuntimeError(f"unexpected installed registered distance operator: {op_distances}")
    op_image = torch.ops.nerve.ph_image(diagram.diagrams[0, :, :2], 4, 4, 0.1)
    if op_image.shape != (4, 4):
        raise RuntimeError(
            f"unexpected installed registered image operator shape: {op_image.shape}"
        )
    if hasattr(torch, "float8_e4m3fn"):
        float8_diagram = torch.tensor(
            [[0.0, 1.0], [0.25, 0.75]],
            dtype=torch.float8_e4m3fn,
        )
        float8_image = tda_torch.persistence_image(float8_diagram, resolution=(4, 4), sigma=0.1)
        if float8_image.shape != (4, 4) or float8_image.dtype != float8_diagram.dtype:
            raise RuntimeError(
                "installed float8 persistence image must preserve dtype; "
                f"got {float8_image.shape}, {float8_image.dtype}"
            )
    loss = torch.nan_to_num(diagram.diagrams[..., :2]).sum()
    loss.backward()
    if points.grad is None or not torch.isfinite(points.grad).all():
        raise RuntimeError("installed torch persistence backward path produced invalid gradients")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()

    _prepend_install_prefix(args.prefix.resolve())
    _check_core_install()
    _check_torch_install(args.require_torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
