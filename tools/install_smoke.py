#!/usr/bin/env python3
"""Smoke-test an installed Pynerve Python prefix."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _prepend_install_prefix(prefix: Path) -> None:
    pynerve_init = prefix / "pynerve" / "__init__.py"
    python_build = prefix / "python"
    if not pynerve_init.exists() and python_build.exists():
        sys.path[:0] = [str(python_build)]
        return
    if not pynerve_init.exists():
        raise FileNotFoundError(f"missing installed pynerve package under {prefix}")
    sys.path[:0] = [str(prefix)]


def _check_core_install() -> None:
    import algorithms_bindings  # type: ignore[import-not-found]
    import pynerve_internal as nerve

    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    options = nerve.PersistenceOptions()
    options.max_dim = 1
    options.max_radius = 2.0
    result = nerve.compute_persistence(points, options)
    for attr in ("pairs", "betti_numbers", "diagnostics"):
        if attr not in result:
            raise RuntimeError(f"unexpected installed persistence result: missing {attr}")
    if not hasattr(nerve, "PH5PH6Engine") or hasattr(nerve, "PH5PH6Prototype"):
        raise RuntimeError("installed public PH5/PH6 engine API must not expose prototype naming")
    _ = nerve.PH5PH6Engine(nerve.PH5PH6Config())

    distances = algorithms_bindings.pairwise_distances(points.astype(np.float32), 3, 2)
    if distances.shape != (3, 3) or not np.isfinite(distances).all():
        raise RuntimeError(f"unexpected installed distance result shape: {distances.shape}")


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
    op_distances = torch.ops.pynerve.filtration_distance_matrix(points[0], "euclidean")
    if op_distances.shape != (3, 3) or op_distances.dtype != torch.float64:
        raise RuntimeError(f"unexpected installed registered distance operator: {op_distances}")
    op_image = torch.ops.pynerve.ph_image(diagram.diagrams[0, :, :2], 4, 4, 0.1)
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
