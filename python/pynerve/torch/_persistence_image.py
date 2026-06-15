"""Persistence image computation for torch tensors."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from ._diagram import PersistenceDiagram
from ._persistence_validators import (
    _core_backend,
    _torch_backend,
    _validate_image_resolution,
    _validate_persistence_image_diagram,
)
from ._vectorization_basis import _validate_positive_finite  # noqa: PLC0415

_SUPPORTED_IMAGE_WEIGHTS = {"constant", "linear", "persistence"}


def persistence_image(
    diagram: PersistenceDiagram | Tensor,
    resolution: tuple[Any, ...] = (20, 20),
    sigma: float = 0.5,
    weight_fn: str = "persistence",
) -> Tensor:
    """Convert persistence diagram(s) to persistence image(s).

    .. note::
        The canonical NumPy implementation is in :func:`pynerve._image_utils.persistence_image`.
        This torch variant adds GPU support and operates on :class:`PersistenceDiagram`
        or :class:`Tensor` directly. Parameter ``weight_fn`` corresponds to ``weight``
        in the NumPy variant (``"persistence"`` ~ ``"persistence"``,
        ``"linear"`` ~ ``"linear"``, ``"constant"`` ~ ``"uniform"``).
    """
    if weight_fn not in _SUPPORTED_IMAGE_WEIGHTS:
        supported = ", ".join(sorted(_SUPPORTED_IMAGE_WEIGHTS))
        raise ValueError(f"Unsupported weight_fn {weight_fn!r}; expected one of {supported}")
    resolution = _validate_image_resolution(resolution)
    sigma = _validate_positive_finite(sigma, "sigma")

    if isinstance(diagram, PersistenceDiagram):
        d = diagram.diagrams[..., :2]
        if d.dim() == 3:
            return torch.stack(
                [
                    _compute_single_persistence_image(d[b], resolution, sigma, weight_fn)
                    for b in range(d.shape[0])
                ],
                dim=0,
            )
        return _compute_single_persistence_image(d, resolution, sigma, weight_fn)
    if diagram.dim() == 3:
        return torch.stack(
            [
                _compute_single_persistence_image(diagram[b], resolution, sigma, weight_fn)
                for b in range(diagram.shape[0])
            ],
            dim=0,
        )
    return _compute_single_persistence_image(diagram, resolution, sigma, weight_fn)


def _compute_single_persistence_image(
    diagram: Tensor, resolution: tuple[Any, ...], sigma: float, weight_fn: str
) -> Tensor:
    diagram = _validate_persistence_image_diagram(diagram)
    torch_c = _torch_backend()
    if torch_c is not None:
        return cast(
            Tensor,
            torch_c.ml_persistence_image(
                diagram,
                resolution[1],
                resolution[0],
                sigma,
                0.0,
                0.0,
                0.0,
                0.0,
                weight_fn,
            ),
        )

    core_c = _core_backend()
    if (
        core_c is not None
        and weight_fn in {"linear", "persistence"}
        and hasattr(torch.ops.pynerve, "ph_image")
    ):
        return cast(
            Tensor, torch.ops.pynerve.ph_image(diagram, resolution[0], resolution[1], sigma)
        )

    img = torch.zeros(resolution, device=diagram.device, dtype=diagram.dtype)
    if diagram.numel() == 0:
        return img

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    finite = torch.isfinite(deaths)
    births = births[finite]
    deaths = deaths[finite]
    if len(births) == 0:
        return img
    if weight_fn == "constant":
        weights = torch.ones_like(births)
    else:
        weights = torch.clamp(deaths - births, min=0)

    min_birth = births.min()
    max_birth = births.max()
    min_death = deaths.min()
    max_death = deaths.max()

    margin = 0.1
    min_birth -= margin
    max_birth += margin
    min_death -= margin
    max_death += margin

    for i in range(len(births)):
        bx = int((births[i] - min_birth) / (max_birth - min_birth) * (resolution[1] - 1))
        dy = int((deaths[i] - min_death) / (max_death - min_death) * (resolution[0] - 1))
        bx = max(0, min(bx, resolution[1] - 1))
        dy = max(0, min(dy, resolution[0] - 1))
        img[resolution[0] - 1 - dy, bx] += weights[i]
    return img
