"""Public vectorization entrypoints for persistence diagrams."""

from __future__ import annotations

from typing import Any, Literal

from torch import Tensor

from ._vectorization_basis import (
    adaptive_persistence_image,
    persistence_image,
    persistence_landscape,
    persistence_silhouette,
)
from ._vectorization_spectral import (
    birth_death_curve,
    heat_kernel_signature,
)


def diagram_to_vector(
    diagram: Tensor,
    method: Literal["image", "landscape", "silhouette", "heat", "histogram"] = "landscape",
    **kwargs: Any,
) -> Tensor:
    """Vectorize a diagram using the selected representation."""
    if method == "image":
        return persistence_image(diagram, **kwargs)
    if method == "landscape":
        return persistence_landscape(diagram, **kwargs)
    if method == "silhouette":
        return persistence_silhouette(diagram, **kwargs)
    if method == "heat":
        return heat_kernel_signature(diagram, **kwargs)
    if method == "histogram":
        return birth_death_curve(diagram, **kwargs)
    raise ValueError(f"Unknown method: {method}")


__all__ = [
    "persistence_image",
    "adaptive_persistence_image",
    "persistence_landscape",
    "persistence_silhouette",
    "heat_kernel_signature",
    "birth_death_curve",
    "diagram_to_vector",
]
