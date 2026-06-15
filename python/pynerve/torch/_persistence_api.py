"""High-level persistence API primitives used by pynerve.torch."""

from __future__ import annotations

import torch

from ..exceptions import InvalidArgumentError
from ._diagram import PersistenceDiagram
from ._persistence_helpers import (
    _diagram_from_backend_parts,
    _diagram_from_distance_matrix_python,
)
from ._persistence_image import (
    _compute_single_persistence_image,
    persistence_image,
)
from ._persistence_validators import (
    _core_backend,
    _count_pairs_by_dimension,
    _torch_backend,
    _valid_vr_output_mask,
    _validate_image_resolution,
    _validate_max_dim,
    _validate_max_radius,
    _validate_metric,
    _validate_persistence_image_diagram,
)
from ._persistence_vr import (
    _compute_zerod_grad_analytical,
    _VRPersistenceFunction,
    vr_persistence,
)

__all__ = [
    "vr_persistence",
    "witness_persistence",
    "alpha_persistence",
    "persistence_from_matrix",
    "persistence_image",
    "_VRPersistenceFunction",
    "_validate_metric",
    "_validate_max_dim",
    "_validate_max_radius",
    "_torch_backend",
    "_core_backend",
    "_valid_vr_output_mask",
    "_count_pairs_by_dimension",
    "_compute_zerod_grad_analytical",
    "_diagram_from_distance_matrix_python",
    "_diagram_from_backend_parts",
    "_compute_single_persistence_image",
    "_validate_image_resolution",
    "_validate_persistence_image_diagram",
]


def witness_persistence(
    landmarks: torch.Tensor,
    witnesses: torch.Tensor,
    max_dim: int = 1,
    max_radius: float = float("inf"),
) -> PersistenceDiagram:
    """Compute weak-witness persistence over landmark points."""
    max_dim = _validate_max_dim(max_dim)
    max_radius = _validate_max_radius(max_radius)
    single_input = landmarks.dim() == 2
    if single_input:
        landmarks = landmarks.unsqueeze(0)
        witnesses = witnesses.unsqueeze(0)
    if landmarks.dim() != 3 or witnesses.dim() != 3:
        raise InvalidArgumentError(
            "landmarks and witnesses must be 2D or 3D tensors",
            parameter="landmarks/witnesses",
        )
    if landmarks.shape[0] != witnesses.shape[0]:
        raise InvalidArgumentError(
            "landmarks and witnesses must have the same batch size",
            parameter="landmarks/witnesses",
        )
    if landmarks.shape[-1] != witnesses.shape[-1]:
        raise InvalidArgumentError(
            "landmarks and witnesses must have the same coordinate dimension",
            parameter="landmarks/witnesses",
        )

    if not torch.is_floating_point(landmarks) or not torch.is_floating_point(witnesses):
        raise InvalidArgumentError(
            "landmarks and witnesses must use floating-point dtypes",
            parameter="landmarks/witnesses",
        )
    if not torch.isfinite(landmarks).all().item() or not torch.isfinite(witnesses).all().item():
        raise InvalidArgumentError(
            "landmarks and witnesses must contain only finite coordinates",
            parameter="landmarks/witnesses",
        )

    distances = torch.cdist(witnesses.to(dtype=torch.float64), landmarks.to(dtype=torch.float64))
    batch_size, _, n_landmarks = distances.shape
    matrix = torch.full(
        (batch_size, n_landmarks, n_landmarks),
        float("inf"),
        dtype=torch.float64,
        device=landmarks.device,
    )
    diagonal = torch.arange(n_landmarks, device=landmarks.device)
    matrix[:, diagonal, diagonal] = 0.0
    for i in range(n_landmarks):
        for j in range(i + 1, n_landmarks):
            best = torch.minimum(
                torch.maximum(distances[:, :, i], distances[:, :, j]).min(dim=1).values,
                torch.tensor(float("inf"), dtype=torch.float64, device=landmarks.device),
            )
            best = torch.where(best <= max_radius, best, torch.full_like(best, float("inf")))
            matrix[:, i, j] = best
            matrix[:, j, i] = best

    matrix = matrix.to(dtype=landmarks.dtype)
    return persistence_from_matrix(matrix.squeeze(0) if single_input else matrix, max_dim)


def alpha_persistence(points: torch.Tensor, max_dim: int = 1) -> PersistenceDiagram:
    """Compute alpha-radius persistence for the point-cloud 1-skeleton."""
    max_dim = _validate_max_dim(max_dim)
    single_input = points.dim() == 2
    if single_input:
        points = points.unsqueeze(0)
    if points.dim() != 3:
        raise InvalidArgumentError(
            f"Expected 2D or 3D input, got {points.dim()}D",
            parameter="points",
        )
    if not torch.is_floating_point(points):
        raise InvalidArgumentError(
            "points must use a floating-point dtype",
            parameter="points",
        )
    if not torch.isfinite(points).all().item():
        raise InvalidArgumentError(
            "points must contain only finite coordinates",
            parameter="points",
        )

    distance_matrix = torch.cdist(points, points) * 0.5
    return persistence_from_matrix(
        distance_matrix.squeeze(0) if single_input else distance_matrix, max_dim
    )


def persistence_from_matrix(distance_matrix: torch.Tensor, max_dim: int = 1) -> PersistenceDiagram:
    """Compute persistence from precomputed distance matrices."""
    max_dim = _validate_max_dim(max_dim)
    single_input = distance_matrix.dim() == 2
    if single_input:
        distance_matrix = distance_matrix.unsqueeze(0)
    if distance_matrix.dim() != 3:
        raise InvalidArgumentError(
            f"Expected 2D or 3D distance matrix, got {distance_matrix.dim()}D",
            parameter="distance_matrix",
        )
    if distance_matrix.shape[-1] != distance_matrix.shape[-2]:
        raise InvalidArgumentError(
            "distance_matrix must be square on the last two dimensions",
            parameter="distance_matrix",
        )

    torch_c = _torch_backend()
    if torch_c is not None:
        diagrams_list = []
        masks_list = []
        num_pairs_list = []
        for b in range(distance_matrix.shape[0]):
            diagram, mask, num_pairs = torch_c.persistence_from_matrix(distance_matrix[b], max_dim)
            diagrams_list.append(diagram)
            masks_list.append(mask)
            num_pairs_list.append(num_pairs)

        return _diagram_from_backend_parts(
            diagrams_list,
            masks_list,
            num_pairs_list,
            batched=not single_input,
        )

    return _diagram_from_distance_matrix_python(distance_matrix, max_dim, single_input=single_input)
