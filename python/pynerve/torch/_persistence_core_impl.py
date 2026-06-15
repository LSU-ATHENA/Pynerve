"""Backend-dispatched persistence computation core."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from .._utils import (
    ensure_batch_dim,
    validate_dtype,
)
from ..exceptions import (
    DtypeError,
    NerveError,
    PersistenceError,
)
from ._backend import backend
from ._persistence_api import (
    _count_pairs_by_dimension,
    _diagram_from_distance_matrix_python,
    _validate_max_dim,
    _validate_max_radius,
    _validate_metric,
)
from ._persistence_python import compute_vr_python


@dataclass
class PersistenceResult:
    """Standardized result container for persistence computations."""

    diagrams: Tensor  # [batch, max_pairs, 3] or [max_pairs, 3]
    mask: Tensor  # [batch, max_pairs] or [max_pairs]
    num_pairs: Tensor  # [batch, max_dim+1] or [max_dim+1]
    was_batched: bool  # Whether input was originally batched

    def unbatch(self) -> PersistenceResult:
        """Remove batch dimension if input was single."""
        if not self.was_batched:
            return PersistenceResult(
                diagrams=self.diagrams.squeeze(0) if self.diagrams.dim() == 3 else self.diagrams,
                mask=self.mask.squeeze(0) if self.mask.dim() == 2 else self.mask,
                num_pairs=self.num_pairs.squeeze(0)
                if self.num_pairs.dim() == 2
                else self.num_pairs,
                was_batched=False,
            )
        return self


class PersistenceComputer:
    """Abstract base for persistence computation backends."""

    def compute_vr(
        self, points: Tensor, max_dim: int, max_radius: float, metric: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Vietoris-Rips persistence."""
        del points, max_dim, max_radius, metric
        raise RuntimeError("abstract backend method called: compute_vr")

    def compute_witness(
        self, landmarks: Tensor, witnesses: Tensor, max_dim: int, max_radius: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute witness complex persistence."""
        del landmarks, witnesses, max_dim, max_radius
        raise RuntimeError("abstract backend method called: compute_witness")

    def compute_alpha(self, points: Tensor, max_dim: int) -> tuple[Tensor, Tensor, Tensor]:
        """Compute alpha complex persistence."""
        del points, max_dim
        raise RuntimeError("abstract backend method called: compute_alpha")

    def compute_distance_matrix(
        self, distance_matrix: Tensor, max_dim: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute persistence from distance matrix."""
        del distance_matrix, max_dim
        raise RuntimeError("abstract backend method called: compute_distance_matrix")


def _stack_backend_parts(
    diagrams_list: list[Tensor],
    masks_list: list[Tensor],
    num_pairs_list: list[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    if not diagrams_list:
        raise ValueError("backend returned no diagrams")

    max_pairs = max(diagram.shape[0] for diagram in diagrams_list)
    max_pair_dims = max(diagram.shape[1] for diagram in diagrams_list)
    max_num_pair_dims = max(num_pairs.reshape(-1).shape[0] for num_pairs in num_pairs_list)

    padded_diagrams = []
    padded_masks = []
    padded_num_pairs = []
    for diagram, mask, num_pairs in zip(diagrams_list, masks_list, num_pairs_list, strict=False):
        if diagram.shape[0] < max_pairs or diagram.shape[1] < max_pair_dims:
            padded = diagram.new_zeros((max_pairs, max_pair_dims))
            padded[: diagram.shape[0], : diagram.shape[1]] = diagram
            diagram = padded
        if mask.shape[0] < max_pairs:
            padded_mask = mask.new_zeros((max_pairs,))
            padded_mask[: mask.shape[0]] = mask
            mask = padded_mask
        num_pairs = num_pairs.reshape(-1)
        if num_pairs.shape[0] < max_num_pair_dims:
            padded_counts = num_pairs.new_zeros((max_num_pair_dims,))
            padded_counts[: num_pairs.shape[0]] = num_pairs
            num_pairs = padded_counts
        padded_diagrams.append(diagram)
        padded_masks.append(mask)
        padded_num_pairs.append(num_pairs)

    return (
        torch.stack(padded_diagrams, dim=0),
        torch.stack(padded_masks, dim=0),
        torch.stack(padded_num_pairs, dim=0),
    )


def _parts_from_diagram(diagram: Tensor, max_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    mask = torch.ones(diagram.shape[:2], dtype=torch.bool, device=diagram.device)
    num_pairs = _count_pairs_by_dimension(diagram, mask, max_dim)
    return diagram, mask, num_pairs


def _distance_matrix_parts(distance_matrix: Tensor, max_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    diagram = _diagram_from_distance_matrix_python(distance_matrix, max_dim, single_input=False)
    return diagram.diagrams, diagram.mask, cast(Tensor, diagram.num_pairs)


def _witness_distance_matrix(landmarks: Tensor, witnesses: Tensor, max_radius: float) -> Tensor:
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
            best = torch.maximum(distances[:, :, i], distances[:, :, j]).min(dim=1).values
            best = torch.where(best <= max_radius, best, torch.full_like(best, float("inf")))
            matrix[:, i, j] = best
            matrix[:, j, i] = best
    return matrix.to(dtype=landmarks.dtype)


class TorchCBackend(PersistenceComputer):
    """Torch-native C++ backend (fastest, most features)."""

    def __init__(self) -> None:
        self._torch_c = backend.get_torch_c_backend()
        if self._torch_c is None:
            raise RuntimeError("Torch C++ backend not loaded")

    def compute_vr(
        self, points: Tensor, max_dim: int, max_radius: float, metric: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Use torch-native C++ VR computation.

        Note:
            The C++ backend receives two additional fixed parameters:
            - ``min_radius`` (0.0): no minimum radius filter
            - ``max_pair_ratio`` (0.1): maximum ratio of output pairs to
              input points, used to bound memory usage.
        """
        result = self._torch_c.vr_persistence_forward(points, max_dim, max_radius, metric, 0, 0.1)
        return result[0], result[1], result[2]

    def compute_witness(
        self, landmarks: Tensor, witnesses: Tensor, max_dim: int, max_radius: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute weak-witness persistence through a landmark distance matrix."""
        return _distance_matrix_parts(
            _witness_distance_matrix(landmarks, witnesses, max_radius), max_dim
        )

    def compute_alpha(self, points: Tensor, max_dim: int) -> tuple[Tensor, Tensor, Tensor]:
        """Compute alpha-radius persistence for the point-cloud 1-skeleton."""
        return _distance_matrix_parts(torch.cdist(points, points) * 0.5, max_dim)

    def compute_distance_matrix(
        self, distance_matrix: Tensor, max_dim: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Use torch-native C++ distance matrix computation."""
        single_input = distance_matrix.dim() == 2
        if single_input:
            distance_matrix = distance_matrix.unsqueeze(0)
        if distance_matrix.dim() != 3:
            raise ValueError("distance_matrix must be rank-2 or rank-3")

        batch_size = distance_matrix.shape[0]
        diagrams_list = []
        masks_list = []
        num_pairs_list = []

        for b in range(batch_size):
            diagram, mask, num_pairs = self._torch_c.persistence_from_matrix(
                distance_matrix[b], max_dim
            )
            diagrams_list.append(diagram)
            masks_list.append(mask)
            num_pairs_list.append(num_pairs)

        if single_input:
            return diagrams_list[0], masks_list[0], num_pairs_list[0]
        return _stack_backend_parts(diagrams_list, masks_list, num_pairs_list)


class CoreCBackend(PersistenceComputer):
    """Core C++ backend path for ops not exposed in torch bindings."""

    def __init__(self) -> None:
        if not hasattr(torch.ops, "nerve"):
            raise RuntimeError("Core C++ backend not loaded")

    def compute_vr(
        self, points: Tensor, max_dim: int, max_radius: float, metric: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute VR persistence with the tensor implementation."""
        return _parts_from_diagram(
            compute_vr_python(points, max_dim=max_dim, metric=metric, max_radius=max_radius),
            max_dim,
        )

    def compute_witness(
        self, landmarks: Tensor, witnesses: Tensor, max_dim: int, max_radius: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute weak-witness persistence through a landmark distance matrix."""
        return _distance_matrix_parts(
            _witness_distance_matrix(landmarks, witnesses, max_radius), max_dim
        )

    def compute_alpha(self, points: Tensor, max_dim: int) -> tuple[Tensor, Tensor, Tensor]:
        """Compute alpha-radius persistence for the point-cloud 1-skeleton."""
        return _distance_matrix_parts(torch.cdist(points, points) * 0.5, max_dim)

    def compute_distance_matrix(
        self, distance_matrix: Tensor, max_dim: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute persistence from a precomputed distance matrix."""
        return _distance_matrix_parts(distance_matrix, max_dim)


class PythonBackend(PersistenceComputer):
    """Tensor-native Python backend for torch persistence."""

    def compute_vr(
        self, points: Tensor, max_dim: int, max_radius: float, metric: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute VR persistence with the tensor implementation."""
        return _parts_from_diagram(
            compute_vr_python(points, max_dim=max_dim, metric=metric, max_radius=max_radius),
            max_dim,
        )

    def compute_witness(
        self, landmarks: Tensor, witnesses: Tensor, max_dim: int, max_radius: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute weak-witness persistence through a landmark distance matrix."""
        return _distance_matrix_parts(
            _witness_distance_matrix(landmarks, witnesses, max_radius), max_dim
        )

    def compute_alpha(self, points: Tensor, max_dim: int) -> tuple[Tensor, Tensor, Tensor]:
        """Compute alpha-radius persistence for the point-cloud 1-skeleton."""
        return _distance_matrix_parts(torch.cdist(points, points) * 0.5, max_dim)

    def compute_distance_matrix(
        self, distance_matrix: Tensor, max_dim: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute persistence from a precomputed distance matrix."""
        return _distance_matrix_parts(distance_matrix, max_dim)


def get_best_backend(requirement: str | None = None) -> PersistenceComputer:
    """Get the best available backend."""
    if requirement == "torch_c":
        return TorchCBackend()
    elif requirement == "core_c":
        return CoreCBackend()
    elif requirement is None:
        try:
            return TorchCBackend()
        except RuntimeError:
            warnings.warn(
                "Torch C++ backend not available, falling back to core C backend",
                RuntimeWarning,
                stacklevel=2,
            )

        try:
            return CoreCBackend()
        except RuntimeError:
            warnings.warn(
                "Core C++ backend not available, falling back to Python backend. "
                "Install C++ extensions for better performance.",
                RuntimeWarning,
                stacklevel=2,
            )
        return PythonBackend()
    else:
        raise ValueError(f"Unknown backend requirement: {requirement}")


_persistence_backend: PersistenceComputer | None = None


def get_persistence_backend() -> PersistenceComputer:
    """Get the global persistence backend (cached)."""
    global _persistence_backend  # noqa: PLW0603
    if _persistence_backend is None:
        _persistence_backend = get_best_backend()
    return _persistence_backend


def compute_persistence_vr(
    points: Tensor,
    max_dim: int = 1,
    max_radius: float = float("inf"),
    metric: str = "euclidean",
    backend_requirement: str | None = None,
) -> PersistenceResult:
    """Unified VR persistence computation."""
    max_dim = _validate_max_dim(max_dim)
    max_radius = _validate_max_radius(max_radius)
    metric = _validate_metric(metric)
    points, was_batched = ensure_batch_dim(points, 3)

    supported_dtypes = {torch.float32, torch.float64}
    if points.dtype == torch.bfloat16:
        if not points.is_cuda or not torch.cuda.is_bf16_supported():
            raise DtypeError(
                "bfloat16 requires CUDA with bfloat16 support",
                actual_dtype=str(points.dtype),
            )
        supported_dtypes.add(torch.bfloat16)

    validate_dtype(points, supported_dtypes, "points")

    computer: PersistenceComputer | None = None
    try:
        computer = get_best_backend(backend_requirement)
        diagrams, mask, num_pairs = computer.compute_vr(points, max_dim, max_radius, metric)
    except (NerveError, RuntimeError) as e:
        backend_name = type(computer).__name__ if computer is not None else "unresolved"
        raise PersistenceError(
            f"VR persistence computation failed: {e}", backend=backend_name
        ) from e

    return PersistenceResult(
        diagrams=diagrams, mask=mask, num_pairs=num_pairs, was_batched=was_batched
    )


__all__ = [
    "PersistenceComputer",
    "TorchCBackend",
    "CoreCBackend",
    "PythonBackend",
    "PersistenceResult",
    "get_best_backend",
    "get_persistence_backend",
    "compute_persistence_vr",
]
