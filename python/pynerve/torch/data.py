"""PyTorch dataset and collation utilities for persistence data."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset

import torch

from .._validation import validate_positive_int as _validate_positive_int
from ._diagram import PersistenceDiagram, batch_diagrams
from ._persistence_api import (
    _validate_max_dim,
    _validate_max_radius,
    _validate_metric,
)
from ._persistence_vr import vr_persistence


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        if name == "num_workers":
            raise ValueError("num_workers must be non-negative")
        raise ValueError(f"{name} must be non-negative")
    return result


def _stack_labels(labels: list[torch.Tensor | list[Any] | tuple[Any, ...]]) -> torch.Tensor:
    return torch.stack(
        [label if isinstance(label, torch.Tensor) else torch.tensor(label) for label in labels]
    )


def _validate_single_point_cloud(
    pc: torch.Tensor, dim: int, device: torch.device, dtype: torch.dtype
) -> None:
    if not isinstance(pc, torch.Tensor):
        raise TypeError("point clouds must be tensors")
    if pc.dim() != 2:
        raise ValueError("point clouds must be 2D tensors")
    if pc.shape[0] == 0:
        raise ValueError("point clouds must contain at least one point")
    if pc.shape[1] == 0:
        raise ValueError("point clouds must contain at least one coordinate")
    if not torch.is_floating_point(pc):
        raise TypeError("point clouds must use a floating-point dtype")
    if not torch.isfinite(pc).all().item():
        raise ValueError("point clouds must contain only finite coordinates")
    if pc.shape[1] != dim:
        raise ValueError("all point clouds must share the same coordinate dimension")
    if pc.device != device:
        raise ValueError("all point clouds must be on the same device")
    if pc.dtype != dtype:
        raise ValueError("all point clouds must have the same dtype")


def _validate_point_cloud_batch(
    batch: list[torch.Tensor],
) -> tuple[int, torch.device, torch.dtype]:
    if not batch:
        raise ValueError("batch must be non-empty")
    if not isinstance(batch[0], torch.Tensor):
        raise TypeError("point clouds must be tensors")
    if batch[0].dim() != 2:
        raise ValueError("point clouds must be 2D tensors")
    dim = batch[0].shape[1]
    device = batch[0].device
    dtype = batch[0].dtype
    for pc in batch:
        _validate_single_point_cloud(pc, dim, device, dtype)
    return dim, device, dtype


def collate_diagrams(
    batch: list[PersistenceDiagram | tuple[PersistenceDiagram, torch.Tensor]],
) -> PersistenceDiagram | tuple[PersistenceDiagram, torch.Tensor]:
    """Batch variable-length diagrams, preserving optional labels.

    :param batch: List of :class:`PersistenceDiagram` objects or
        ``(diagram, label)`` tuples.
    :returns: A single batched :class:`PersistenceDiagram`, or a
        ``(batched_diagram, stacked_labels)`` tuple if labels were present.
    """
    if len(batch) == 0:
        return PersistenceDiagram(
            torch.zeros((0, 0, 3)),
            torch.zeros((0, 0), dtype=torch.bool),
            torch.zeros((0, 0), dtype=torch.long),
        )

    if isinstance(batch[0], tuple):
        tuple_batch = cast(list[tuple[PersistenceDiagram, torch.Tensor]], batch)
        diagrams = [item[0] for item in tuple_batch]
        labels = _stack_labels([item[1] for item in tuple_batch])
        return batch_diagrams(diagrams), labels
    return batch_diagrams(cast(list[PersistenceDiagram], batch))


def _pad_point_clouds(
    batch: list[torch.Tensor],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad point clouds to the largest point count in the batch."""
    if not math.isfinite(float(pad_value)):
        raise ValueError("pad_value must be finite")
    if len(batch) == 0:
        return torch.empty(0)
    dim, _, _ = _validate_point_cloud_batch(batch)
    max_points = max(pc.shape[0] for pc in batch)

    padded: list[torch.Tensor] = []
    for pc in batch:
        n_points = pc.shape[0]
        if n_points < max_points:
            padding = torch.full(
                (max_points - n_points, dim),
                pad_value,
                dtype=pc.dtype,
                device=pc.device,
            )
            pc_padded = torch.cat([pc, padding], dim=0)
        else:
            pc_padded = pc
        padded.append(pc_padded)

    return torch.stack(padded)


def collate_point_clouds(
    batch: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    pad_value: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pad point clouds to the largest point count in the batch, preserving optional labels.

    :param batch: List of 2D tensors or ``(point_cloud, label)`` tuples.
    :param pad_value: Value used for padding (must be finite).
    :returns: Batched padded tensor of shape ``(B, max_N, D)``, or a
        ``(batched_tensor, stacked_labels)`` tuple if labels were present.
    :raises ValueError: If ``pad_value`` is not finite or batch entries
        are inconsistent.
    """
    if len(batch) == 0:
        return torch.empty(0)
    if isinstance(batch[0], tuple):
        tuple_batch = cast(list[tuple[torch.Tensor, torch.Tensor]], batch)
        if not all(isinstance(item, tuple) and len(item) == 2 for item in tuple_batch):
            raise ValueError("batch entries must all be (point_cloud, label) tuples")
        points: list[torch.Tensor] = [item[0] for item in tuple_batch]
        labels = _stack_labels([item[1] for item in tuple_batch])
        return _pad_point_clouds(points, pad_value), labels
    pc_batch: list[torch.Tensor] = cast(list[torch.Tensor], batch)
    return _pad_point_clouds(pc_batch, pad_value)


class PersistenceDataset(Dataset[Any]):
    """Dataset that stores point clouds and returns persistence diagrams."""

    def __init__(
        self,
        point_clouds: list[torch.Tensor],
        labels: torch.Tensor | None = None,
        max_dim: int = 2,
        max_radius: float = float("inf"),
        metric: str = "euclidean",
        cache: bool = False,
    ):
        """Initialise a persistence dataset.

        :param point_clouds: List of 2D point cloud tensors.
        :param labels: Label tensor whose length matches
            ``point_clouds``; optional.
        :param max_dim: Maximum homology dimension (non-negative integer).
        :param max_radius: Filtration cutoff radius (positive).
        :param metric: Distance metric name.
        :param cache: If ``True``, cache computed diagrams in memory.
        :raises ValueError: If ``labels`` length is inconsistent or
            ``point_clouds`` is invalid.
        """
        if labels is not None and len(labels) != len(point_clouds):
            raise ValueError("labels length must match point_clouds length")
        if point_clouds:
            _validate_point_cloud_batch(point_clouds)
        self.point_clouds = point_clouds
        self.labels = labels
        self.max_dim = _validate_max_dim(max_dim)
        self.max_radius = _validate_max_radius(max_radius)
        self.metric = _validate_metric(metric)
        self.cache = cache
        self._cache: dict[int, PersistenceDiagram] | None = {} if cache else None

    def __len__(self) -> int:
        """Return the number of point clouds."""
        return len(self.point_clouds)

    def __getitem__(self, idx: int) -> PersistenceDiagram | tuple[PersistenceDiagram, torch.Tensor]:
        """Compute (or fetch cached) the persistence diagram at index ``idx``.

        :param idx: Dataset index.
        :returns: A :class:`PersistenceDiagram` or, if labels exist, a
            ``(diagram, label)`` tuple.
        """
        _cache = self._cache
        if _cache is not None and idx in _cache:
            diagram = _cache[idx]
        else:
            points = self.point_clouds[idx]
            diagram = cast(
                PersistenceDiagram,
                vr_persistence(
                    points,
                    max_dim=self.max_dim,
                    max_radius=self.max_radius,
                    metric=self.metric,
                ),
            )

            if _cache is not None:
                _cache[idx] = diagram

        if self.labels is not None:
            return diagram, self.labels[idx]
        return diagram


class PointCloudDataset(Dataset[Any]):
    """Dataset that returns point clouds directly."""

    def __init__(self, point_clouds: list[torch.Tensor], labels: torch.Tensor | None = None):
        """Initialise a point cloud dataset.

        :param point_clouds: List of 2D point cloud tensors.
        :param labels: Label tensor whose length matches
            ``point_clouds``; optional.
        :raises ValueError: If ``labels`` length is inconsistent or
            ``point_clouds`` is invalid.
        """
        if labels is not None and len(labels) != len(point_clouds):
            raise ValueError("labels length must match point_clouds length")
        if point_clouds:
            _validate_point_cloud_batch(point_clouds)
        self.point_clouds = point_clouds
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of point clouds."""
        return len(self.point_clouds)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the point cloud at index ``idx``.

        :param idx: Dataset index.
        :returns: A 2D tensor or, if labels exist, a
            ``(point_cloud, label)`` tuple.
        """
        points = self.point_clouds[idx]
        if self.labels is not None:
            return points, self.labels[idx]
        return points


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs: Any,
) -> DataLoader[Any]:
    """Create a :class:`DataLoader` with appropriate collation for the dataset type.

    :param dataset: A :class:`PersistenceDataset`,
        :class:`PointCloudDataset`, or generic PyTorch dataset.
    :param batch_size: Batch size (positive integer).
    :param shuffle: Whether to shuffle the data each epoch.
    :param num_workers: Number of data-loading subprocesses.
    :param kwargs: Additional arguments forwarded to :class:`DataLoader`.
    :returns: Configured :class:`DataLoader`.
    :raises ValueError: If ``batch_size`` is not positive or
        ``num_workers`` is negative.
    """
    batch_size = _validate_positive_int(batch_size, "batch_size")  # batch_size must be positive
    num_workers = _validate_nonnegative_int(num_workers, "num_workers")
    collate_fn: Callable[..., Any]
    if isinstance(dataset, PersistenceDataset):
        collate_fn = collate_diagrams
    elif isinstance(dataset, PointCloudDataset):
        collate_fn = collate_point_clouds
    else:
        collate_fn = collate_diagrams

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


__all__ = [
    "collate_diagrams",
    "collate_point_clouds",
    "PersistenceDataset",
    "PointCloudDataset",
    "create_dataloader",
]
