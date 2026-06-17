"""Shared type aliases and protocols for Nerve."""

from __future__ import annotations

from collections.abc import AsyncIterator as StdAsyncIterator
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor
else:
    try:
        from torch import Tensor
    except ImportError:

        class Tensor:  # type: ignore[no-redef]
            """Runtime fallback when PyTorch is not installed.

            Preserves structural type information in unions (``np.ndarray | Tensor``)
            rather than collapsing to ``Any``. At runtime this class is never
            instantiated because torch-dependent code paths check for torch first.
            """


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

ArrayLike: TypeAlias = np.ndarray | Tensor | list[Any] | tuple[Any, ...]
"""Array-like input accepted by Nerve operations.
Accepts NumPy arrays, PyTorch tensors, or plain Python lists/tuples."""

Numeric: TypeAlias = int | float | np.number | Tensor
"""A numeric value accepted by Nerve operations."""

PointCloud: TypeAlias = np.ndarray | Tensor
"""A point cloud as an ``(N, D)`` array.
Can be a NumPy array (float64) or a PyTorch tensor."""

DistanceMatrix: TypeAlias = np.ndarray | Tensor
"""A pairwise distance matrix as an ``(N, N)`` array."""

PersistencePair: TypeAlias = tuple[float, float, int]
"""A single persistence pair: ``(birth, death, dimension)``."""


@runtime_checkable
class PersistenceDiagramLike(Protocol):
    """Protocol for objects that behave like a persistence diagram.

    Implementations must expose a ``pairs`` list or ``pairs_array``
    property returning an (N, 3) array of ``(birth, death, dimension)``.
    """

    @property
    def pairs(self) -> list[tuple[float, float, int]]: ...

    @property
    def pairs_array(self) -> Any: ...


@runtime_checkable
class FilterFunction(Protocol):
    """Callable protocol for Mapper filter/lens functions.

    Takes a point cloud tensor and returns a 1D tensor of filter values.
    """

    def __call__(self, points: Tensor) -> Tensor: ...


@runtime_checkable
class ClusteringAlgorithm(Protocol):
    """Callable protocol for Mapper clustering algorithms.

    Takes a point cloud tensor and returns integer cluster labels.
    """

    def __call__(self, points: Tensor, **kwargs: Any) -> Tensor: ...


@runtime_checkable
class DistanceMetric(Protocol):
    """Callable protocol for distance functions.

    Takes two point cloud tensors and returns a distance matrix.
    """

    def __call__(self, a: Tensor, b: Tensor) -> Tensor: ...


@runtime_checkable
class VectorizationMethod(Protocol):
    """Callable protocol for persistence diagram vectorization.

    Takes a diagram tensor and returns a feature vector.
    """

    def __call__(self, diagram: Tensor, **kwargs: Any) -> Tensor: ...


@runtime_checkable
class PersistenceComputer(Protocol[T_co]):
    """Protocol for persistence computation backends.

    Implementations compute persistent homology for point-cloud or
    distance-matrix input.
    """

    def compute(self, data: PointCloud, max_dim: int, **kwargs: Any) -> T_co: ...


@runtime_checkable
class AsyncIterable(Protocol[T_co]):
    """Runtime-checkable async iterable protocol."""

    def __aiter__(self) -> StdAsyncIterator[T_co]: ...


class PersistenceConfig(Protocol):
    """Protocol for persistence configuration objects.

    Attributes:
        max_dim: Maximum homology dimension.
        max_radius: Filtration radius cutoff.
        metric: Distance metric identifier.
    """

    max_dim: int
    max_radius: float
    metric: str

    def validate(self) -> None: ...


class MapperConfig(Protocol):
    """Protocol for Mapper configuration objects.

    Attributes:
        filter_function: Filter/lens function or name.
        cover_resolution: Number of cover intervals.
        cover_overlap: Overlap fraction between intervals.
        clusterer: Clustering algorithm identifier.
    """

    filter_function: str | FilterFunction
    cover_resolution: int
    cover_overlap: float
    clusterer: str

    def validate(self) -> None: ...


__all__ = [
    "ArrayLike",
    "Numeric",
    "PointCloud",
    "DistanceMatrix",
    "PersistencePair",
    "PersistenceDiagramLike",
    "FilterFunction",
    "ClusteringAlgorithm",
    "DistanceMetric",
    "VectorizationMethod",
    "PersistenceComputer",
    "AsyncIterable",
    "PersistenceConfig",
    "MapperConfig",
]
