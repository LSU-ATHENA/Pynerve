"""Persistence and sketch building blocks."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from .._compute_api import compute_persistence as core_compute_persistence
from ._building_blocks_diagram import PersistenceDiagram

_SPARSE_RIPS_ALGORITHMS = {"greedy_permutation"}
_WITNESS_METHODS = {"farthest", "random", "kmeans"}
_SKETCH_METHODS = {"statistics", "landscape", "image", "betti_curve"}


def _validate_point_cloud(points: Tensor) -> None:
    if not isinstance(points, Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.dim() != 2:
        raise ValueError("points must have shape (n_points, dimension)")
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("points must contain at least one point and one coordinate")
    if not torch.isfinite(points).all():
        raise ValueError("points must contain only finite values")


def _zero_dimensional_diagram(points: Tensor) -> PersistenceDiagram:
    points = points.detach()
    n_points = points.shape[0]
    births = points.new_zeros(n_points)
    dimensions = torch.zeros(n_points, dtype=torch.long, device=points.device)
    if n_points == 1:
        return PersistenceDiagram(births, points.new_tensor([float("inf")]), dimensions)

    distances = torch.cdist(points, points)
    visited = torch.zeros(n_points, dtype=torch.bool, device=points.device)
    best = points.new_full((n_points,), float("inf"))
    visited[0] = True
    best[:] = distances[0]
    best[0] = float("inf")
    finite_deaths: list[Tensor] = []

    for _ in range(n_points - 1):
        masked = best.masked_fill(visited, float("inf"))
        edge_length, next_idx = torch.min(masked, dim=0)
        if not torch.isfinite(edge_length):
            break
        finite_deaths.append(edge_length)
        visited[next_idx] = True
        best = torch.minimum(best, distances[next_idx])

    if finite_deaths:
        deaths = torch.cat([torch.stack(finite_deaths), points.new_tensor([float("inf")])])
    else:
        deaths = points.new_full((1,), float("inf"))
    if deaths.numel() < n_points:
        deaths = torch.cat([deaths, points.new_full((n_points - deaths.numel(),), float("inf"))])
    return PersistenceDiagram(births, deaths[:n_points], dimensions)


def _core_diagram(points: Tensor, max_dim: int) -> PersistenceDiagram:
    result = core_compute_persistence(points.detach().cpu().numpy(), max_dim=max_dim)
    dim_pairs: list[list[list[float]]] = [[] for _ in range(max_dim + 1)]
    for birth, death, dim in result.pairs:
        dim_index = int(dim)
        if 0 <= dim_index <= max_dim:
            dim_pairs[dim_index].append([float(birth), float(death)])

    births = []
    deaths = []
    dimensions = []
    for dim, pairs in enumerate(dim_pairs):
        if not pairs:
            continue
        values = torch.as_tensor(pairs, dtype=points.dtype, device=points.device)
        births.append(values[:, 0])
        deaths.append(values[:, 1])
        dimensions.append(
            torch.full((values.shape[0],), dim, dtype=torch.long, device=points.device)
        )

    if not births:
        return PersistenceDiagram(
            points.new_empty(0),
            points.new_empty(0),
            torch.empty(0, dtype=torch.long, device=points.device),
        )
    return PersistenceDiagram(torch.cat(births), torch.cat(deaths), torch.cat(dimensions))


def _compute_diagram(points: Tensor, max_dim: int) -> PersistenceDiagram:
    if max_dim == 0:
        return _zero_dimensional_diagram(points)
    return _core_diagram(points, max_dim)


def _greedy_cover_landmarks(points: Tensor, cover_radius: float) -> Tensor:
    n_points = points.shape[0]
    selected = [0]
    min_dist = torch.cdist(points[:1], points).squeeze(0)
    while selected:
        farthest_distance, farthest_idx = torch.max(min_dist, dim=0)
        if farthest_distance <= cover_radius or len(selected) == n_points:
            break
        selected.append(int(farthest_idx.item()))
        dist = torch.cdist(points[farthest_idx : farthest_idx + 1], points).squeeze(0)
        min_dist = torch.minimum(min_dist, dist)
    return torch.tensor(selected, dtype=torch.long, device=points.device)


def _farthest_landmarks(points: Tensor, n_landmarks: int) -> Tensor:
    n_points = points.shape[0]
    count = min(n_landmarks, n_points)
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    selected[0] = 0
    min_dist = torch.cdist(points[:1], points).squeeze(0)
    for i in range(1, count):
        selected[i] = torch.argmax(min_dist)
        dist = torch.cdist(points[selected[i] : selected[i] + 1], points).squeeze(0)
        min_dist = torch.minimum(min_dist, dist)
    return selected


def _kmeans_landmarks(points: Tensor, n_landmarks: int, iterations: int = 20) -> Tensor:
    count = min(n_landmarks, points.shape[0])
    centers = points[_farthest_landmarks(points, count)].clone()
    for _ in range(iterations):
        assignments = torch.argmin(torch.cdist(points, centers), dim=1)
        next_centers = centers.clone()
        for cluster in range(count):
            members = points[assignments == cluster]
            if members.numel() != 0:
                next_centers[cluster] = members.mean(dim=0)
        if torch.allclose(next_centers, centers):
            break
        centers = next_centers
    return centers


class SparseRipsPersistence:
    """Sparse Rips approximation on a greedy cover."""

    def __init__(
        self,
        sparse_parameter: float = 0.1,
        max_dim: int = 1,
        algorithm: Literal["greedy_permutation"] = "greedy_permutation",
    ):
        """Initialize sparse Rips persistence.

        :param sparse_parameter: Cover radius for greedy landmark selection (must be positive finite)
        :param max_dim: Maximum homology dimension to compute (must be non-negative)
        :param algorithm: Algorithm used for sparse Rips approximation
        :raises ValueError: If sparse_parameter is not a positive finite value
        :raises ValueError: If max_dim is negative
        :raises ValueError: If algorithm is not one of ``greedy_permutation``
        """
        if not math.isfinite(sparse_parameter) or sparse_parameter <= 0:
            raise ValueError("sparse_parameter must be a positive finite value")
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        if algorithm not in _SPARSE_RIPS_ALGORITHMS:
            raise ValueError("unknown Sparse Rips algorithm")

        self.sparse_parameter = sparse_parameter
        self.max_dim = max_dim
        self.algorithm = algorithm

    def __call__(self, points: Tensor) -> PersistenceDiagram:
        """Compute sparse Rips persistence diagram from a point cloud.

        :param points: Point cloud tensor of shape ``(n_points, dimension)``
        :returns: Persistence diagram
        :raises TypeError: If points is not a ``torch.Tensor``
        :raises ValueError: If points shape is invalid or contains non-finite values
        """
        _validate_point_cloud(points)
        landmark_idx = _greedy_cover_landmarks(points, self.sparse_parameter)
        return _compute_diagram(points[landmark_idx], self.max_dim)


class WitnessComplexPersistence:
    """Witness complex approximation on selected landmarks."""

    def __init__(
        self,
        n_landmarks: int = 1000,
        max_dim: int = 1,
        method: Literal["farthest", "random", "kmeans"] = "farthest",
        random_seed: int | None = None,
    ):
        """Initialize witness complex persistence.

        :param n_landmarks: Number of landmark points to select (must be positive)
        :param max_dim: Maximum homology dimension to compute (must be non-negative)
        :param method: Landmark selection method (``"farthest"``, ``"random"``, or ``"kmeans"``)
        :param random_seed: Seed for reproducible random landmark selection
        :raises ValueError: If n_landmarks is not positive
        :raises ValueError: If max_dim is negative
        :raises ValueError: If method is not one of ``"farthest"``, ``"random"``, ``"kmeans"``
        """
        if n_landmarks <= 0:
            raise ValueError("n_landmarks must be positive")
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        if method not in _WITNESS_METHODS:
            raise ValueError("unknown witness landmark method")

        self.n_landmarks = n_landmarks
        self.max_dim = max_dim
        self.method = method
        self.random_seed = random_seed

    def __call__(self, points: Tensor) -> PersistenceDiagram:
        """Compute witness complex persistence diagram from a point cloud.

        :param points: Point cloud tensor of shape ``(n_points, dimension)``
        :returns: Persistence diagram
        :raises TypeError: If points is not a ``torch.Tensor``
        :raises ValueError: If points shape is invalid or contains non-finite values
        """
        _validate_point_cloud(points)
        count = min(self.n_landmarks, points.shape[0])
        if self.method == "random":
            generator = None
            if self.random_seed is not None:
                generator = torch.Generator(device=points.device)
                generator.manual_seed(self.random_seed)
            landmark_idx = torch.randperm(
                points.shape[0], generator=generator, device=points.device
            )[:count]
            landmarks = points[landmark_idx]
        elif self.method == "kmeans":
            landmarks = _kmeans_landmarks(points, count)
        else:
            landmark_idx = _farthest_landmarks(points, count)
            landmarks = points[landmark_idx]
        return _compute_diagram(landmarks, self.max_dim)


class PersistenceSketch:
    """Fixed-size topological signatures."""

    def __init__(
        self,
        output_dim: int = 64,
        method: Literal["statistics", "landscape", "image", "betti_curve"] = "statistics",
        max_dim: int = 1,
    ):
        """Initialize persistence sketch.

        :param output_dim: Size of the output sketch vector (must be positive)
        :param method: Sketch method (``"statistics"``, ``"landscape"``, ``"image"``, or ``"betti_curve"``)
        :param max_dim: Maximum homology dimension (must be non-negative)
        :raises ValueError: If output_dim is not positive
        :raises ValueError: If max_dim is negative
        :raises ValueError: If method is not one of ``"statistics"``, ``"landscape"``, ``"image"``, ``"betti_curve"``
        """
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        if method not in _SKETCH_METHODS:
            raise ValueError("unknown persistence sketch method")

        self.output_dim = output_dim
        self.method = method
        self.max_dim = max_dim

    def __call__(self, points_or_diagram: Tensor | PersistenceDiagram) -> Tensor:
        """Compute persistence sketch from a point cloud or diagram.

        :param points_or_diagram: Point cloud tensor of shape ``(n_points, dimension)`` or a ``PersistenceDiagram``
        :returns: Fixed-size sketch tensor of length ``output_dim``
        :raises TypeError: If a point cloud is given and is not a ``torch.Tensor``
        :raises ValueError: If a point cloud is given and its shape is invalid or contains non-finite values
        """
        if isinstance(points_or_diagram, PersistenceDiagram):
            return self._from_diagram(points_or_diagram)

        _validate_point_cloud(points_or_diagram)
        return self._from_diagram(_zero_dimensional_diagram(points_or_diagram))

    def _from_diagram(self, diagram: PersistenceDiagram) -> Tensor:
        if self.method == "statistics":
            stats = self._statistics_from_diagram(diagram)
        elif self.method == "betti_curve":
            stats = self._betti_curve_from_diagram(diagram)
        elif self.method == "landscape":
            stats = self._landscape_from_diagram(diagram)
        else:
            stats = self._image_from_diagram(diagram)
        if stats.numel() >= self.output_dim:
            return stats[: self.output_dim]
        padding = stats.new_zeros(self.output_dim - stats.numel())
        return torch.cat([stats, padding])

    def _statistics_from_diagram(self, diagram: PersistenceDiagram) -> Tensor:
        dtype = diagram.births.dtype if diagram.births.is_floating_point() else torch.float32
        births = diagram.births.detach().to(dtype=dtype)
        deaths = diagram.deaths.detach().to(dtype=dtype)
        persistence = deaths - births
        finite_deaths = deaths[torch.isfinite(deaths)]
        finite_persistence = persistence[torch.isfinite(persistence)]

        def summary(values: Tensor) -> Tensor:
            if values.numel() == 0:
                return births.new_zeros(4)
            return torch.stack(
                [
                    values.mean(),
                    values.std(unbiased=False),
                    values.min(),
                    values.max(),
                ]
            )

        count = births.new_tensor([float(len(diagram))])
        max_dimension = (
            diagram.dimensions.detach().to(dtype=dtype).max().view(1)
            if len(diagram)
            else births.new_zeros(1)
        )
        return torch.cat(
            [
                count,
                max_dimension,
                summary(births),
                summary(finite_deaths),
                summary(finite_persistence),
            ]
        )

    def _betti_curve_from_diagram(self, diagram: PersistenceDiagram) -> Tensor:
        dtype = diagram.births.dtype if diagram.births.is_floating_point() else torch.float32
        births = diagram.births.detach().to(dtype=dtype)
        deaths = diagram.deaths.detach().to(dtype=dtype)
        finite_deaths = deaths[torch.isfinite(deaths)]
        if births.numel() == 0:
            return births.new_zeros(self.output_dim)

        end = finite_deaths.max() if finite_deaths.numel() else births.max()
        grid = torch.linspace(births.min(), end, self.output_dim, dtype=dtype, device=births.device)
        alive = (births[:, None] <= grid[None, :]) & (deaths[:, None] > grid[None, :])
        return alive.sum(dim=0).to(dtype=dtype)

    def _landscape_from_diagram(self, diagram: PersistenceDiagram) -> Tensor:
        persistence = diagram.persistence_values().detach()
        persistence = persistence[torch.isfinite(persistence)]
        if persistence.numel() == 0:
            return diagram.births.detach().new_zeros(self.output_dim)

        values = torch.sort(torch.clamp(persistence, min=0), descending=True).values
        if values.numel() >= self.output_dim:
            return values[: self.output_dim]
        return torch.cat([values, values.new_zeros(self.output_dim - values.numel())])

    def _image_from_diagram(self, diagram: PersistenceDiagram) -> Tensor:
        dtype = diagram.births.dtype if diagram.births.is_floating_point() else torch.float32
        births = diagram.births.detach().to(dtype=dtype)
        deaths = diagram.deaths.detach().to(dtype=dtype)
        finite_mask = torch.isfinite(deaths)
        if not torch.any(finite_mask):
            return births.new_zeros(self.output_dim)

        births = births[finite_mask]
        persistence = torch.clamp(deaths[finite_mask] - births, min=0)
        bins = max(1, self.output_dim)
        if births.max() > births.min():
            scaled = ((births - births.min()) / (births.max() - births.min()) * (bins - 1)).long()
            scaled = torch.clamp(scaled, 0, bins - 1)
        else:
            scaled = torch.zeros_like(births, dtype=torch.long)

        image = births.new_zeros(bins)
        image.scatter_add_(0, scaled, persistence)
        return image
