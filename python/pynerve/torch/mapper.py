"""Public Mapper API for topological data analysis."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast

import torch
from torch import Tensor

from ..exceptions import ValidationError
from ._mapper_impl import (
    _CLUSTERERS,
    _FILTER_FUNCTIONS,
    _filter_eccentricity_python,
    _filter_pca_python,
    _mapper_python,
    _mapper_python_custom,
    _validate_finite_scalar,
    _validate_finite_tensor,
    _validate_positive_int,
)


def _validate_public_point_cloud(point_cloud: Tensor) -> None:
    try:
        _validate_finite_tensor(point_cloud, "point_cloud")
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc), parameter="point_cloud") from exc
    if point_cloud.dim() != 2:
        raise ValidationError(
            f"point_cloud must be 2D [N, D], got shape {tuple(point_cloud.shape)}",
            parameter="point_cloud",
        )
    if point_cloud.shape[0] == 0:
        raise ValidationError("point_cloud must be non-empty", parameter="point_cloud")


def _validate_public_positive_int(name: str, value: int) -> int:
    try:
        return cast(int, _validate_positive_int(value, name))
    except ValueError as exc:
        raise ValidationError(str(exc), parameter=name) from exc


def _validate_public_finite_scalar(name: str, value: float) -> float:
    try:
        return cast(float, _validate_finite_scalar(value, name))
    except ValueError as exc:
        raise ValidationError(str(exc), parameter=name) from exc


def _validate_mapper_params(
    point_cloud: Tensor,
    cover_resolution: int,
    cover_overlap: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
    clusterer: str,
    filter_function: str | Callable[[Tensor], Tensor],
) -> tuple[int, float, float, int]:
    _validate_public_point_cloud(point_cloud)
    cover_resolution = _validate_public_positive_int("cover_resolution", cover_resolution)
    cover_overlap = _validate_public_finite_scalar("cover_overlap", cover_overlap)
    dbscan_eps = _validate_public_finite_scalar("dbscan_eps", dbscan_eps)
    dbscan_min_samples = _validate_public_positive_int("dbscan_min_samples", dbscan_min_samples)
    if not 0.0 <= cover_overlap < 1.0:
        raise ValidationError("cover_overlap must be in [0, 1)", parameter="cover_overlap")
    if dbscan_eps <= 0:
        raise ValidationError("dbscan_eps must be > 0", parameter="dbscan_eps")
    if clusterer not in _CLUSTERERS:
        raise ValidationError(f"Unknown clusterer: {clusterer}", parameter="clusterer")
    if not callable(filter_function) and filter_function not in _FILTER_FUNCTIONS:
        raise ValidationError(
            f"Unknown filter_function: {filter_function}",
            parameter="filter_function",
        )
    return cover_resolution, cover_overlap, dbscan_eps, dbscan_min_samples


def _build_internal_result(
    point_cloud: Tensor,
    filter_function: str,
    cover_resolution: int,
    cover_overlap: float,
    clusterer: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    return_graph: bool,
) -> dict[str, Any]:
    import pynerve_torch_internal as _torch_C  # noqa: N812, PLC0415

    config = _torch_C.MapperConfig()
    config.filter_function = filter_function
    config.cover_resolution = cover_resolution
    config.cover_overlap = cover_overlap

    _clusterer_map = {
        "dbscan": _torch_C.ClustererType.DBSCAN,
        "single_linkage": _torch_C.ClustererType.SINGLE_LINKAGE,
        "connected": _torch_C.ClustererType.CONNECTED,
    }
    clusterer_type = _clusterer_map.get(clusterer)
    if clusterer_type is None:
        raise ValidationError(f"Unknown clusterer: {clusterer}", parameter="clusterer")
    config.clusterer = clusterer_type
    config.dbscan_eps = dbscan_eps
    config.dbscan_min_samples = dbscan_min_samples

    mapper_obj = _torch_C.Mapper(config)
    graph = mapper_obj.fit_transform(point_cloud)

    nodes = [
        {
            "id": node.id,
            "point_indices": node.point_indices,
            "centroid": node.centroid,
            "filter_centroid": node.filter_centroid,
            "cover_index": node.cover_index,
        }
        for node in graph.nodes
    ]
    edges = [
        {"source": edge.source, "target": edge.target, "weight": edge.weight}
        for edge in graph.edges
    ]
    result: dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "filter_values": mapper_obj.get_last_filter_values(),
    }

    if return_graph:
        with suppress(ImportError):
            import networkx as nx  # noqa: PLC0415

            nx_graph = nx.Graph()
            for node in nodes:
                nx_graph.add_node(
                    node["id"],
                    point_indices=node["point_indices"],
                    centroid=node["centroid"],
                    filter_centroid=node["filter_centroid"],
                )
            for edge in edges:
                nx_graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])
            result["graph"] = nx_graph
    return result


def mapper(
    point_cloud: Tensor,
    filter_function: str | Callable[[Tensor], Tensor] = "pca_2d",
    cover_resolution: int = 10,
    cover_overlap: float = 0.25,
    clusterer: str = "dbscan",
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    return_graph: bool = True,
) -> dict[str, Any]:
    """Apply the Mapper algorithm to point-cloud data.

    :param point_cloud: 2D tensor of shape ``(N, D)``.
    :param filter_function: Name of a built-in filter (``"pca_2d"``,
        ``"pca_1d"``, ``"eccentricity"``, ``"identity"``) or a custom
        callable.
    :param cover_resolution: Number of cover intervals (positive integer).
    :param cover_overlap: Overlap fraction in ``[0, 1)``.
    :param clusterer: Clustering algorithm (``"dbscan"``,
        ``"single_linkage"``, ``"connected"``).
    :param dbscan_eps: Epsilon for DBSCAN (must be positive).
    :param dbscan_min_samples: Minimum DBSCAN samples (positive integer).
    :param return_graph: If ``True``, include a ``networkx`` graph in the
        result.
    :returns: Dict with ``"nodes"``, ``"edges"``, ``"filter_values"``,
        and optionally ``"graph"``.
    :raises ValidationError: If parameters are invalid.
    """
    cover_resolution, cover_overlap, dbscan_eps, dbscan_min_samples = _validate_mapper_params(
        point_cloud,
        cover_resolution,
        cover_overlap,
        dbscan_eps,
        dbscan_min_samples,
        clusterer,
        filter_function,
    )

    if callable(filter_function):
        return _mapper_python_custom(
            point_cloud,
            filter_function,
            cover_resolution,
            cover_overlap,
            clusterer,
            dbscan_eps,
            dbscan_min_samples,
            return_graph,
        )

    try:
        return _build_internal_result(
            point_cloud,
            filter_function,
            cover_resolution,
            cover_overlap,
            clusterer,
            dbscan_eps,
            dbscan_min_samples,
            return_graph,
        )
    except ImportError:
        return _mapper_python(
            point_cloud,
            filter_function,
            cover_resolution,
            cover_overlap,
            clusterer,
            dbscan_eps,
            dbscan_min_samples,
            return_graph,
        )


class MapperTransformer:
    """scikit-learn style estimator for Mapper graph fitting/assignment.

    Provides ``fit``, ``fit_transform``, ``transform``, ``get_params``,
    and ``set_params`` following scikit-learn conventions.
    """

    mapper_result_: dict[str, Any] | None
    training_filter_values_: Tensor | None

    def __init__(
        self,
        filter_function: str = "pca_2d",
        cover_resolution: int = 10,
        cover_overlap: float = 0.25,
        clusterer: str = "dbscan",
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
    ):
        """Initialise the Mapper transformer.

        :param filter_function: Built-in filter name.
        :param cover_resolution: Number of cover intervals.
        :param cover_overlap: Overlap fraction.
        :param clusterer: Clustering algorithm name.
        :param dbscan_eps: DBSCAN epsilon.
        :param dbscan_min_samples: Minimum DBSCAN samples.
        """
        self.filter_function = filter_function
        self.cover_resolution = cover_resolution
        self.cover_overlap = cover_overlap
        self.clusterer = clusterer
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.mapper_result_ = None
        self.training_filter_values_ = None

    def fit(self, X: Tensor, y: Any = None) -> MapperTransformer:  # noqa: N803
        """Fit the Mapper graph to the point cloud.

        :param X: Point cloud tensor of shape ``(N, D)``.
        :param y: Ignored; present for scikit-learn compatibility.
        :returns: ``self``.
        """
        del y
        self.mapper_result_ = mapper(
            X,
            filter_function=self.filter_function,
            cover_resolution=self.cover_resolution,
            cover_overlap=self.cover_overlap,
            clusterer=self.clusterer,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
        )
        self.training_filter_values_ = self.mapper_result_["filter_values"]
        return self

    def fit_transform(self, X: Tensor, y: Any = None) -> dict[str, Any]:  # noqa: N803
        """Fit and return the Mapper result dict.

        :param X: Point cloud tensor of shape ``(N, D)``.
        :param y: Ignored; present for scikit-learn compatibility.
        :returns: Mapper result dict.
        """
        self.fit(X, y)
        return cast(dict[str, Any], self.mapper_result_)

    def transform(self, X: Tensor) -> Tensor:  # noqa: N803
        """Assign new points to the nearest Mapper node.

        :param X: Point cloud tensor of shape ``(N, D)``.
        :returns: 1D tensor of node IDs.
        :raises ValidationError: If the transformer has not been fitted or
            the filter function is unknown.
        """
        if self.mapper_result_ is None:
            raise ValidationError("Mapper not fitted yet. Call fit() first.")
        _validate_public_point_cloud(X)

        if self.filter_function == "pca_2d":
            filter_vals = _filter_pca_python(X, 2)
        elif self.filter_function == "pca_1d":
            filter_vals = _filter_pca_python(X, 1)
        elif self.filter_function == "eccentricity":
            filter_vals = _filter_eccentricity_python(X)
        elif self.filter_function == "identity":
            filter_vals = X[:, :2]
        else:
            raise ValidationError(
                f"Unknown filter_function: {self.filter_function}",
                parameter="filter_function",
            )

        nodes = self.mapper_result_["nodes"]
        if not nodes:
            raise ValidationError("Mapper result has no nodes.")
        assignments = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        for i, point_filter in enumerate(filter_vals):
            min_dist = float("inf")
            best_node = 0
            for node in nodes:
                dist = torch.norm(point_filter - node["filter_centroid"]).item()
                if dist < min_dist:
                    min_dist = dist
                    best_node = node["id"]
            assignments[i] = best_node
        return assignments

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the estimator parameters.

        :param deep: Ignored; present for scikit-learn compatibility.
        :returns: Dict of parameter names and values.
        """
        del deep
        return {
            "filter_function": self.filter_function,
            "cover_resolution": self.cover_resolution,
            "cover_overlap": self.cover_overlap,
            "clusterer": self.clusterer,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
        }

    def set_params(self, **params: Any) -> MapperTransformer:
        """Set estimator parameters.

        :param params: Keyword arguments mapping parameter names to values.
        :returns: ``self``.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


def visualize_mapper_graph(
    mapper_result: dict[str, Any],
    color_by: str = "size",
    layout: str = "spring",
    figsize: tuple[int, int] = (10, 8),
) -> Any:
    """Visualize Mapper graph using networkx/matplotlib.

    :param mapper_result: Dict returned by :func:`mapper` with
        ``return_graph=True``.
    :param color_by: Node colouring criterion (``"size"`` or ``"degree"``).
    :param layout: Graph layout algorithm (``"spring"``, ``"circular"``,
        or ``"kamada_kawai"``).
    :param figsize: Matplotlib figure size ``(width, height)``.
    :returns: Matplotlib ``Figure``.
    :raises ImportError: If ``networkx`` or ``matplotlib`` is not
        installed.
    :raises ValidationError: If the result has no graph or parameters are
        invalid.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import networkx as nx  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(f"visualize_mapper_graph requires networkx and matplotlib: {e}") from e

    graph = mapper_result.get("graph")
    if graph is None:
        raise ValidationError("Mapper result has no graph. Run mapper(..., return_graph=True).")

    if color_by == "size":
        node_colors = [len(node["point_indices"]) for node in mapper_result["nodes"]]
    elif color_by == "degree":
        node_colors = [graph.degree(node["id"]) for node in mapper_result["nodes"]]
    else:
        raise ValidationError("color_by must be 'size' or 'degree'", parameter="color_by")

    if layout == "spring":
        pos = nx.spring_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        raise ValidationError(
            "layout must be 'spring', 'circular', or 'kamada_kawai'",
            parameter="layout",
        )

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, cmap=plt.get_cmap("viridis"), node_size=500, ax=ax
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    ax.set_title(
        f"Mapper Graph ({len(mapper_result['nodes'])} nodes, {len(mapper_result['edges'])} edges)"
    )
    ax.axis("off")

    if node_colors:
        colorbar = plt.cm.ScalarMappable(
            cmap=plt.get_cmap("viridis"),
            norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)),  # pyright: ignore[reportPrivateImportUsage]
        )
        colorbar.set_array([])
        plt.colorbar(colorbar, ax=ax, label=color_by)
    return fig


__all__ = [
    "mapper",
    "MapperTransformer",
    "visualize_mapper_graph",
]
