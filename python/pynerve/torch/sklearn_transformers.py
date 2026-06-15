"""sklearn-compatible transformers for TDA features."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np

import torch

from ._diagram import PersistenceDiagram
from ._persistence_api import (
    _count_pairs_by_dimension,
    _valid_vr_output_mask,
    _validate_max_dim,
    _validate_max_radius,
    _validate_metric,
    alpha_persistence,
    witness_persistence,
)
from ._persistence_vr import vr_persistence
from ._sklearn_batched import BatchedTransformer
from ._sklearn_compat import BaseEstimator, TransformerMixin, _require_non_empty
from .preprocessing import clean_diagram
from .statistics import all_statistics
from .vectorization import (
    birth_death_curve,
    heat_kernel_signature,
    persistence_image,
    persistence_landscape,
    persistence_silhouette,
)

_COMPLEX_TYPES = {"vr", "witness", "alpha"}
_VECTORIZATION_METHODS: dict[str, Callable[..., torch.Tensor]] = {
    "image": persistence_image,
    "landscape": persistence_landscape,
    "silhouette": persistence_silhouette,
    "heat": heat_kernel_signature,
    "histogram": birth_death_curve,
}


def _as_float_tensor(value: torch.Tensor | np.ndarray | float) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value if torch.is_floating_point(value) else value.to(torch.float32)
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    if tensor.numel() > 0 and not torch.isfinite(tensor).all().item():
        raise ValueError("tensor inputs must contain only finite values")
    return tensor


def _validate_sequence(name: str, values: Sequence[Any]) -> None:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise TypeError(f"{name} must be a sequence")
    _require_non_empty(name, values)


def _validate_point_cloud(point_cloud: torch.Tensor, name: str = "point cloud") -> None:
    if point_cloud.dim() != 2:
        raise ValueError(f"{name} must have shape (n_points, n_coordinates)")
    if point_cloud.shape[0] == 0 or point_cloud.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")


def _validate_diagram_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float32)
    if tensor.dim() != 2 or tensor.shape[-1] < 2:
        raise ValueError("diagram must have shape (n_pairs, at least 2)")
    if tensor.numel() == 0:
        return tensor
    births = tensor[:, 0]
    deaths = tensor[:, 1]
    if not torch.isfinite(births).all().item():
        raise ValueError("diagram births must be finite")
    if torch.isnan(deaths).any().item():
        raise ValueError("diagram deaths must not be NaN")
    finite_deaths = torch.isfinite(deaths)
    if (
        finite_deaths.any().item()
        and not (deaths[finite_deaths] >= births[finite_deaths]).all().item()
    ):
        raise ValueError("diagram finite deaths must be greater than or equal to births")
    if tensor.shape[-1] >= 3:
        dimensions = tensor[:, 2]
        if not torch.isfinite(dimensions).all().item() or (dimensions < 0).any().item():
            raise ValueError("diagram dimensions must be finite and non-negative")
    return tensor


def _tensor_to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _single_diagram_tensor(value: PersistenceDiagram | torch.Tensor) -> torch.Tensor:
    tensor = value.diagrams if isinstance(value, PersistenceDiagram) else value
    if not isinstance(tensor, torch.Tensor):
        tensor = _as_float_tensor(tensor)
    if tensor.dim() == 3:
        if tensor.shape[0] != 1:
            raise ValueError("expected a single diagram, got a batched diagram tensor")
        tensor = tensor.squeeze(0)
    return _validate_diagram_tensor(tensor)


def _diagram_from_tensor(tensor: torch.Tensor, max_dim: int) -> PersistenceDiagram:
    if tensor.shape[-1] < 3:
        dim_col = torch.zeros_like(tensor[..., :1])
        tensor = torch.cat([tensor, dim_col], dim=-1)
    batched = tensor if tensor.dim() == 3 else tensor.unsqueeze(0)
    mask = _valid_vr_output_mask(batched, batched.shape[0], python_output=False)
    num_pairs = _count_pairs_by_dimension(batched, mask, max_dim)
    if tensor.dim() == 2:
        mask = mask.squeeze(0)
        num_pairs = num_pairs.squeeze(0)
    return PersistenceDiagram(tensor, mask, num_pairs)


class PersistenceTransformer(BaseEstimator, TransformerMixin):
    """Compute persistence diagrams from point clouds."""

    def __init__(
        self,
        complex_type: Literal["vr", "witness", "alpha"] = "vr",
        max_dim: int = 1,
        max_radius: float = float("inf"),
        metric: str = "euclidean",
        preprocessing_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the persistence transformer.

        :param complex_type: Type of simplicial complex: ``"vr"`` (Vietoris-Rips),
            ``"witness"`` (witness complex), or ``"alpha"`` (Alpha complex).
        :param max_dim: Maximum homology dimension to compute.
        :param max_radius: Maximum filtration radius.
        :param metric: Distance metric for the Vietoris-Rips complex.
        :param preprocessing_params: Keyword arguments forwarded to :func:`clean_diagram`.
        :raises ValueError: If any parameter fails validation.
        """
        max_dim = _validate_max_dim(max_dim)
        max_radius = _validate_max_radius(max_radius)
        metric = _validate_metric(metric)
        if complex_type not in _COMPLEX_TYPES:
            raise ValueError(f"Unknown complex type: {complex_type}")
        self.complex_type = complex_type
        self.max_dim = max_dim
        self.max_radius = max_radius
        self.metric = metric
        self.preprocessing_params = dict(preprocessing_params or {})

    def fit(self, X: Sequence[Any], y: Any = None) -> PersistenceTransformer:  # noqa: N803
        """Validate input without performing computation.

        :param X: Sequence of input point clouds.
        :param y: Ignored; present for sklearn compatibility.
        :returns: *self*.
        """
        del y
        _validate_sequence("X", X)
        return self

    def transform(self, X: Sequence[Any]) -> list[PersistenceDiagram]:  # noqa: N803
        """Transform point clouds to persistence diagrams.

        :param X: Sequence of point clouds or (landmarks, witnesses) tuples.
        :returns: List of :class:`PersistenceDiagram` objects, one per input.
        :raises ValueError: If any point cloud is invalid.
        :raises TypeError: If ``X`` is not a sequence.
        """
        _validate_sequence("X", X)
        if self.complex_type not in _COMPLEX_TYPES:
            raise ValueError(f"Unknown complex type: {self.complex_type}")

        diagrams: list[PersistenceDiagram] = []

        for x in X:
            if self.complex_type == "vr":
                x_tensor = _as_float_tensor(x)
                _validate_point_cloud(x_tensor)
                pd = cast(
                    PersistenceDiagram,
                    vr_persistence(
                        x_tensor,
                        max_dim=self.max_dim,
                        max_radius=self.max_radius,
                        metric=self.metric,
                    ),
                )
            elif self.complex_type == "witness":
                landmarks, witnesses = self._unpack_witness_sample(x)
                landmarks = _as_float_tensor(landmarks)
                witnesses = _as_float_tensor(witnesses)
                _validate_point_cloud(landmarks, "landmarks")
                _validate_point_cloud(witnesses, "witnesses")
                if landmarks.shape[-1] != witnesses.shape[-1]:
                    raise ValueError(
                        "landmarks and witnesses must have the same coordinate dimension"
                    )
                pd = witness_persistence(
                    landmarks,
                    witnesses,
                    max_dim=self.max_dim,
                    max_radius=self.max_radius,
                )
            elif self.complex_type == "alpha":
                x_tensor = _as_float_tensor(x)
                _validate_point_cloud(x_tensor)
                pd = alpha_persistence(x_tensor, max_dim=self.max_dim)
            else:
                raise RuntimeError(f"Unexpected complex_type: {self.complex_type}")

            if self.preprocessing_params:
                clean_diagrams = clean_diagram(pd.diagrams, **self.preprocessing_params)
                pd = _diagram_from_tensor(clean_diagrams, self.max_dim)

            diagrams.append(pd)

        return diagrams

    @staticmethod
    def _unpack_witness_sample(sample: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(sample, dict):
            if "landmarks" not in sample or "witnesses" not in sample:
                raise ValueError("Witness samples must include 'landmarks' and 'witnesses'")
            return sample["landmarks"], sample["witnesses"]
        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            return sample[0], sample[1]
        raise ValueError(
            "Witness persistence expects (landmarks, witnesses) or a mapping "
            "with 'landmarks' and 'witnesses'"
        )

    def fit_transform(
        self,
        X: Sequence[Any],
        y: Any = None,
        **fit_params: Any,  # noqa: N803, ARG002
    ) -> list[PersistenceDiagram]:
        """Fit and transform in one call. Delegates to :meth:`fit` and :meth:`transform`.

        :param X: Input point clouds.
        :param y: Ignored.
        :param \\**fit_params: Ignored; for sklearn compatibility.
        :returns: List of :class:`PersistenceDiagram` objects.
        """
        return self.fit(X, y).transform(X)


class VectorizationTransformer(BaseEstimator, TransformerMixin):
    """Transform persistence diagrams to fixed-size vector representations."""

    def __init__(
        self,
        method: Literal["image", "landscape", "silhouette", "heat", "histogram"] = "landscape",
        **params: Any,
    ) -> None:
        """Initialize the vectorization transformer.

        :param method: Vectorization method: ``"image"`` (persistence image),
            ``"landscape"`` (persistence landscape), ``"silhouette"`` (persistence
            silhouette), ``"heat"`` (heat kernel signature), or ``"histogram"``
            (birth-death curve).
        :param params: Additional keyword arguments forwarded to the vectorization function.
        """
        self.method = method
        if method not in _VECTORIZATION_METHODS:
            raise ValueError(f"Unknown vectorization method: {method}")
        self.params = params

    def _get_vectorization_fn(self) -> Callable[..., Any]:
        fn = _VECTORIZATION_METHODS.get(self.method)
        if fn is None:
            raise ValueError(f"Unknown vectorization method: {self.method}")
        return fn

    def fit(self, X: Sequence[Any], y: Any = None) -> VectorizationTransformer:  # noqa: ARG002
        """Validate input without computation.

        :param X: Input diagrams.
        :param y: Ignored.
        :returns: *self*.
        """
        del y
        return self

    def transform(self, X: Sequence[Any]) -> np.ndarray:
        """Transform diagrams to fixed-size vector representations.

        :param X: Sequence of :class:`PersistenceDiagram` objects or diagram tensors.
        :returns: Stacked NumPy array of vectorized representations.
        :raises ValueError: If ``X`` is empty or a diagram is invalid.
        """
        _require_non_empty("X", X)
        fn = self._get_vectorization_fn()
        results = []
        for diagram in X:
            if isinstance(diagram, PersistenceDiagram):
                vec = fn(diagram, **self.params)
            else:
                vec = fn(diagram, **self.params)
            results.append(vec)
        return np.stack(results)

    def fit_transform(self, X: Sequence[Any], y: Any = None, **fit_params: Any) -> np.ndarray:
        """Fit and transform in one call.

        :param X: Input diagrams.
        :param y: Ignored.
        :param \\**fit_params: Ignored; for sklearn compatibility.
        :returns: Stacked NumPy array of vectorized representations.
        """
        del y
        del fit_params
        return self.transform(X)


class StatisticsTransformer(BaseEstimator, TransformerMixin):
    """Extract topological statistics from persistence diagrams."""

    def __init__(
        self,
        dims: list[int] | None = None,
        features: list[str] | None = None,
        track_stats: list[str] | None = None,
    ) -> None:
        """Initialize the statistics transformer.

        :param dims: Homology dimensions to extract statistics for (default: ``[0, 1]``).
        :param features: Subset of statistic names to track. If empty, all available
            statistics are used.
        :param track_stats: Statistics to track: ``"total_persistence"``,
            ``"num_features"``, ``"mean_lifetime"``, ``"max_lifetime"``.
        """
        self.dims = dims or [0, 1]
        self.features = features or []
        self.track_stats = track_stats or [
            "total_persistence",
            "num_features",
            "mean_lifetime",
            "max_lifetime",
        ]

    def fit(self, X: Sequence[Any], y: Any = None) -> StatisticsTransformer:  # noqa: ARG002
        """Validate input without computation.

        :param X: Input diagrams.
        :param y: Ignored.
        :returns: *self*.
        """
        del y
        return self

    def transform(self, X: Sequence[Any]) -> np.ndarray:
        """Extract topological statistics from diagrams.

        :param X: Sequence of :class:`PersistenceDiagram` objects or diagram tensors.
        :returns: NumPy array of shape ``(len(X), len(track_stats))``.
        :raises ValueError: If ``X`` is empty.
        """
        _require_non_empty("X", X)
        features_list = []
        for diagram in X:
            if isinstance(diagram, PersistenceDiagram):
                d = diagram
            else:
                d = _diagram_from_tensor(diagram, max(dim for dim in self.dims))
            stats = all_statistics(d.diagrams, dims=self.dims)
            row = [stats.get(name, 0.0) for name in self.track_stats]
            features_list.append(row)
        return np.array(features_list)

    def fit_transform(self, X: Sequence[Any], y: Any = None, **fit_params: Any) -> np.ndarray:
        """Fit and transform in one call.

        :param X: Input diagrams.
        :param y: Ignored.
        :param \\**fit_params: Ignored; for sklearn compatibility.
        :returns: NumPy array of statistical features.
        """
        del y
        del fit_params
        return self.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return feature names for the output columns.

        :param input_features: Ignored; for sklearn compatibility.
        :returns: Array of statistic feature names.
        """
        del input_features
        if torch is not None:
            feature_probe = torch.zeros((10, 3))
            feature_probe[:, 1] = torch.linspace(0, 1, 10)
            stats_dict = all_statistics(feature_probe, dims=self.dims)
            if self.features:
                stats_dict = {
                    k: v for k, v in stats_dict.items() if any(f in k for f in self.features)
                }
            return np.array(list(stats_dict.keys()))
        return np.array(
            [
                name
                for name in ["total_persistence", "num_features", "mean_lifetime", "max_lifetime"]
                if name in self.track_stats
            ]
        )


from ._sklearn_pipeline import (  # noqa: E402
    PersistencePipeline,
    make_tda_pipeline,
    persistence_train_test_split,
)

__all__ = [
    "BatchedTransformer",
    "PersistencePipeline",
    "PersistenceTransformer",
    "StatisticsTransformer",
    "VectorizationTransformer",
    "make_tda_pipeline",
    "persistence_train_test_split",
]
