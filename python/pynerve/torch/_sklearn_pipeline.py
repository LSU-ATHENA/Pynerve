"""Persistence pipeline compositions and train-test split utilities."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.pipeline import Pipeline

from .._validation import validate_finite_scalar as _validate_finite_scalar
from ._persistence_api import _validate_max_dim, _validate_max_radius
from .sklearn_transformers import (
    _COMPLEX_TYPES,
    _VECTORIZATION_METHODS,
    PersistenceTransformer,
    StatisticsTransformer,
    VectorizationTransformer,
    _validate_sequence,
)


class PersistencePipeline:
    """Composable persistence + vectorization pipeline."""

    def __init__(
        self,
        complex_type: Literal["vr", "alpha"] = "vr",
        max_dim: int = 1,
        max_radius: float = float("inf"),
        preprocessing: dict[str, Any] | None = None,
        vectorization: Literal[
            "image", "landscape", "silhouette", "heat", "histogram", "stats"
        ] = "landscape",
        vectorization_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the persistence pipeline.

        :param complex_type: Simplicial complex type: ``"vr"`` (Vietoris-Rips) or
            ``"alpha"`` (Alpha complex).
        :param max_dim: Maximum homology dimension.
        :param max_radius: Maximum filtration radius.
        :param preprocessing: Keyword arguments forwarded to :func:`clean_diagram`.
        :param vectorization: Vectorization method: ``"image"``, ``"landscape"``,
            ``"silhouette"``, ``"heat"``, ``"histogram"``, or ``"stats"``.
        :param vectorization_params: Keyword arguments forwarded to the vectorization
            method.
        :raises ValueError: If ``complex_type`` or ``vectorization`` is invalid.
        """
        max_dim = _validate_max_dim(max_dim)
        max_radius = _validate_max_radius(max_radius)
        if complex_type not in _COMPLEX_TYPES:
            raise ValueError(f"Unknown complex type: {complex_type}")
        if vectorization != "stats" and vectorization not in _VECTORIZATION_METHODS:
            raise ValueError(f"Unknown vectorization method: {vectorization}")
        self.complex_type = complex_type
        self.max_dim = max_dim
        self.max_radius = max_radius
        self.preprocessing = preprocessing or {}
        self.vectorization = vectorization
        self.vectorization_params = vectorization_params or {}

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        self.persistence_transformer = PersistenceTransformer(
            complex_type=self.complex_type,  # pyright: ignore[reportArgumentType]
            max_dim=self.max_dim,
            max_radius=self.max_radius,
            preprocessing_params=self.preprocessing,
        )

        if self.vectorization == "stats":
            self.vectorization_transformer: VectorizationTransformer | StatisticsTransformer = (
                StatisticsTransformer()
            )
        else:
            self.vectorization_transformer = VectorizationTransformer(
                method=self.vectorization,  # pyright: ignore[reportArgumentType]
                **self.vectorization_params,
            )

    def fit(self, X: list[Any], y: Any = None) -> PersistencePipeline:
        """Fit the persistence and vectorization stages.

        :param X: List of input point clouds.
        :param y: Ignored; for sklearn compatibility.
        :returns: *self*.
        """
        self.persistence_transformer.fit(X, y)
        diagrams = self.persistence_transformer.transform(X)
        self.vectorization_transformer.fit(diagrams, y)
        return self

    def transform(self, X: list[Any]) -> np.ndarray:
        """Transform point clouds to feature vectors.

        :param X: List of input point clouds.
        :returns: NumPy array of vectorized features.
        """
        diagrams = self.persistence_transformer.transform(X)
        return self.vectorization_transformer.transform(diagrams)

    def fit_transform(self, X: list[Any], y: Any = None) -> np.ndarray:
        """Fit and transform in one call.

        :param X: List of input point clouds.
        :param y: Ignored; for sklearn compatibility.
        :returns: NumPy array of vectorized features.
        """
        return self.fit(X, y).transform(X)


def make_tda_pipeline(
    classifier: Any = None,
    complex_type: str = "vr",
    max_dim: int = 1,
    vectorization: str = "landscape",
    preprocessing: dict[str, Any] | None = None,
) -> Pipeline:
    """Create a complete sklearn pipeline with TDA features.

    :param classifier: Optional sklearn classifier/regressor to append as the final step.
    :param complex_type: Simplicial complex type (``"vr"`` or ``"alpha"``).
    :param max_dim: Maximum homology dimension.
    :param vectorization: Vectorization method (``"image"``, ``"landscape"``,
        ``"silhouette"``, ``"heat"``, or ``"histogram"``).
    :param preprocessing: Keyword arguments forwarded to :func:`clean_diagram`.
    :returns: An sklearn ``Pipeline`` with TDA and vectorization steps.
    :raises ValueError: If ``complex_type`` or ``vectorization`` is invalid.
    """
    max_dim = _validate_max_dim(max_dim)
    if complex_type not in _COMPLEX_TYPES:
        raise ValueError(f"Unknown complex type: {complex_type}")
    if vectorization not in _VECTORIZATION_METHODS:
        raise ValueError(f"Unknown vectorization method: {vectorization}")

    steps: list[Any] = [
        (
            "tda",
            PersistenceTransformer(
                complex_type=complex_type,  # type: ignore[arg-type]
                max_dim=max_dim,
                preprocessing_params=preprocessing,
            ),
        ),
        (
            "vec",
            VectorizationTransformer(
                method=vectorization,  # type: ignore[arg-type]
            ),
        ),
    ]

    if classifier is not None:
        steps.append(("classifier", classifier))

    return Pipeline(steps)


def persistence_train_test_split(
    X: list[Any],  # noqa: N803
    y: list[Any],
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Split data while preserving list-based point-cloud inputs."""
    _validate_sequence("X", X)
    _validate_sequence("y", y)
    if len(X) != len(y):
        raise ValueError("X and y must have matching lengths")
    test_size_f: float = _validate_finite_scalar(test_size, "test_size")
    if not 0.0 < test_size_f < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    try:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
    except ImportError:
        raise ImportError("sklearn is required for persistence_train_test_split") from None

    n = len(X)
    indices = np.arange(n)

    indices_train, indices_test, y_train, y_test = train_test_split(
        indices, y, test_size=test_size_f, random_state=random_state
    )

    X_train = [X[i] for i in indices_train]  # noqa: N806
    X_test = [X[i] for i in indices_test]  # noqa: N806

    return X_train, X_test, y_train, y_test


__all__ = [
    "PersistencePipeline",
    "make_tda_pipeline",
    "persistence_train_test_split",
]
