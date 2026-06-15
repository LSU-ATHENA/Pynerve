"""Shared sklearn-compatibility fallbacks for the torch subpackage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from sklearn.base import (
        BaseEstimator,  # pyright: ignore[reportAssignmentType]
        TransformerMixin,  # pyright: ignore[reportAssignmentType]
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    class BaseEstimator:  # type: ignore[no-redef]
        """Minimal sklearn-compatible estimator base class."""

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            """Return estimator parameters.

            :param deep: Ignored; included for sklearn compatibility.
            :returns: Empty dict.
            """
            del deep
            return {}

        def set_params(self, **params: Any) -> BaseEstimator:
            """Set estimator parameters.

            :param \\**params: Keyword parameters to set as attributes.
            :returns: *self*.
            """
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class TransformerMixin:  # type: ignore[no-redef]
        """Minimal sklearn-compatible transformer mixin."""

        def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
            """Fit and transform in one call.

            :param X: Input data.
            :param y: Target values (ignored).
            :param \\**fit_params: Additional fit parameters.
            :returns: Transformed output.
            """
            return self.fit(X, y, **fit_params).transform(X)  # type: ignore[attr-defined]


def _require_non_empty(name: str, values: Sequence[Any]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")


__all__ = ["BaseEstimator", "TransformerMixin", "SKLEARN_AVAILABLE", "_require_non_empty"]
