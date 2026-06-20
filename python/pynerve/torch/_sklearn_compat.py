"""Shared sklearn-compatibility fallbacks for the torch subpackage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class _FallbackBaseEstimator:
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {}

    def set_params(self, **params: Any) -> _FallbackBaseEstimator:
        for key, value in params.items():
            setattr(self, key, value)
        return self


class _FallbackTransformerMixin:
    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        return self.fit(X, y, **fit_params).transform(X)  # type: ignore[attr-defined]


SKLEARN_AVAILABLE = False
SKLEARN_IMPORT_ERROR: str | None = None

try:
    from sklearn.base import BaseEstimator, TransformerMixin  # noqa: F811

    SKLEARN_AVAILABLE = True
except ImportError as exc:
    SKLEARN_IMPORT_ERROR = str(exc)
    BaseEstimator = _FallbackBaseEstimator  # type: ignore[misc,assignment]
    TransformerMixin = _FallbackTransformerMixin  # type: ignore[misc,assignment]


def _require_non_empty(name: str, values: Sequence[Any]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")


__all__ = [
    "BaseEstimator",
    "TransformerMixin",
    "SKLEARN_AVAILABLE",
    "SKLEARN_IMPORT_ERROR",
    "_FallbackBaseEstimator",
    "_FallbackTransformerMixin",
    "_require_non_empty",
]
