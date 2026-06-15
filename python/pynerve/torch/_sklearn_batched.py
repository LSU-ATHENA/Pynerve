"""Batched sklearn transformer wrapper - isolated from TDA dependencies."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from ._sklearn_compat import BaseEstimator, TransformerMixin, _require_non_empty


def _validation_error(msg: str, param: str | None = None) -> None:
    from pynerve.exceptions import ValidationError  # noqa: PLC0415

    raise ValidationError(msg, parameter=param)


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _validation_error(f"{name} must be an integer", param=name)
    result = int(value)
    if result <= 0:
        _validation_error(f"{name} must be positive", param=name)
    return result


class BatchedTransformer(BaseEstimator, TransformerMixin):
    """Wrapper for processing large datasets in batches."""

    def __init__(
        self, transformer: TransformerMixin, batch_size: int = 32, n_jobs: int = 1
    ) -> None:
        """Initialize the batched transformer.

        :param transformer: An sklearn-compatible transformer to wrap.
        :param batch_size: Number of samples per batch.
        :param n_jobs: Number of parallel workers (requires ``joblib`` if > 1).
        :raises ValidationError: If ``batch_size`` or ``n_jobs`` is not a positive integer.
        """
        batch_size = _validate_positive_int(batch_size, "batch_size")
        n_jobs = _validate_positive_int(n_jobs, "n_jobs")
        self.transformer = transformer
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def fit(self, X: list[Any], y: Any = None) -> BatchedTransformer:
        """Fit the wrapped transformer on a subset of the data.

        :param X: List of input samples.
        :param y: Optional target values.
        :returns: *self*.
        :raises ValueError: If ``X`` is empty.
        """
        _require_non_empty("X", X)
        batch = X[: min(self.batch_size, len(X))]
        self.transformer.fit(batch, y[: len(batch)] if y is not None else None)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def transform(self, X: list[Any]) -> np.ndarray | list[Any]:
        """Transform data in batches.

        :param X: List of input samples.
        :returns: NumPy array if the underlying transformer output is NumPy arrays,
            otherwise a concatenated list.
        :raises ValueError: If ``X`` is empty.
        :raises ImportError: If ``n_jobs`` > 1 and ``joblib`` is not installed.
        """
        _require_non_empty("X", X)
        n = len(X)
        batches = [X[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        if self.n_jobs == 1:
            results: list[Any] = [self.transformer.transform(batch) for batch in batches]  # pyright: ignore[reportAttributeAccessIssue]
        else:
            try:
                from joblib import Parallel, delayed  # noqa: PLC0415
            except ImportError:
                raise ImportError("joblib is required when n_jobs is greater than 1") from None

            results = list(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(self.transformer.transform)(batch)  # pyright: ignore[reportAttributeAccessIssue]
                    for batch in batches  # pyright: ignore[reportAttributeAccessIssue]
                )
            )

        if isinstance(results[0], np.ndarray):
            return np.vstack(results)
        return [item for sublist in results for item in sublist]

    def fit_transform(
        self,
        X: list[Any],
        y: Any = None,
        **fit_params: Any,  # noqa: ARG002
    ) -> np.ndarray | list[Any]:
        """Fit and transform in one call.

        :param X: List of input samples.
        :param y: Optional target values.
        :param \\**fit_params: Ignored; for sklearn compatibility.
        :returns: NumPy array or concatenated list of transformed outputs.
        """
        return self.fit(X, y).transform(X)
