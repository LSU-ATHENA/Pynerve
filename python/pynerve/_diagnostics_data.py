"""Data quality checking utilities."""

from __future__ import annotations

import os
from typing import Any, TypedDict

import numpy as np


class DataQualityReport(TypedDict):
    """Result of a data quality check.

    Keys:
        valid (bool): ``True`` if all checks passed.
        warnings (list[str]): Non-blocking issues found during checks.
        errors (list[str]): Critical issues that require fixing.
    """

    valid: bool
    warnings: list[str]
    errors: list[str]


def _extract_array(data: np.ndarray | Any) -> np.ndarray | None:
    if hasattr(data, "numel") and hasattr(data, "numpy"):
        if data.numel() == 0:  # pyright: ignore[reportAttributeAccessIssue]
            return None
        if hasattr(data, "is_cuda") and data.is_cuda:  # pyright: ignore[reportAttributeAccessIssue]
            data = data.cpu()  # type: ignore[union-attr]
        result = data.numpy()  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(result, np.ndarray):
            return result
        return None
    if isinstance(data, np.ndarray):
        return data
    return None


def _validate_shape_and_dtype(arr: np.ndarray, report: DataQualityReport) -> bool:
    if arr.ndim != 2:
        report["errors"].append(f"Expected 2D array [n, dim], got shape {arr.shape}")
        report["valid"] = False
        return False
    if arr.size == 0:
        report["errors"].append("Data must be non-empty")
        report["valid"] = False
        return False
    if not np.issubdtype(arr.dtype, np.number):
        report["errors"].append("Data must have a numeric dtype")
        report["valid"] = False
        return False
    return True


def _check_nan_inf(arr: np.ndarray, report: DataQualityReport) -> None:
    if np.isnan(arr).any():
        report["errors"].append("Data contains NaN values")
        report["valid"] = False
    if np.isinf(arr).any():
        report["warnings"].append("Data contains Inf values")


def _check_duplicates(arr: np.ndarray, report: DataQualityReport) -> None:
    unique = np.unique(arr, axis=0)
    if len(unique) < len(arr):
        n_dup = len(arr) - len(unique)
        report["warnings"].append(f"Data contains {n_dup} duplicate points")


def _check_variance_and_range(arr: np.ndarray, report: DataQualityReport) -> None:
    data_range = np.ptp(arr, axis=0)
    if np.any(data_range == 0):
        report["warnings"].append("Data has zero variance in some dimensions")

    max_range = np.max(data_range)
    _range_threshold = float(os.environ.get("NERVE_COORD_RANGE_WARN", "1e6"))
    if max_range > _range_threshold:
        report["warnings"].append(f"Large coordinate range ({max_range:.2e}), consider normalizing")


def _check_size(arr: np.ndarray, report: DataQualityReport) -> None:
    n_points = arr.shape[0] if arr.ndim == 2 else 0
    _npoint_threshold = int(os.environ.get("NERVE_NPOINTS_WARN", "50000"))
    if n_points > _npoint_threshold:
        report["warnings"].append(
            f"Large dataset ({n_points} points), consider subsampling or streaming"
        )


def check_data_quality(data: np.ndarray | Any) -> DataQualityReport:
    """Validate data quality before persistence computation.

    Checks for shape, dtype, NaN/Inf, duplicate points, zero-variance
    dimensions, large coordinate ranges (env ``NERVE_COORD_RANGE_WARN``,
    default 1e6), and large dataset size (env ``NERVE_NPOINTS_WARN``,
    default 50000).

    :param data: Input data as numpy array or torch Tensor.
    :returns: A :class:`DataQualityReport` with ``valid``, ``warnings``,
        and ``errors`` fields.
    """
    report: DataQualityReport = {"valid": True, "warnings": [], "errors": []}

    arr = _extract_array(data)
    if arr is None:
        if hasattr(data, "numel") and data.numel() == 0:  # pyright: ignore[reportAttributeAccessIssue]
            report["errors"].append("Data must be non-empty")
        else:
            report["errors"].append(f"Data must be numpy array or torch Tensor, got {type(data)}")
        report["valid"] = False
        return report

    if not _validate_shape_and_dtype(arr, report):
        return report

    _check_nan_inf(arr, report)
    _check_duplicates(arr, report)
    _check_variance_and_range(arr, report)
    _check_size(arr, report)

    return report
