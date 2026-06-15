"""Preprocessing utilities for persistence diagrams."""

from __future__ import annotations

from typing import Any, Literal

import torch

from .._constants import EPS
from .._validation import (
    validate_nonnegative_finite as _validate_nonnegative_finite,  # noqa: PLC0415
)
from ._vectorization_basis import (
    _validate_diagram,
    _validate_positive_finite,
    _validate_range,
)

_SUPPORTED_INF_STRATEGIES = {"max", "remove", "large_value"}
_SUPPORTED_NORMALIZATION_METHODS = {"minmax", "standard", "none"}
_SUPPORTED_SUBSAMPLE_STRATEGIES = {"most_persistent", "uniform", "random"}
_SUPPORTED_OUTLIER_METHODS = {"iqr", "zscore", "isolation_forest"}


def _validate_nonnegative_int(value: int, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_finite_deaths(diagram: torch.Tensor, name: str) -> None:
    if diagram.numel() > 0 and not torch.isfinite(diagram[:, 1]).all().item():
        raise ValueError(f"{name} requires finite deaths")


def _pad_diagrams(diagrams: list[torch.Tensor]) -> torch.Tensor:
    max_rows = max((diagram.shape[0] for diagram in diagrams), default=0)
    if max_rows == 0:
        return torch.zeros(
            (len(diagrams), 0, diagrams[0].shape[-1]),
            dtype=diagrams[0].dtype,
            device=diagrams[0].device,
        )
    padded = []
    for diagram in diagrams:
        if diagram.shape[0] == max_rows:
            padded.append(diagram)
            continue
        padding = torch.zeros(
            (max_rows - diagram.shape[0], diagram.shape[1]),
            dtype=diagram.dtype,
            device=diagram.device,
        )
        padded.append(torch.cat([diagram, padding], dim=0))
    return torch.stack(padded, dim=0)


def handle_infinite_deaths(
    diagram: torch.Tensor,
    strategy: Literal["max", "remove", "large_value"] = "max",
    large_value_factor: float = 2.0,
) -> torch.Tensor:
    """Replace or remove infinite death times.

    Supports batched input (3D): returns batched output.

    :param diagram: Persistence diagram tensor of shape ``(N, C)`` or ``(B, N, C)``.
    :param strategy: How to handle infinite deaths (``"max"``, ``"remove"``,
        ``"large_value"``).
    :param large_value_factor: Multiplier applied to the maximum finite
        death (``"max"`` strategy) or a literal replacement value
        (``"large_value"`` strategy).
    :returns: Diagram tensor with finite deaths.
    :raises ValueError: If ``strategy`` is unsupported or
        ``large_value_factor`` is too small.
    """
    if diagram.dim() == 3:
        return torch.stack(
            [
                handle_infinite_deaths(diagram[i], strategy, large_value_factor)
                for i in range(diagram.shape[0])
            ]
        )
    diagram = _validate_diagram(diagram)
    if strategy not in _SUPPORTED_INF_STRATEGIES:
        raise ValueError("strategy must be 'max', 'remove', or 'large_value'")
    large_value_factor = _validate_positive_finite(large_value_factor, "large_value_factor")
    births = diagram[:, 0]
    deaths = diagram[:, 1]

    infinite_mask = ~torch.isfinite(deaths)

    if not infinite_mask.any():
        return diagram.clone()

    if strategy == "remove":
        finite_mask = torch.isfinite(deaths)
        return diagram[finite_mask]

    if strategy == "max":
        finite_deaths = deaths[~infinite_mask]
        max_finite = finite_deaths.max() if len(finite_deaths) > 0 else births.max() * 2

        new_deaths = deaths.clone()
        replacement = max_finite * large_value_factor
        replacement = torch.maximum(replacement, births[infinite_mask].max())
        new_deaths[infinite_mask] = replacement

        result = diagram.clone()
        result[:, 1] = new_deaths
        return result

    if strategy == "large_value":
        replacement = torch.as_tensor(
            large_value_factor, dtype=diagram.dtype, device=diagram.device
        )
        min_birth = births[infinite_mask].min()
        if (replacement < births[infinite_mask]).any().item():
            raise ValueError(
                f"large_value_factor ({large_value_factor}) must be >= all births with "
                f"infinite deaths (min birth: {min_birth.item():.4f})"
            )
        new_deaths = deaths.clone()
        new_deaths[infinite_mask] = replacement

        result = diagram.clone()
        result[:, 1] = new_deaths
        return result

    raise ValueError("strategy must be 'max', 'remove', or 'large_value'")


def threshold_diagram(
    diagram: torch.Tensor,
    min_persistence: float = 0.0,
    max_persistence: float | None = None,
) -> torch.Tensor:
    """Keep features whose persistence lies inside the requested range.

    :param diagram: Persistence diagram tensor of shape ``(N, C)``.
    :param min_persistence: Minimum allowed persistence (non-negative).
    :param max_persistence: Optional maximum persistence (must be >= min).
    :returns: Filtered diagram tensor.
    :raises ValueError: If thresholds are invalid or out of order.
    """
    if diagram.dim() == 3:
        results = [
            threshold_diagram(diagram[i], min_persistence, max_persistence)
            for i in range(diagram.shape[0])
        ]
        return _pad_diagrams(results)
    diagram = _validate_diagram(diagram)
    min_persistence = _validate_nonnegative_finite(min_persistence, "min_persistence")
    if max_persistence is not None:
        max_persistence = _validate_nonnegative_finite(max_persistence, "max_persistence")
        if max_persistence < min_persistence:
            raise ValueError("max_persistence must be greater than or equal to min_persistence")
    births = diagram[:, 0]
    deaths = diagram[:, 1]

    persistence = deaths - births

    mask = persistence >= min_persistence

    if max_persistence is not None:
        mask = mask & (persistence <= max_persistence)

    return diagram[mask]


def normalize_diagram(
    diagram: torch.Tensor,
    method: Literal["minmax", "standard", "none"] = "minmax",
    birth_range: tuple[Any, ...] | None = None,
    death_range: tuple[Any, ...] | None = None,
) -> torch.Tensor:
    """Normalize birth and death coordinates.

    :param diagram: Persistence diagram tensor of shape ``(N, C)``.
    :param method: Normalisation method (``"minmax"``, ``"standard"``,
        ``"none"``).
    :param birth_range: Optional ``(min, max)`` for min-max birth scaling.
    :param death_range: Optional ``(min, max)`` for min-max death scaling.
    :returns: Normalised diagram tensor.
    :raises ValueError: If ``method`` is unsupported or diagram data is
        invalid.
    """
    if diagram.dim() == 3:
        return torch.stack(
            [
                normalize_diagram(diagram[i], method, birth_range, death_range)
                for i in range(diagram.shape[0])
            ]
        )
    diagram = _validate_diagram(diagram)
    if method not in _SUPPORTED_NORMALIZATION_METHODS:
        raise ValueError("method must be 'minmax', 'standard', or 'none'")
    if method == "none":
        return diagram.clone()
    if diagram.numel() == 0:
        return diagram.clone()
    _validate_finite_deaths(diagram, "normalize_diagram")
    birth_range = _validate_range(birth_range, "birth_range")
    death_range = _validate_range(death_range, "death_range")

    result = diagram.clone()
    births = result[:, 0]
    deaths = result[:, 1]

    if method == "minmax":
        if birth_range is None:
            b_min = births.min()
            b_max = births.max()
        else:
            b_min = torch.as_tensor(birth_range[0], dtype=diagram.dtype, device=diagram.device)
            b_max = torch.as_tensor(birth_range[1], dtype=diagram.dtype, device=diagram.device)

        if death_range is None:
            d_min = deaths.min()
            d_max = deaths.max()
        else:
            d_min = torch.as_tensor(death_range[0], dtype=diagram.dtype, device=diagram.device)
            d_max = torch.as_tensor(death_range[1], dtype=diagram.dtype, device=diagram.device)

        b_scale = torch.as_tensor(
            b_max - b_min, dtype=diagram.dtype, device=diagram.device
        ).clamp_min(EPS)
        d_scale = torch.as_tensor(
            d_max - d_min, dtype=diagram.dtype, device=diagram.device
        ).clamp_min(EPS)

        result[:, 0] = (births - b_min) / b_scale
        result[:, 1] = (deaths - d_min) / d_scale

    elif method == "standard":
        b_mean, b_std = births.mean(), births.std(unbiased=False)
        d_mean, d_std = deaths.mean(), deaths.std(unbiased=False)

        b_std = b_std.clamp_min(EPS)
        d_std = d_std.clamp_min(EPS)

        result[:, 0] = (births - b_mean) / b_std
        result[:, 1] = (deaths - d_mean) / d_std
    return result


def subsample_diagram(
    diagram: torch.Tensor,
    max_features: int,
    strategy: Literal["most_persistent", "uniform", "random"] = "most_persistent",
) -> torch.Tensor:
    """Keep at most ``max_features`` rows from a diagram.

    :param diagram: Persistence diagram tensor of shape ``(N, C)``.
    :param max_features: Maximum number of features to retain.
    :param strategy: Subsampling strategy (``"most_persistent"``,
        ``"uniform"``, ``"random"``).
    :returns: Subsampled diagram tensor.
    :raises ValueError: If ``strategy`` is unsupported or parameters are
        invalid.
    """
    if diagram.dim() == 3:
        results = [
            subsample_diagram(diagram[i], max_features, strategy) for i in range(diagram.shape[0])
        ]
        return _pad_diagrams(results)
    diagram = _validate_diagram(diagram)
    max_features = _validate_nonnegative_int(max_features, "max_features")
    if strategy not in _SUPPORTED_SUBSAMPLE_STRATEGIES:
        raise ValueError("strategy must be 'most_persistent', 'uniform', or 'random'")
    n = diagram.shape[0]

    if n <= max_features:
        return diagram.clone()

    if strategy == "most_persistent":
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistence = deaths - births

        persistence = torch.where(
            torch.isfinite(persistence),
            persistence,
            torch.full_like(persistence, float("inf")),
        )

        _, indices = torch.topk(persistence, max_features)
        return diagram[indices]

    if strategy == "uniform":
        births = diagram[:, 0]
        b_min, b_max = births.min(), births.max()

        bin_edges = torch.linspace(
            b_min.item(),
            b_max.item(),
            max_features + 1,
            dtype=diagram.dtype,
            device=diagram.device,
        )

        selected = []
        for i in range(max_features):
            mask = (births >= bin_edges[i]) & (births < bin_edges[i + 1])
            indices = torch.where(mask)[0]
            if len(indices) > 0:
                bin_persistence = diagram[indices, 1] - diagram[indices, 0]
                selected.append(indices[bin_persistence.argmax()])

        if len(selected) == 0:
            return diagram[:max_features]

        return diagram[torch.stack(selected)]

    if strategy == "random":
        indices = torch.randperm(n, device=diagram.device)[:max_features]
        return diagram[indices]

    raise ValueError("strategy must be 'most_persistent', 'uniform', or 'random'")


def _remove_outliers_iqr(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> torch.Tensor:
    finite = persistence[torch.isfinite(persistence)]
    if len(finite) == 0:
        return diagram.clone()
    q1 = torch.quantile(finite, 0.25)
    q3 = torch.quantile(finite, 0.75)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    outlier_mask = ((persistence < lower) | (persistence > upper)) & torch.isfinite(persistence)
    return diagram[~outlier_mask]


def _remove_outliers_zscore(
    persistence: torch.Tensor, diagram: torch.Tensor, threshold: float
) -> torch.Tensor:
    finite = persistence[torch.isfinite(persistence)]
    if len(finite) == 0:
        return diagram.clone()
    mean = finite.mean()
    std = finite.std(unbiased=False)
    if std < EPS:
        return diagram.clone()
    outlier_mask = (torch.abs((persistence - mean) / std) > threshold) & torch.isfinite(persistence)
    return diagram[~outlier_mask]


def _remove_outliers_isolation_forest(diagram: torch.Tensor, threshold: float) -> torch.Tensor:
    if not torch.isfinite(diagram[:, :2]).all().item():
        return _remove_outliers_iqr(diagram[:, 1] - diagram[:, 0], diagram, threshold)
    try:
        from sklearn.ensemble import IsolationForest  # noqa: PLC0415

        X = diagram[:, :2].detach().cpu().numpy()  # noqa: N806
        clf = IsolationForest(contamination=0.1, random_state=42)  # pyright: ignore[reportArgumentType]
        is_inlier = clf.fit_predict(X) == 1
        return diagram[torch.as_tensor(is_inlier, dtype=torch.bool, device=diagram.device)]
    except ImportError:
        return _remove_outliers_iqr(diagram[:, 1] - diagram[:, 0], diagram, threshold)


_REMOVE_OUTLIERS_DISPATCH = {
    "iqr": lambda d, t: _remove_outliers_iqr(d[:, 1] - d[:, 0], d, t),
    "zscore": lambda d, t: _remove_outliers_zscore(d[:, 1] - d[:, 0], d, t),
    "isolation_forest": _remove_outliers_isolation_forest,
}


def remove_outliers(
    diagram: torch.Tensor,
    method: Literal["iqr", "zscore", "isolation_forest"] = "iqr",
    threshold: float = 1.5,
) -> torch.Tensor:
    """Remove outlier features based on persistence.

    :param diagram: Persistence diagram tensor of shape ``(N, C)``.
    :param method: Outlier detection method (``"iqr"``, ``"zscore"``,
        ``"isolation_forest"``).
    :param threshold: Detection threshold (non-negative).
    :returns: Diagram tensor with outliers removed.
    :raises ValueError: If ``method`` is unsupported or ``threshold`` is
        invalid.
    """
    diagram = _validate_diagram(diagram)
    if method not in _SUPPORTED_OUTLIER_METHODS:
        raise ValueError("method must be 'iqr', 'zscore', or 'isolation_forest'")
    threshold = _validate_nonnegative_finite(threshold, "threshold")
    return _REMOVE_OUTLIERS_DISPATCH[method](diagram, threshold)


def clean_diagram(
    diagram: torch.Tensor,
    handle_inf: bool = True,
    min_persistence: float = 0.0,
    max_features: int | None = None,
    normalize: bool = True,
    remove_outliers_flag: bool = False,
) -> torch.Tensor:
    """Apply the standard diagram cleaning pipeline.

    Handles infinite deaths, thresholds by persistence, removes outliers,
    subsamples, and normalises coordinates.  Batched (3D) diagrams are
    processed element-wise.

    :param diagram: Persistence diagram tensor of shape ``(N, C)`` or
        ``(B, N, C)``.
    :param handle_inf: If ``True``, replace infinite deaths with the max
        finite death.
    :param min_persistence: Minimum persistence threshold (non-negative).
    :param max_features: Optional cap on the number of features to retain.
    :param normalize: If ``True``, apply min-max normalisation.
    :param remove_outliers_flag: If ``True``, remove outliers via IQR.
    :returns: Cleaned diagram tensor.
    :raises ValueError: If the diagram shape, ``min_persistence``, or
        ``max_features`` is invalid.
    """
    if diagram.dim() == 3:
        if diagram.shape[0] == 0:
            return diagram.clone()
        return _pad_diagrams(
            [
                clean_diagram(
                    item,
                    handle_inf=handle_inf,
                    min_persistence=min_persistence,
                    max_features=max_features,
                    normalize=normalize,
                    remove_outliers_flag=remove_outliers_flag,
                )
                for item in diagram
            ]
        )
    if diagram.dim() != 2:
        raise ValueError("diagram must be a 2D or 3D tensor")
    diagram = _validate_diagram(diagram)
    min_persistence = _validate_nonnegative_finite(min_persistence, "min_persistence")
    if max_features is not None:
        max_features = _validate_nonnegative_int(max_features, "max_features")

    result = diagram.clone()

    if handle_inf:
        result = handle_infinite_deaths(result, strategy="max")

    if min_persistence > 0:
        result = threshold_diagram(result, min_persistence=min_persistence)

    if remove_outliers_flag:
        result = remove_outliers(result, method="iqr")

    if max_features is not None:
        result = subsample_diagram(result, max_features)

    if normalize:
        result = normalize_diagram(result, method="minmax")

    return result


__all__ = [
    "handle_infinite_deaths",
    "threshold_diagram",
    "normalize_diagram",
    "subsample_diagram",
    "remove_outliers",
    "clean_diagram",
]
