"""Merge and match multiple persistence diagrams."""

from __future__ import annotations

import numpy as np

from ._validation import validate_diagram_array


def match_persistence_diagrams(
    diagrams: list[np.ndarray],
    threshold: float = float("inf"),
    dim: int | None = None,
) -> list[np.ndarray]:
    """Merge multiple persistence diagrams via bottleneck matching.

    Each diagram is matched to a chosen reference diagram (the first
    in the list). Unmatched pairs beyond *threshold* are discarded.

    Args:
        diagrams: List of diagram arrays, each of shape ``(n_i, >=2)``.
        threshold: Bottleneck distance threshold for matching.
        dim: If given, only match pairs of this dimension.

    Returns:
        List of matched diagram arrays, one per input diagram.
    """
    if len(diagrams) < 2:
        return list(diagrams)

    validated = [validate_diagram_array(d, require_dims=False) for d in diagrams]

    if dim is not None:
        validated = [_filter_by_dim(d, dim) for d in validated]

    ref = validated[0]
    matched: list[np.ndarray] = [ref]

    for diag in validated[1:]:
        matched_diag = _bottleneck_match(ref, diag, threshold)
        matched.append(matched_diag)

    return matched


def _filter_by_dim(diagram: np.ndarray, dim: int) -> np.ndarray:
    if diagram.shape[1] < 3:
        return diagram
    mask = diagram[:, 2].astype(int) == dim
    return diagram[mask]


def _bottleneck_match(ref: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    ref_finite = ref[np.isfinite(ref[:, 1]) & (ref[:, 1] > ref[:, 0])]
    tgt_finite = target[np.isfinite(target[:, 1]) & (target[:, 1] > target[:, 0])]

    if len(ref_finite) == 0:
        return tgt_finite
    if len(tgt_finite) == 0:
        return ref_finite

    cost = np.zeros((len(ref_finite), len(tgt_finite)))
    for i in range(len(ref_finite)):
        for j in range(len(tgt_finite)):
            cost[i, j] = max(
                abs(ref_finite[i, 0] - tgt_finite[j, 0]),
                abs(ref_finite[i, 1] - tgt_finite[j, 1]),
            )

    matched_pairs = []
    col_used = np.zeros(len(tgt_finite), dtype=bool)
    for i in range(len(ref_finite)):
        min_cost = np.inf
        best_j = -1
        for j in range(len(tgt_finite)):
            if not col_used[j] and cost[i, j] < min_cost:
                min_cost = cost[i, j]
                best_j = j
        if best_j >= 0 and min_cost <= threshold:
            col_used[best_j] = True
            matched_pairs.append(tgt_finite[best_j])

    if matched_pairs:
        return np.array(matched_pairs)
    return np.empty((0, ref_finite.shape[1]))


__all__ = [
    "match_persistence_diagrams",
]
