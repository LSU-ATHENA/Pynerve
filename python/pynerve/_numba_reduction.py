"""Numba boundary-reduction kernels."""

from __future__ import annotations

import numpy as np

from ._numba_compat import njit


@njit(cache=True)
def numba_column_reduction(boundary_matrix: np.ndarray, _filtration: np.ndarray) -> np.ndarray:
    n_lower, n_higher = boundary_matrix.shape
    pivot_of_row = np.full(n_lower, -1, dtype=np.int64)
    pivots = np.full(n_higher, -1, dtype=np.int64)

    for j in range(n_higher):
        lowest = -1
        for i in range(n_lower - 1, -1, -1):
            if boundary_matrix[i, j] == 1:
                lowest = i
                break

        while lowest >= 0 and pivot_of_row[lowest] >= 0:
            pivot_col = pivot_of_row[lowest]
            pivot_col_view = boundary_matrix[:, pivot_col]

            for i in range(n_lower):
                boundary_matrix[i, j] ^= pivot_col_view[i]

            lowest = -1
            for i in range(n_lower - 1, -1, -1):
                if boundary_matrix[i, j] == 1:
                    lowest = i
                    break

        if lowest >= 0:
            pivot_of_row[lowest] = j
            pivots[j] = lowest

    return pivots


@njit(cache=True)
def numba_sparse_reduction(
    columns: np.ndarray,
    col_lengths: np.ndarray,
) -> np.ndarray:
    n_cols = columns.shape[0]
    max_row = -1
    for j in range(n_cols):
        for k in range(col_lengths[j]):
            row = columns[j, k]
            max_row = max(max_row, row)

    pivot_col_of_row = np.full(max_row + 1 if max_row >= 0 else 1, -1, dtype=np.int64)
    pivots = np.full(n_cols, -1, dtype=np.int64)

    for j in range(n_cols):
        col_len = col_lengths[j]
        col_set = set(columns[j, :col_len])

        while len(col_set) > 0:
            lowest = max(col_set)

            if pivot_col_of_row[lowest] >= 0:
                pivot_j = pivot_col_of_row[lowest]
                pivot_len = col_lengths[pivot_j]
                pivot_set = set(columns[pivot_j, :pivot_len])

                col_set = col_set.symmetric_difference(pivot_set)
            else:
                break

        if len(col_set) > 0:
            lowest = max(col_set)
            pivot_col_of_row[lowest] = j
            pivots[j] = lowest

    return pivots
