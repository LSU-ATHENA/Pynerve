"""Sparse boundary matrix helpers."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._fast_simplices import simplex_boundary_fast


def boundary_matrix_sparse(simplices: list[np.ndarray], max_dim: int = 2) -> list[csr_matrix]:
    boundary_matrices = []
    max_dim = min(max_dim, len(simplices) - 1)

    for d in range(1, max_dim + 1):
        lower_simplices = simplices[d - 1]
        higher_simplices = simplices[d]
        lower_index = {
            tuple(sorted(int(v) for v in simplex)): idx
            for idx, simplex in enumerate(lower_simplices)
        }

        rows, cols, data = [], [], []

        for j, simplex in enumerate(higher_simplices):
            boundary = simplex_boundary_fast(simplex)

            for i, face in enumerate(boundary):
                key = tuple(sorted(int(v) for v in face))
                if key in lower_index:
                    rows.append(lower_index[key])
                    cols.append(j)
                    data.append((-1) ** i)

        mat = csr_matrix((data, (rows, cols)), shape=(len(lower_simplices), len(higher_simplices)))
        boundary_matrices.append(mat)

    return boundary_matrices


def column_reduction_sparse(
    boundary_matrix: csr_matrix,
    filtration_values: np.ndarray,
    row_filtration_values: np.ndarray | None = None,
) -> list[tuple[int, int, float, float]]:
    n_cols = boundary_matrix.shape[1]  # type: ignore[optional]
    pivot_rows: dict[int, int] = {}
    persistence_pairs: list[tuple[int, int, float, float]] = []

    mat = boundary_matrix.tocsc()

    for j in range(n_cols):
        col = mat[:, j].nonzero()[0]
        col_set = set(col.tolist())

        while col_set:
            pivot = max(col_set)
            if pivot in pivot_rows:
                pivot_col = pivot_rows[pivot]
                pivot_col_set = set(mat[:, pivot_col].nonzero()[0].tolist())
                col_set.symmetric_difference_update(pivot_col_set)
            else:
                break

        if col_set:
            pivot = max(col_set)
            pivot_rows[pivot] = j
            birth_time = (
                row_filtration_values[pivot]
                if row_filtration_values is not None and pivot < len(row_filtration_values)
                else 0.0
            )
            death_time = filtration_values[j] if j < len(filtration_values) else birth_time
            persistence_pairs.append((pivot, j, birth_time, death_time))  # pyright: ignore[reportArgumentType]

    return persistence_pairs
