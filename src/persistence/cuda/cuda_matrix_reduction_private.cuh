#pragma once

#include "nerve/persistence/cuda/cuda_matrix_reduction.hpp"

#include <cuda_runtime.h>

namespace nerve::persistence::accelerated
{

__global__ void matrixReductionAcceleratedKernel(const int *__restrict__ columns,
                                                 const Size *__restrict__ column_sizes,
                                                 const double *__restrict__ weights,
                                                 int *__restrict__ low_row_to_col,
                                                 int *__restrict__ col_pivot, Size n_columns,
                                                 Size max_dim, bool use_clearing);

__global__ void matrixReductionStreamingKernel(const int *__restrict__ columns,
                                               const Size *__restrict__ column_sizes,
                                               const double *__restrict__ weights,
                                               int *__restrict__ low_row_to_col,
                                               int *__restrict__ col_pivot, Size n_columns,
                                               Size max_dim, bool use_clearing, Size chunk_size,
                                               Size chunk_offset);

__global__ void computeApparentPairsKernel(const int *__restrict__ low_row_to_col,
                                           const int *__restrict__ col_pivot,
                                           const double *__restrict__ weights, Size n_columns,
                                           int *__restrict__ pair_count,
                                           int *__restrict__ pair_values, Size max_pairs,
                                           bool use_optimization);

__global__ void hybridMatrixReductionKernel(const int *__restrict__ columns,
                                            const Size *__restrict__ column_sizes,
                                            const double *__restrict__ weights,
                                            int *__restrict__ low_row_to_col,
                                            int *__restrict__ col_pivot, Size n_columns,
                                            Size max_dim, Size gpu_columns, bool use_clearing);

errors::ErrorResult<void>
launchMatrixReductionKernel(const int *d_columns, const Size *d_column_sizes,
                            const double *d_weights, int *d_low_row_to_col, int *d_col_pivot,
                            Size n_columns, Size max_dim, const MatrixReductionConfig &config);

} // namespace nerve::persistence::accelerated
