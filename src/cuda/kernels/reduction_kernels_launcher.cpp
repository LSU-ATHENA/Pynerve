#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

extern __global__ void parallelColumnReductionKernel(const int *__restrict__ column_data,
                                                     const int *__restrict__ column_indices,
                                                     const int *__restrict__ column_starts,
                                                     int *__restrict__ pivots,
                                                     int *__restrict__ pairings, int n_columns,
                                                     int max_column_height);

extern __global__ void sparseColumnAddScaledKernel(
    int *__restrict__ col_j_data, int *__restrict__ col_j_indices, int *__restrict__ col_j_starts,
    const int *__restrict__ col_i_data, const int *__restrict__ col_i_indices,
    const int *__restrict__ col_i_starts, int scalar, int col_i, int col_j, int max_height);

extern __global__ void cochainReductionKernel(const int *__restrict__ coboundary_data,
                                              const int *__restrict__ coboundary_indices,
                                              const int *__restrict__ coboundary_starts,
                                              int *__restrict__ pivots, int *__restrict__ processed,
                                              int n_cochains, int max_height);

extern __global__ void symmetricDifferenceKernel(const int *__restrict__ col_a_data, int col_a_size,
                                                 const int *__restrict__ col_b_data, int col_b_size,
                                                 int *__restrict__ out_result,
                                                 int *__restrict__ out_count, int max_result_size);

extern __global__ void detectBirthSimplicesKernel(
    const int *__restrict__ boundary_data, const int *__restrict__ boundary_indices,
    const int *__restrict__ boundary_starts, const int *__restrict__ simplex_dimensions,
    bool *__restrict__ out_cleared, int *__restrict__ out_count, int n_simplices, int max_height);

extern __global__ void detectEmergentPairsKernel(const int *__restrict__ boundary_data,
                                                 const int *__restrict__ boundary_indices,
                                                 const int *__restrict__ boundary_starts,
                                                 bool *__restrict__ out_emergent,
                                                 int *__restrict__ out_count, int n_simplices,
                                                 int max_height);

namespace nerve
{
namespace gpu
{
namespace detail
{

constexpr int kReductionBlockSize = 256;
constexpr int kReductionSingleBlock = 1;

int reductionGridSize(int count)
{
    return (count + kReductionBlockSize - 1) / kReductionBlockSize;
}

void launchParallelColumnReduction(const int *column_data, const int *column_indices,
                                   const int *column_starts, int *pivots, int *pairings,
                                   int n_columns, int max_column_height, cudaStream_t stream)
{
    if (n_columns <= 0 || max_column_height <= 0)
        return;

    parallelColumnReductionKernel<<<reductionGridSize(n_columns), kReductionBlockSize, 0, stream>>>(
        column_data, column_indices, column_starts, pivots, pairings, n_columns, max_column_height);
    GPU_CHECK(cudaGetLastError());
}

void launchSparseColumnAdd(int *col_j_data, int *col_j_indices, const int *col_i_data,
                           const int *col_i_indices, int col_i, int col_j, int max_height,
                           cudaStream_t stream)
{
    if (max_height <= 0 || col_i < 0 || col_j < 0)
        return;

    sparseColumnAddScaledKernel<<<kReductionSingleBlock, kReductionBlockSize, 0, stream>>>(
        col_j_data, col_j_indices, nullptr, col_i_data, col_i_indices, nullptr, 1, col_i, col_j,
        max_height);
    GPU_CHECK(cudaGetLastError());
}

void launchCochainReduction(const int *coboundary_data, const int *coboundary_indices,
                            const int *coboundary_starts, int *pivots, int *processed,
                            int n_cochains, int max_height, cudaStream_t stream)
{
    if (n_cochains <= 0 || max_height <= 0)
        return;

    cochainReductionKernel<<<reductionGridSize(n_cochains), kReductionBlockSize, 0, stream>>>(
        coboundary_data, coboundary_indices, coboundary_starts, pivots, processed, n_cochains,
        max_height);
    GPU_CHECK(cudaGetLastError());
}

void launchSymmetricDifference(const int *col_a_data, int col_a_size, const int *col_b_data,
                               int col_b_size, int *out_result, int *out_count, int max_result_size,
                               cudaStream_t stream)
{
    if (out_count == nullptr)
        return;
    if (out_result == nullptr || col_a_size < 0 || col_b_size < 0 || max_result_size <= 0 ||
        col_a_size + col_b_size == 0)
    {
        GPU_CHECK(cudaMemsetAsync(out_count, 0, sizeof(int), stream));
        return;
    }

    symmetricDifferenceKernel<<<kReductionSingleBlock, kReductionBlockSize, 0, stream>>>(
        col_a_data, col_a_size, col_b_data, col_b_size, out_result, out_count, max_result_size);
    GPU_CHECK(cudaGetLastError());
}

void launchDetectBirthSimplices(const int *boundary_data, const int *boundary_indices,
                                const int *boundary_starts, const int *simplex_dimensions,
                                bool *out_cleared, int *out_count, int n_simplices, int max_height,
                                cudaStream_t stream)
{
    if (n_simplices <= 0)
        return;

    detectBirthSimplicesKernel<<<reductionGridSize(n_simplices), kReductionBlockSize, 0, stream>>>(
        boundary_data, boundary_indices, boundary_starts, simplex_dimensions, out_cleared,
        out_count, n_simplices, max_height);
    GPU_CHECK(cudaGetLastError());
}

void launchDetectEmergentPairs(const int *boundary_data, const int *boundary_indices,
                               const int *boundary_starts, bool *out_emergent, int *out_count,
                               int n_simplices, int max_height, cudaStream_t stream)
{
    if (n_simplices <= 0)
        return;

    detectEmergentPairsKernel<<<reductionGridSize(n_simplices), kReductionBlockSize, 0, stream>>>(
        boundary_data, boundary_indices, boundary_starts, out_emergent, out_count, n_simplices,
        max_height);
    GPU_CHECK(cudaGetLastError());
}

} // namespace detail
} // namespace gpu
} // namespace nerve
