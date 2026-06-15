#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>

namespace nerve::gpu::persistence::detail
{
namespace
{

constexpr int kClearingBlockSize = 256;

int gridFor(int items)
{
    if (items <= 0)
    {
        return 0;
    }
    return (items / kClearingBlockSize) + ((items % kClearingBlockSize) == 0 ? 0 : 1);
}

__global__ void __launch_bounds__(256)
    detectPositiveSimplicesKernel(const int *__restrict__ boundary_data,
                                  const int *__restrict__ boundary_indices,
                                  const int *__restrict__ boundary_starts,
                                  const int *__restrict__ simplex_dimensions,
                                  int *__restrict__ is_positive, int n_simplices)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
    {
        return;
    }

    if (simplex_dimensions[idx] == 0)
    {
        is_positive[idx] = 1;
        return;
    }

    const int start = boundary_starts[idx];
    const int end = boundary_starts[idx + 1];
    bool has_boundary = false;
    for (int i = start; i < end; ++i)
    {
        if (boundary_indices[i] >= 0 && boundary_data[i] != 0)
        {
            has_boundary = true;
            break;
        }
    }
    is_positive[idx] = has_boundary ? 0 : 1;
}

__global__ void __launch_bounds__(256)
    markClearingColumnsKernel(const int *__restrict__ is_positive,
                              const int *__restrict__ simplex_dimensions,
                              int *__restrict__ clear_column, int target_dimension, int n_simplices)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
    {
        return;
    }
    clear_column[idx] =
        (simplex_dimensions[idx] == target_dimension && is_positive[idx] != 0) ? 1 : 0;
}

__global__ void __launch_bounds__(256)
    applyClearingKernel(int *__restrict__ boundary_data, int *__restrict__ boundary_indices,
                        const int *__restrict__ clear_column,
                        const int *__restrict__ boundary_starts, int n_simplices)
{
    const int col = blockIdx.x;
    if (col >= n_simplices || clear_column[col] == 0)
    {
        return;
    }

    const int start = boundary_starts[col];
    const int end = boundary_starts[col + 1];
    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        boundary_data[i] = 0;
        boundary_indices[i] = -1;
    }
}

__global__ void __launch_bounds__(256)
    clearingPairingKernel(const int *__restrict__ boundary_indices,
                          const int *__restrict__ boundary_starts,
                          const int *__restrict__ simplex_dimensions,
                          const int *__restrict__ is_negative, int *__restrict__ birth_simplex,
                          int n_simplices)
{
    (void)simplex_dimensions;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simplices)
    {
        return;
    }
    if (is_negative[idx] == 0)
    {
        birth_simplex[idx] = -1;
        return;
    }

    const int start = boundary_starts[idx];
    const int end = boundary_starts[idx + 1];
    int lowest = -1;
    for (int i = start; i < end; ++i)
    {
        const int row = boundary_indices[i];
        if (row >= 0 && row > lowest)
        {
            lowest = row;
        }
    }
    birth_simplex[idx] = lowest;
}

} // namespace

void launchDetectPositiveSimplices(const int *d_boundary_data, const int *d_boundary_indices,
                                   const int *d_boundary_starts, const int *d_dimensions,
                                   int *d_is_positive, int n_simplices, cudaStream_t stream)
{
    if (d_boundary_data == nullptr || d_boundary_indices == nullptr ||
        d_boundary_starts == nullptr || d_dimensions == nullptr || d_is_positive == nullptr)
    {
        return;
    }
    const int grid = gridFor(n_simplices);
    if (grid == 0)
    {
        return;
    }
    detectPositiveSimplicesKernel<<<grid, kClearingBlockSize, 0, stream>>>(
        d_boundary_data, d_boundary_indices, d_boundary_starts, d_dimensions, d_is_positive,
        n_simplices);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchMarkClearingColumns(const int *d_is_positive, const int *d_dimensions,
                               int *d_clear_column, int target_dimension, int n_simplices,
                               cudaStream_t stream)
{
    if (d_is_positive == nullptr || d_dimensions == nullptr || d_clear_column == nullptr)
    {
        return;
    }
    const int grid = gridFor(n_simplices);
    if (grid == 0)
    {
        return;
    }
    markClearingColumnsKernel<<<grid, kClearingBlockSize, 0, stream>>>(
        d_is_positive, d_dimensions, d_clear_column, target_dimension, n_simplices);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchApplyClearing(int *d_boundary_data, int *d_boundary_indices, const int *d_clear_column,
                         const int *d_starts, int n_simplices, cudaStream_t stream)
{
    if (d_boundary_data == nullptr || d_boundary_indices == nullptr || d_clear_column == nullptr ||
        d_starts == nullptr || n_simplices <= 0)
    {
        return;
    }
    applyClearingKernel<<<n_simplices, kClearingBlockSize, 0, stream>>>(
        d_boundary_data, d_boundary_indices, d_clear_column, d_starts, n_simplices);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchClearingPairing(const int *d_boundary_indices, const int *d_starts,
                           const int *d_dimensions, const int *d_is_negative, int *d_birth_simplex,
                           int n_simplices, cudaStream_t stream)
{
    if (d_boundary_indices == nullptr || d_starts == nullptr || d_dimensions == nullptr ||
        d_is_negative == nullptr || d_birth_simplex == nullptr)
    {
        return;
    }
    const int grid = gridFor(n_simplices);
    if (grid == 0)
    {
        return;
    }
    clearingPairingKernel<<<grid, kClearingBlockSize, 0, stream>>>(
        d_boundary_indices, d_starts, d_dimensions, d_is_negative, d_birth_simplex, n_simplices);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace nerve::gpu::persistence::detail
