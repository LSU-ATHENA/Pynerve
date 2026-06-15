
#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace nerve
{
namespace gpu
{
namespace kernels
{

// GPU Kernel Launch Constants
constexpr int LAPLACIAN_BLOCK_SIZE = 256;
constexpr int LAPLACIAN_TILE_SIZE = 16;

namespace
{

size_t checkedProduct(size_t lhs, size_t rhs, const char *context)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        throw std::length_error(context);
    }
    return lhs * rhs;
}

size_t checkedGridSize(size_t total_elements, const char *context)
{
    const size_t block_size = static_cast<size_t>(LAPLACIAN_BLOCK_SIZE);
    const size_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::length_error(context);
    }
    return grid_size;
}

unsigned int checkedTiledGridAxis(size_t n, const char *context)
{
    const size_t tile_size = static_cast<size_t>(LAPLACIAN_TILE_SIZE);
    const size_t axis = (n + tile_size - 1) / tile_size;
    if (axis > static_cast<size_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::length_error(context);
    }
    return static_cast<unsigned int>(axis);
}

} // namespace

__device__ inline double deviceInfinityDouble()
{
    return __longlong_as_double(0x7ff0000000000000ULL);
}

__device__ inline bool accumulateMatrixProduct(double lhs, double rhs, double &sum)
{
    const double product = lhs * rhs;
    const double next = sum + product;
    if (!isfinite(lhs) || !isfinite(rhs) || !isfinite(product) || !isfinite(next))
    {
        return false;
    }
    sum = next;
    return true;
}

// Kernel to compute up-Laplacian: L_up = B_{d+1} * B_{d+1}^T
// where B is the boundary matrix
// Each thread computes one element of the resulting Laplacian matrix
__global__ __launch_bounds__(LAPLACIAN_BLOCK_SIZE) void computeUpLaplacianKernel(
    const double *boundary_matrix, // [size x size] - full boundary matrix
    const int *simplex_dimensions, // [size] - dimension of each simplex
    double *up_laplacian,          // [n_d x n_d] output
    size_t size,                   // total number of simplices
    int d,                         // target dimension
    size_t n_d,                    // number of d-simplices
    const size_t *d_indices        // indices of d-simplices in boundary_matrix
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_d != 0 && n_d > static_cast<size_t>(-1) / n_d)
    {
        return;
    }
    size_t total_elements = n_d * n_d;

    if (idx >= total_elements)
    {
        return;
    }

    size_t i = idx / n_d; // row in output
    size_t j = idx % n_d; // col in output

    // Map to boundary matrix indices
    size_t row_idx = d_indices[i];
    size_t col_idx = d_indices[j];

    double sum = 0.0;
    bool valid_sum = true;

    // Compute (B_{d+1} * B_{d+1}^T)[i,j]
    // Sum over (d+1)-simplices k: B[i,k] * B[j,k]
    for (size_t k = 0; k < size; ++k)
    {
        if (simplex_dimensions[k] == d + 1)
        {
            double b_ik = boundary_matrix[row_idx * size + k];
            double b_jk = boundary_matrix[col_idx * size + k];
            if (!accumulateMatrixProduct(b_ik, b_jk, sum))
            {
                valid_sum = false;
                break;
            }
        }
    }

    up_laplacian[i * n_d + j] = valid_sum ? sum : deviceInfinityDouble();
}

// Kernel to compute down-Laplacian: L_down = B_d^T * B_d
// Each thread computes one element
__global__ __launch_bounds__(LAPLACIAN_BLOCK_SIZE) void computeDownLaplacianKernel(
    const double *boundary_matrix, const int *simplex_dimensions, double *down_laplacian,
    size_t size, int d, size_t n_d, const size_t *d_indices)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_d != 0 && n_d > static_cast<size_t>(-1) / n_d)
    {
        return;
    }
    size_t total_elements = n_d * n_d;

    if (idx >= total_elements)
    {
        return;
    }

    size_t i = idx / n_d;
    size_t j = idx % n_d;

    size_t row_idx = d_indices[i];
    size_t col_idx = d_indices[j];

    double sum = 0.0;
    bool valid_sum = true;

    // Compute (B_d^T * B_d)[i,j]
    // Sum over (d-1)-simplices k: B[k,i] * B[k,j]
    for (size_t k = 0; k < size; ++k)
    {
        if (simplex_dimensions[k] == d - 1)
        {
            double b_ki = boundary_matrix[k * size + row_idx];
            double b_kj = boundary_matrix[k * size + col_idx];
            if (!accumulateMatrixProduct(b_ki, b_kj, sum))
            {
                valid_sum = false;
                break;
            }
        }
    }

    down_laplacian[i * n_d + j] = valid_sum ? sum : deviceInfinityDouble();
}

// Kernel to compute Hodge Laplacian: L = L_up + L_down
__global__ __launch_bounds__(LAPLACIAN_BLOCK_SIZE) void addMatricesKernel(const double *matrix_a,
                                                                          const double *matrix_b,
                                                                          double *result, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n != 0 && n > static_cast<size_t>(-1) / n)
    {
        return;
    }
    size_t total_elements = n * n;

    if (idx >= total_elements)
    {
        return;
    }

    const double sum = matrix_a[idx] + matrix_b[idx];
    result[idx] = isfinite(matrix_a[idx]) && isfinite(matrix_b[idx]) && isfinite(sum)
                      ? sum
                      : deviceInfinityDouble();
}

// Kernel to extract submatrix based on dimension indices
__global__ __launch_bounds__(LAPLACIAN_BLOCK_SIZE) void extractDimensionIndicesKernel(
    const int *simplex_dimensions, size_t *d_indices, unsigned long long *d_count, size_t size,
    int target_dimension)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
    {
        return;
    }

    if (simplex_dimensions[idx] == target_dimension)
    {
        size_t pos = static_cast<size_t>(atomicAdd(d_count, 1ULL));
        d_indices[pos] = idx;
    }
}

// Optimized kernel using shared memory for boundary matrix tiles
// This is much faster for large complexes
__global__ __launch_bounds__(256) void computeUpLaplacianTiledKernel(
    const double *boundary_matrix, const size_t *d_indices,
    const size_t *d1_indices, // (d+1)-simplex indices
    size_t n_d, size_t n_d1, double *up_laplacian, size_t size)
{
    constexpr int TILE_SIZE = LAPLACIAN_TILE_SIZE;

    __shared__ double tile_i[TILE_SIZE][TILE_SIZE]; // B[row, d1_cols]
    __shared__ double tile_j[TILE_SIZE][TILE_SIZE]; // B[col, d1_cols]

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    bool valid_sum = true;

    // Loop over tiles of (d+1)-simplices
    for (size_t tile = 0; tile < (n_d1 + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        size_t tile_offset = tile * TILE_SIZE;

        // Load tile of boundary values
        // tile_i[ty][tx] = B[i, tile_offset + tx]
        if (row < n_d && tile_offset + threadIdx.x < n_d1)
        {
            size_t global_row = d_indices[row];
            size_t global_col = d1_indices[tile_offset + threadIdx.x];
            tile_i[threadIdx.y][threadIdx.x] = boundary_matrix[global_row * size + global_col];
        }
        else
        {
            tile_i[threadIdx.y][threadIdx.x] = 0.0;
        }

        // tile_j[ty][tx] = B[j, tile_offset + tx]
        if (col < n_d && tile_offset + threadIdx.x < n_d1)
        {
            size_t global_row = d_indices[col];
            size_t global_col = d1_indices[tile_offset + threadIdx.x];
            tile_j[threadIdx.y][threadIdx.x] = boundary_matrix[global_row * size + global_col];
        }
        else
        {
            tile_j[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

// Accumulate dot product for this tile
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            if (!accumulateMatrixProduct(tile_i[threadIdx.y][k], tile_j[threadIdx.x][k], sum))
            {
                valid_sum = false;
                break;
            }
        }

        __syncthreads();
    }

    // Write result
    if (row < n_d && col < n_d)
    {
        up_laplacian[row * n_d + col] = valid_sum ? sum : deviceInfinityDouble();
    }
}

// Host wrapper functions
void launchUpLaplacianKernel(const double *d_boundary_matrix, const int *d_simplex_dimensions,
                             double *d_up_laplacian, size_t size, int d, size_t n_d,
                             const size_t *d_indices, cudaStream_t stream)
{
    size_t total_elements = checkedProduct(n_d, n_d, "up-Laplacian grid size overflows");
    size_t blockSize = LAPLACIAN_BLOCK_SIZE;
    size_t gridSize = checkedGridSize(total_elements, "up-Laplacian grid exceeds CUDA limits");

    computeUpLaplacianKernel<<<gridSize, blockSize, 0, stream>>>(
        d_boundary_matrix, d_simplex_dimensions, d_up_laplacian, size, d, n_d, d_indices);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchDownLaplacianKernel(const double *d_boundary_matrix, const int *d_simplex_dimensions,
                               double *d_down_laplacian, size_t size, int d, size_t n_d,
                               const size_t *d_indices, cudaStream_t stream)
{
    size_t total_elements = checkedProduct(n_d, n_d, "down-Laplacian grid size overflows");
    size_t blockSize = LAPLACIAN_BLOCK_SIZE;
    size_t gridSize = checkedGridSize(total_elements, "down-Laplacian grid exceeds CUDA limits");

    computeDownLaplacianKernel<<<gridSize, blockSize, 0, stream>>>(
        d_boundary_matrix, d_simplex_dimensions, d_down_laplacian, size, d, n_d, d_indices);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchAddMatricesKernel(const double *d_matrix_a, const double *d_matrix_b, double *d_result,
                             size_t n, cudaStream_t stream)
{
    size_t total_elements = checkedProduct(n, n, "Laplacian add grid size overflows");
    size_t blockSize = LAPLACIAN_BLOCK_SIZE;
    size_t gridSize = checkedGridSize(total_elements, "Laplacian add grid exceeds CUDA limits");

    addMatricesKernel<<<gridSize, blockSize, 0, stream>>>(d_matrix_a, d_matrix_b, d_result, n);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchUpLaplacianTiledKernel(const double *d_boundary_matrix, const size_t *d_indices,
                                  const size_t *d1_indices, size_t n_d, size_t n_d1,
                                  double *d_up_laplacian, size_t size, cudaStream_t stream)
{
    constexpr int TILE_SIZE = LAPLACIAN_TILE_SIZE;

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    const unsigned int grid_axis =
        checkedTiledGridAxis(n_d, "tiled up-Laplacian grid exceeds CUDA limits");
    dim3 gridSize(grid_axis, grid_axis);

    computeUpLaplacianTiledKernel<<<gridSize, blockSize, 0, stream>>>(
        d_boundary_matrix, d_indices, d1_indices, n_d, n_d1, d_up_laplacian, size);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace kernels
} // namespace gpu
} // namespace nerve
