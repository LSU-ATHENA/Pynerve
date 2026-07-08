#include "nerve/core_types.hpp"
#include "nerve/gpu/gpu_error.hpp"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
#include <cuda/pipeline>
#endif

namespace cg = cooperative_groups;

#ifdef NDEBUG
#define KERNEL_ASSERT(cond) ((void)0)
#else
#define KERNEL_ASSERT(cond)                                                                        \
    do                                                                                             \
    {                                                                                              \
        if (!(cond))                                                                               \
        {                                                                                          \
            printf("[KERNEL] ASSERT FAIL: %s  thread(%d,%d) block(%d,%d)\n", #cond, threadIdx.x,   \
                   threadIdx.y, blockIdx.x, blockIdx.y);                                           \
            __trap();                                                                              \
        }                                                                                          \
    } while (0)
#endif

constexpr int DIST_TILE = 16;
constexpr int DIST_TILE_THREADS = 16; // Threads per dimension for FP16 kernel

__device__ inline double deviceInfinityDouble()
{
    return __longlong_as_double(0x7ff0000000000000ULL);
}

__device__ inline float deviceInfinityFloat()
{
    return __int_as_float(0x7f800000);
}

__device__ inline bool accumulateDoubleDifference(double diff, double &dist_sq)
{
    const double next = fma(diff, diff, dist_sq);
    if (!isfinite(diff) || !isfinite(next))
    {
        return false;
    }
    dist_sq = next;
    return true;
}

__device__ inline bool accumulateFloatDifference(float diff, float &dist_sq)
{
    const float next = fmaf(diff, diff, dist_sq);
    if (!isfinite(diff) || !isfinite(next))
    {
        return false;
    }
    dist_sq = next;
    return true;
}

__host__ __device__ inline std::size_t matrixIndex(nerve::Size row, nerve::Size col,
                                                   nerve::Size leading_dim)
{
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(leading_dim) +
           static_cast<std::size_t>(col);
}

__global__ void __launch_bounds__(256)
    computeDistanceKernel(const double *__restrict__ points, // [n x dim] row-major
                          double *__restrict__ out_dist,     // [n x n]   row-major
                          nerve::Size n, nerve::Size dim)
{
    __shared__ double tile_a[DIST_TILE][DIST_TILE];
    __shared__ double tile_b[DIST_TILE][DIST_TILE];

    const nerve::Size row = blockIdx.y * DIST_TILE + threadIdx.y;
    const nerve::Size col = blockIdx.x * DIST_TILE + threadIdx.x;

    double dist_sq = 0.0;
    bool valid_distance = true;
    const nerve::Size n_tiles = (dim + DIST_TILE - 1) / DIST_TILE;

    for (nerve::Size t = 0; t < n_tiles; ++t)
    {
        // Load tile A
        nerve::Size dim_a = t * DIST_TILE + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] =
            (row < n && dim_a < dim) ? points[matrixIndex(row, dim_a, dim)] : 0.0;

        // Load tile B
        nerve::Size dim_b = t * DIST_TILE + threadIdx.y;
        tile_b[threadIdx.x][threadIdx.y] =
            (col < n && dim_b < dim) ? points[matrixIndex(col, dim_b, dim)] : 0.0;

        __syncthreads();

// Compute partial squared distance for this tile
#pragma unroll
        for (int k = 0; k < DIST_TILE; ++k)
        {
            const double diff = tile_a[threadIdx.y][k] - tile_b[threadIdx.x][k];
            if (!accumulateDoubleDifference(diff, dist_sq))
            {
                valid_distance = false;
                break;
            }
        }

        __syncthreads();
    }

    // Write output with bounds check
    if (row < n && col < n)
    {
        out_dist[matrixIndex(row, col, n)] =
            valid_distance ? sqrt(dist_sq) : deviceInfinityDouble();
    }
}

// -- FP16 kernel: FP32 accumulation prevents catastrophic cancellation --
__global__ void __launch_bounds__(256)
    computeDistanceFp16Kernel(const __half *__restrict__ points, __half *__restrict__ out_dist,
                              nerve::Size n, nerve::Size dim)
{
    const nerve::Size row = blockIdx.y * blockDim.y + threadIdx.y;
    const nerve::Size col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n)
        return;

    // CRITICAL: accumulate in float32, not float16
    float dist_sq = 0.0f;
    bool valid_distance = true;
    for (nerve::Size k = 0; k < dim; ++k)
    {
        float a = __half2float(points[matrixIndex(row, k, dim)]);
        float b = __half2float(points[matrixIndex(col, k, dim)]);
        float d = a - b;
        if (!accumulateFloatDifference(d, dist_sq))
        {
            valid_distance = false;
            break;
        }
    }

    out_dist[matrixIndex(row, col, n)] =
        __float2half(valid_distance ? sqrtf(dist_sq) : deviceInfinityFloat());
}

__global__ void __launch_bounds__(256)
    edgeFiltrationKernel(const double *__restrict__ dist_matrix, // [n x n]
                         const nerve::Index *__restrict__ edges, // [n_edges x 2]
                         double *__restrict__ filtration,        // [n_edges]
                         nerve::Size n_edges, nerve::Size n)
{
    const nerve::Size e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges)
        return;

    const nerve::Index i = edges[2 * e];
    const nerve::Index j = edges[2 * e + 1];

    KERNEL_ASSERT(i >= 0 && i < n && j >= 0 && j < n);
    KERNEL_ASSERT(e < n_edges);

    const double value = dist_matrix[matrixIndex(i, j, n)];
    filtration[e] = isfinite(value) ? value : deviceInfinityDouble();
}

__global__ void __launch_bounds__(256)
    batchedDistanceKernel(const double *__restrict__ points_a, // [batch_a x dim]
                          const double *__restrict__ points_b, // [batch_b x dim]
                          double *__restrict__ distances,      // [batch_a x batch_b]
                          nerve::Size batch_a, nerve::Size batch_b, nerve::Size dim)
{
    constexpr int TILE = 16;
    __shared__ double tile_a[TILE][TILE];
    __shared__ double tile_b[TILE][TILE];

    const nerve::Size row = blockIdx.y * TILE + threadIdx.y;
    const nerve::Size col = blockIdx.x * TILE + threadIdx.x;

    if (row >= batch_a || col >= batch_b)
        return;

    double dist_sq = 0.0;
    bool valid_distance = true;
    const nerve::Size n_tiles = (dim + TILE - 1) / TILE;

    for (nerve::Size t = 0; t < n_tiles; ++t)
    {
        const nerve::Size dim_idx = t * TILE + threadIdx.x;

        tile_a[threadIdx.y][threadIdx.x] =
            (dim_idx < dim) ? points_a[matrixIndex(row, dim_idx, dim)] : 0.0;
        tile_b[threadIdx.x][threadIdx.y] =
            (dim_idx < dim) ? points_b[matrixIndex(col, dim_idx, dim)] : 0.0;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            const double diff = tile_a[threadIdx.y][k] - tile_b[threadIdx.x][k];
            if (!accumulateDoubleDifference(diff, dist_sq))
            {
                valid_distance = false;
                break;
            }
        }

        __syncthreads();
    }

    distances[matrixIndex(row, col, batch_b)] =
        valid_distance ? sqrt(dist_sq) : deviceInfinityDouble();
}

// Async-copy pipeline kernel (needs CUDA >= 11.0)
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000

template <typename T>
__global__ void __launch_bounds__(256)
    tiledDistanceKernelAsync(const T *__restrict__ points, T *__restrict__ out_dist, nerve::Size n,
                             nerve::Size dim)
{
    constexpr int TILE = DIST_TILE;
    __shared__ T tile_a[TILE][TILE];
    __shared__ T tile_b[TILE][TILE];

    const nerve::Size row = blockIdx.y * TILE + threadIdx.y;
    const nerve::Size col = blockIdx.x * TILE + threadIdx.x;

    T dist_sq = T(0);
    bool valid_distance = true;
    const nerve::Size n_tiles = (dim + TILE - 1) / TILE;

    for (nerve::Size t = 0; t < n_tiles; ++t)
    {
        auto pipeline = cuda::make_pipeline();
        pipeline.producer_acquire();

        const nerve::Size dim_base = t * TILE;
        const size_t tile_elems =
            (dim_base + TILE <= dim)
                ? static_cast<size_t>(TILE)
                : (dim > dim_base ? static_cast<size_t>(dim - dim_base) : static_cast<size_t>(0));

        // copy each row of tile_a as a contiguous block
        for (int r = 0; r < TILE; ++r)
        {
            const T *src =
                (row + r < n) ? &points[static_cast<size_t>(row + r) * dim + dim_base] : &points[0];
            cuda::memcpy_async(tile_a[r], src, tile_elems, pipeline);
        }

        // copy each row of tile_b as a contiguous block
        for (int r = 0; r < TILE; ++r)
        {
            const T *src =
                (col + r < n) ? &points[static_cast<size_t>(col + r) * dim + dim_base] : &points[0];
            cuda::memcpy_async(tile_b[r], src, tile_elems, pipeline);
        }

        pipeline.producer_commit();
        pipeline.consumer_wait();
        pipeline.consumer_release();

        // Zero elements copied from out-of-bounds source rows
        if (row + threadIdx.y >= n)
        {
            tile_a[threadIdx.y][threadIdx.x] = T(0);
        }
        if (col + threadIdx.x >= n)
        {
            tile_b[threadIdx.x][threadIdx.y] = T(0);
        }

        // Zero elements in out-of-bounds dimension columns
        if (threadIdx.x >= tile_elems && row + threadIdx.y < n)
        {
            tile_a[threadIdx.y][threadIdx.x] = T(0);
        }
        if (threadIdx.y >= tile_elems && col + threadIdx.x < n)
        {
            tile_b[threadIdx.x][threadIdx.y] = T(0);
        }
        __syncthreads();

        // Compute partial squared distance for this tile
#pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            const T diff = tile_a[threadIdx.y][k] - tile_b[threadIdx.x][k];
            if constexpr (sizeof(T) == sizeof(double))
            {
                if (!accumulateDoubleDifference(static_cast<double>(diff), dist_sq))
                {
                    valid_distance = false;
                }
            }
            else
            {
                if (!accumulateFloatDifference(static_cast<float>(diff), dist_sq))
                {
                    valid_distance = false;
                }
            }
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        if constexpr (sizeof(T) == sizeof(double))
        {
            out_dist[matrixIndex(row, col, n)] =
                valid_distance ? static_cast<T>(sqrt(static_cast<double>(dist_sq)))
                               : deviceInfinityDouble();
        }
        else
        {
            out_dist[matrixIndex(row, col, n)] =
                valid_distance ? static_cast<T>(sqrtf(static_cast<float>(dist_sq)))
                               : deviceInfinityFloat();
        }
    }
}

#endif // CUDART_VERSION >= 11000

namespace nerve::gpu
{

CudaResult<void> launchDistanceMatrix(const double *d_points, // device pointer
                                      double *d_out,          // device pointer
                                      nerve::Size n, nerve::Size dim, cudaStream_t stream = nullptr)
{
    if (n <= 0 || dim <= 0)
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "n and dim must be positive");
    if (d_points == nullptr || d_out == nullptr)
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "distance matrix buffers must be non-null");
    }

    // Validate grid dimensions (multi-GPU: query current device, not hardcoded 0)
    cudaDeviceProp prop;
    int current_device = 0;
    CUDA_CALL(cudaGetDevice(&current_device));
    CUDA_CALL(cudaGetDeviceProperties(&prop, current_device));

    dim3 block(DIST_TILE, DIST_TILE);
    const nerve::Size grid_extent = ((n - 1) / DIST_TILE) + 1;
    if (grid_extent > std::numeric_limits<unsigned>::max())
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "Input too large for single kernel launch. Use batched mode.");
    }
    dim3 grid(static_cast<unsigned>(grid_extent), static_cast<unsigned>(grid_extent));

    if (grid.x > (unsigned)prop.maxGridSize[0] || grid.y > (unsigned)prop.maxGridSize[1])
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "Input too large for single kernel launch. Use batched mode.");
    }

    // Clear any pre-existing async errors
    cudaGetLastError();

    // Launch
    computeDistanceKernel<<<grid, block, 0, stream>>>(d_points, d_out, n, dim);

    // Check BOTH launch errors and execution errors
    CUDA_KERNEL_CHECK("computeDistanceKernel");

    return CudaResult<void>::ok();
}

// FP16 version with error bounds documentation
CudaResult<void> launchDistanceMatrixFp16(const __half *d_points, // device pointer
                                          __half *d_out,          // device pointer
                                          nerve::Size n, nerve::Size dim,
                                          cudaStream_t stream = nullptr)
{
    if (n <= 0 || dim <= 0)
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "n and dim must be positive");
    if (d_points == nullptr || d_out == nullptr)
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "distance matrix buffers must be non-null");
    }

    dim3 block(DIST_TILE_THREADS, DIST_TILE_THREADS);
    const nerve::Size grid_extent = ((n - 1) / DIST_TILE_THREADS) + 1;
    if (grid_extent > std::numeric_limits<unsigned>::max())
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "Input too large for single kernel launch. Use batched mode.");
    }
    dim3 grid(static_cast<unsigned>(grid_extent), static_cast<unsigned>(grid_extent));

    computeDistanceFp16Kernel<<<grid, block, 0, stream>>>(d_points, d_out, n, dim);

    CUDA_KERNEL_CHECK("computeDistanceFp16Kernel");
    return CudaResult<void>::ok();
}

// Edge filtration for VR complexes
CudaResult<void> launchEdgeFiltration(const double *d_dist_matrix, const nerve::Index *d_edges,
                                      double *d_filtration, nerve::Size n_edges,
                                      nerve::Size n_points, cudaStream_t stream = nullptr)
{
    if (n_edges == 0)
    {
        return CudaResult<void>::ok();
    }
    if (n_points == 0 || d_dist_matrix == nullptr || d_edges == nullptr || d_filtration == nullptr)
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "edge filtration buffers and dimensions must be valid");
    }
    const nerve::Size block_size = 256;
    const nerve::Size grid_size = ((n_edges - 1) / block_size) + 1;
    if (grid_size > std::numeric_limits<unsigned>::max())
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "edge filtration input too large for single kernel launch");
    }

    edgeFiltrationKernel<<<static_cast<unsigned>(grid_size), static_cast<unsigned>(block_size), 0,
                           stream>>>(d_dist_matrix, d_edges, d_filtration, n_edges, n_points);

    CUDA_KERNEL_CHECK("edgeFiltrationKernel");
    return CudaResult<void>::ok();
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000

template <typename T>
CudaResult<void> launchTiledDistanceAsync(const T *d_points, T *d_out, nerve::Size n,
                                          nerve::Size dim, cudaStream_t stream = nullptr)
{
    if (n <= 0 || dim <= 0)
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "n and dim must be positive");
    }
    if (d_points == nullptr || d_out == nullptr)
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "distance matrix buffers must be non-null");
    }

    cudaDeviceProp prop;
    int current_device = 0;
    CUDA_CALL(cudaGetDevice(&current_device));
    CUDA_CALL(cudaGetDeviceProperties(&prop, current_device));

    dim3 block(DIST_TILE, DIST_TILE);
    const nerve::Size grid_extent = ((n - 1) / DIST_TILE) + 1;
    if (grid_extent > std::numeric_limits<unsigned>::max())
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "Input too large for single kernel launch. Use batched mode.");
    }
    dim3 grid(static_cast<unsigned>(grid_extent), static_cast<unsigned>(grid_extent));

    if (grid.x > static_cast<unsigned>(prop.maxGridSize[0]) ||
        grid.y > static_cast<unsigned>(prop.maxGridSize[1]))
    {
        return CudaResult<void>::err(nerve::error::TDAErrorCode::InvalidInput,
                                     "Input too large for single kernel launch. Use batched mode.");
    }

    cudaGetLastError();

    tiledDistanceKernelAsync<T><<<grid, block, 0, stream>>>(d_points, d_out, n, dim);
    CUDA_KERNEL_CHECK("tiledDistanceKernelAsync");

    return CudaResult<void>::ok();
}

template CudaResult<void> launchTiledDistanceAsync<double>(const double *, double *, nerve::Size,
                                                           nerve::Size, cudaStream_t);
template CudaResult<void> launchTiledDistanceAsync<float>(const float *, float *, nerve::Size,
                                                          nerve::Size, cudaStream_t);

#endif // CUDART_VERSION >= 11000

} // namespace nerve::gpu
