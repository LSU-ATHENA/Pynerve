
#include "nerve/gpu/gpu_error.hpp"
#include "nerve/persistence/cuda/cuda_tensor_core.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h> // WMMA API for Tensor Cores

#include <cmath>
#include <cstddef>
#include <limits>

namespace nerve
{
namespace gpu
{
namespace tensorcore
{
using namespace nvcuda::wmma;
constexpr int WMMA_M = 16;                  // Tile M dimension
constexpr int WMMA_N = 16;                  // Tile N dimension
constexpr int WMMA_K = 16;                  // Tile K dimension
constexpr int TENSOR_CORE_BLOCK_SIZE = 256; // Optimal for tensor core operations
constexpr int TENSOR_CORE_WARP_SIZE = 32;   // CUDA warp size
constexpr int TENSOR_CORE_WARPS_PER_BLOCK =
    TENSOR_CORE_BLOCK_SIZE / TENSOR_CORE_WARP_SIZE; // 8 warps
constexpr float TENSOR_CORE_NO_EDGE = std::numeric_limits<float>::infinity();

bool checkedSizeProduct(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedByteSize(size_t count, size_t element_size, size_t &bytes)
{
    return checkedSizeProduct(count, element_size, bytes);
}

bool checkedIntSquare(int value, int &out)
{
    if (value <= 0 || value > std::numeric_limits<int>::max() / value)
    {
        return false;
    }
    out = value * value;
    return true;
}

int ceilDivInt(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

bool canUseTensorCorePath(int n_points, int point_dim, const cudaDeviceProp &prop)
{
    return prop.major >= 7 && n_points % WMMA_M == 0 && point_dim >= WMMA_K;
}

__device__ inline bool accumulateFloatProduct(float value, float &sum)
{
    const float contribution = value * value;
    const float next_sum = sum + contribution;
    if (!isfinite(value) || !isfinite(contribution) || !isfinite(next_sum))
    {
        sum = INFINITY;
        return false;
    }
    sum = next_sum;
    return true;
}

__device__ inline bool accumulateDoubleDifference(double diff, double &sum_sq)
{
    const double contribution = diff * diff;
    const double next_sum = sum_sq + contribution;
    if (!isfinite(diff) || !isfinite(contribution) || !isfinite(next_sum))
    {
        sum_sq = INFINITY;
        return false;
    }
    sum_sq = next_sum;
    return true;
}

__global__ void __launch_bounds__(TENSOR_CORE_BLOCK_SIZE) distanceMatrixTensorCoreKernel(
    const half *__restrict__ d_points,     // Points in half precision
    float *__restrict__ d_distance_matrix, // Output distances [n_points,
    float *__restrict__ d_norms,           // Precomputed squared norms [n_points]
    int n_points, int point_dim, float max_radius_sq)
{
    int warp_id = threadIdx.x / TENSOR_CORE_WARP_SIZE;
    int lane_id = threadIdx.x % TENSOR_CORE_WARP_SIZE;
    int tile_i = (blockIdx.x * blockDim.x / TENSOR_CORE_WARP_SIZE + warp_id) * WMMA_M;
    int tile_j = blockIdx.y * WMMA_N;

    if (tile_i >= n_points || tile_j >= n_points)
        return;
    if (tile_i + WMMA_M > n_points || tile_j + WMMA_N > n_points)
        return;
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    __shared__ float tile_products[TENSOR_CORE_WARPS_PER_BLOCK][WMMA_M * WMMA_N];
    fill_fragment(c_frag, 0.0f);

    const int tail_start = (point_dim / WMMA_K) * WMMA_K;
    for (int k = 0; k < tail_start; k += WMMA_K)
    {
        load_matrix_sync(a_frag, d_points + tile_i * point_dim + k, point_dim);
        load_matrix_sync(b_frag, d_points + tile_j * point_dim + k, point_dim);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    store_matrix_sync(tile_products[warp_id], c_frag, WMMA_N, mem_row_major);
    for (int elem = lane_id; elem < WMMA_M * WMMA_N; elem += TENSOR_CORE_WARP_SIZE)
    {
        int i = elem / WMMA_N;
        int j = elem % WMMA_N;
        int global_i = tile_i + i;
        int global_j = tile_j + j;
        if (global_i < n_points && global_j < n_points && global_i < global_j)
        {
            float dot_product = tile_products[warp_id][elem];
            for (int d = tail_start; d < point_dim; ++d)
            {
                const float ai = __half2float(d_points[global_i * point_dim + d]);
                const float bj = __half2float(d_points[global_j * point_dim + d]);
                const float contribution = ai * bj;
                const float next_dot = dot_product + contribution;
                if (!isfinite(ai) || !isfinite(bj) || !isfinite(contribution) ||
                    !isfinite(next_dot))
                {
                    dot_product = INFINITY;
                    break;
                }
                dot_product = next_dot;
            }
            float norm_i = d_norms[global_i];
            float norm_j = d_norms[global_j];
            float dist_sq = norm_i + norm_j - 2.0f * dot_product;
            if (isfinite(dist_sq) && dist_sq <= max_radius_sq && dist_sq >= 0.0f)
            {
                float dist = sqrtf(dist_sq);
                d_distance_matrix[global_i * n_points + global_j] = dist;
                d_distance_matrix[global_j * n_points + global_i] = dist;
            }
            else
            {
                d_distance_matrix[global_i * n_points + global_j] = TENSOR_CORE_NO_EDGE;
                d_distance_matrix[global_j * n_points + global_i] = TENSOR_CORE_NO_EDGE;
            }
        }
    }
}
__global__ void __launch_bounds__(256)
    computePointNormsKernel(const half *__restrict__ d_points, float *__restrict__ d_norms,
                            int n_points, int point_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points)
        return;

    float norm_sq = 0.0f;
    for (int d = 0; d < point_dim; ++d)
    {
        float val = __half2float(d_points[idx * point_dim + d]);
        if (!accumulateFloatProduct(val, norm_sq))
        {
            break;
        }
    }
    d_norms[idx] = norm_sq;
}

__global__ void __launch_bounds__(256)
    convertDoubleToHalfKernel(const double *__restrict__ input, half *__restrict__ output,
                              size_t total)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        output[idx] = __double2half(input[idx]);
    }
}

__global__ void __launch_bounds__(256)
    standardDistanceMatrixKernel(const double *__restrict__ points, float *__restrict__ distances,
                                 int n_points, int point_dim, double max_radius_sq)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_points * n_points;
    if (idx >= total)
    {
        return;
    }
    const int i = idx / n_points;
    const int j = idx % n_points;
    if (i == j)
    {
        distances[idx] = 0.0f;
        return;
    }
    double sum_sq = 0.0;
    for (int d = 0; d < point_dim; ++d)
    {
        const double diff = points[i * point_dim + d] - points[j * point_dim + d];
        if (!accumulateDoubleDifference(diff, sum_sq))
        {
            break;
        }
    }
    if (isfinite(sum_sq) && sum_sq <= max_radius_sq)
    {
        distances[idx] = static_cast<float>(sqrt(sum_sq));
    }
    else
    {
        distances[idx] = TENSOR_CORE_NO_EDGE;
    }
}

void launchStandardDistanceMatrix(const double *d_points_double, float *d_distance_matrix,
                                  int n_points, int point_dim, double max_radius_sq,
                                  cudaStream_t stream)
{
    int total = 0;
    if (!checkedIntSquare(n_points, total))
    {
        return;
    }
    const int threads = TENSOR_CORE_BLOCK_SIZE;
    const int blocks = ceilDivInt(total, threads);
    standardDistanceMatrixKernel<<<blocks, threads, 0, stream>>>(
        d_points_double, d_distance_matrix, n_points, point_dim, max_radius_sq);
    GPU_CHECK(cudaPeekAtLastError());
}
cudaError_t convertToHalfPrecision(const double *d_points_double, half *d_points_half,
                                   size_t n_points, size_t point_dim, cudaStream_t stream)
{
    size_t total = 0;
    if (!checkedSizeProduct(n_points, point_dim, total) || total == 0)
    {
        return cudaErrorInvalidValue;
    }
    const int threads = TENSOR_CORE_BLOCK_SIZE;
    const size_t block_count = (total + threads - 1) / threads;
    if (block_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return cudaErrorInvalidConfiguration;
    }
    const int blocks = static_cast<int>(block_count);
    convertDoubleToHalfKernel<<<blocks, threads, 0, stream>>>(d_points_double, d_points_half,
                                                              total);
    return cudaPeekAtLastError();
}
void launchTensorCoreDistanceMatrix(const double *d_points_double, // Input in double precision
                                    float *d_distance_matrix,      // Output in float precision
                                    int n_points, int point_dim, double max_radius,
                                    cudaStream_t stream)
{
    if (d_points_double == nullptr || d_distance_matrix == nullptr || n_points <= 0 ||
        point_dim <= 0 || !std::isfinite(max_radius) || !(max_radius > 0.0))
    {
        return;
    }
    int total_distance_entries = 0;
    if (!checkedIntSquare(n_points, total_distance_entries))
    {
        return;
    }
    const double max_radius_sq = max_radius * max_radius;
    if (!std::isfinite(max_radius_sq))
    {
        return;
    }
    cudaDeviceProp prop;
    int device_id = 0;
    const cudaError_t device_status = cudaGetDevice(&device_id);
    const cudaError_t prop_status =
        (device_status == cudaSuccess) ? cudaGetDeviceProperties(&prop, device_id) : device_status;

    if (prop_status != cudaSuccess || !canUseTensorCorePath(n_points, point_dim, prop))
    {
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }
    half *d_points_half = nullptr;
    float *d_norms = nullptr;

    size_t point_elements = 0;
    size_t points_size = 0;
    size_t norms_size = 0;
    size_t distance_size = 0;
    const size_t distance_elements = static_cast<size_t>(total_distance_entries);
    if (!checkedSizeProduct(static_cast<size_t>(n_points), static_cast<size_t>(point_dim),
                            point_elements) ||
        !checkedByteSize(point_elements, sizeof(half), points_size) ||
        !checkedByteSize(static_cast<size_t>(n_points), sizeof(float), norms_size) ||
        !checkedByteSize(distance_elements, sizeof(float), distance_size))
    {
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }

    if (cudaMalloc(&d_points_half, points_size) != cudaSuccess ||
        cudaMalloc(&d_norms, norms_size) != cudaSuccess)
    {
        if (d_points_half != nullptr)
        {
            cudaFree(d_points_half);
        }
        if (d_norms != nullptr)
        {
            cudaFree(d_norms);
        }
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }
    if (convertToHalfPrecision(d_points_double, d_points_half, n_points, point_dim, stream) !=
        cudaSuccess)
    {
        cudaFree(d_points_half);
        cudaFree(d_norms);
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }
    int norm_threads = TENSOR_CORE_BLOCK_SIZE;
    int norm_blocks = (n_points + norm_threads - 1) / norm_threads;
    computePointNormsKernel<<<norm_blocks, norm_threads, 0, stream>>>(d_points_half, d_norms,
                                                                      n_points, point_dim);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaFree(d_points_half);
        cudaFree(d_norms);
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }
    int tiles_m = (n_points + WMMA_M - 1) / WMMA_M;
    int tiles_n = (n_points + WMMA_N - 1) / WMMA_N;
    const int tile_row_blocks = ceilDivInt(tiles_m, TENSOR_CORE_WARPS_PER_BLOCK);
    if (tile_row_blocks > prop.maxGridSize[0] || tiles_n > prop.maxGridSize[1])
    {
        cudaFree(d_points_half);
        cudaFree(d_norms);
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }

    dim3 grid(tile_row_blocks, tiles_n);
    dim3 block(TENSOR_CORE_WARP_SIZE * TENSOR_CORE_WARPS_PER_BLOCK); // 256 threads

    if (cudaMemsetAsync(d_distance_matrix, 0, distance_size, stream) != cudaSuccess)
    {
        cudaFree(d_points_half);
        cudaFree(d_norms);
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }
    if (max_radius_sq > static_cast<double>(std::numeric_limits<float>::max()))
    {
        cudaFree(d_points_half);
        cudaFree(d_norms);
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }
    const float max_radius_sq_f = static_cast<float>(max_radius_sq);

    distanceMatrixTensorCoreKernel<<<grid, block, 0, stream>>>(
        d_points_half, d_distance_matrix, d_norms, n_points, point_dim, max_radius_sq_f);
    if (cudaGetLastError() != cudaSuccess)
    {
        cudaFree(d_points_half);
        cudaFree(d_norms);
        launchStandardDistanceMatrix(d_points_double, d_distance_matrix, n_points, point_dim,
                                     max_radius_sq, stream);
        return;
    }
    cudaFree(d_points_half);
    cudaFree(d_norms);
}
bool areTensorCoresAvailable()
{
    cudaDeviceProp prop;
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
    {
        return false;
    }
    return prop.major >= 7;
}
TensorCoreInfo getTensorCoreInfo()
{
    TensorCoreInfo info;
    cudaDeviceProp prop;
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
    {
        info.available = false;
        info.generation = "Missing";
        info.supports_fp16 = false;
        info.supports_bf16 = false;
        info.supports_tf32 = false;
        info.supports_fp8 = false;
        return info;
    }

    info.available = prop.major >= 7;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    if (prop.major == 7)
    {
        info.generation = "Volta/Turing";
    }
    else if (prop.major == 8)
    {
        info.generation = "Ampere/Ada";
    }
    else if (prop.major == 9)
    {
        info.generation = "Hopper";
    }
    else if (prop.major >= 10)
    {
        info.generation = "Blackwell+";
    }

    info.supports_fp16 = true;            // All Tensor Core gens support FP16
    info.supports_bf16 = prop.major >= 8; // Ampere+
    info.supports_tf32 = prop.major >= 8; // Ampere+
    info.supports_fp8 = prop.major >= 9;  // Hopper+

    return info;
}

} // namespace tensorcore
} // namespace gpu
} // namespace nerve
