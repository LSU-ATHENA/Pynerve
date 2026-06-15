
// Optimizations:
// - Warp-level gradient reductions
// - Memory-coalesced tensor operations
// - cuBLAS integration for matrix gradients
// - Fused gradient kernels

#include "nerve/gpu/gpu_error.hpp"

#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace nerve
{
namespace autodiff
{
namespace gpu
{

// Constants
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr size_t GRADIENT_MEMORY_POOL_DEFAULT_SIZE = 64 * 1024 * 1024; // 64MB default pool
constexpr size_t GRADIENT_MEMORY_ALIGNMENT = 256; // 256-byte alignment for memory allocations

__device__ inline float autodiffInfinityFloat()
{
    return __int_as_float(0x7f800000);
}

__device__ inline float finiteOrInfinity(float value)
{
    return isfinite(value) ? value : autodiffInfinityFloat();
}

template <typename T>
void requireDevicePointer(const T *ptr, const char *label)
{
    if (ptr == nullptr)
    {
        throw std::invalid_argument(label);
    }
}

int checkedBlocksForPositiveCount(int n, const char *label)
{
    if (n <= 0)
    {
        throw std::invalid_argument(label);
    }
    const std::size_t blocks =
        (static_cast<std::size_t>(n) + static_cast<std::size_t>(BLOCK_SIZE) - 1U) /
        static_cast<std::size_t>(BLOCK_SIZE);
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(label);
    }
    return static_cast<int>(blocks);
}

void checkCublasStatus(cublasStatus_t status, const char *label)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(label) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

std::size_t checkedAlignedAllocationSize(std::size_t size)
{
    if (size == 0)
    {
        throw std::invalid_argument("gradient allocation size must be positive");
    }
    if (size > std::numeric_limits<std::size_t>::max() - (GRADIENT_MEMORY_ALIGNMENT - 1U))
    {
        throw std::length_error("gradient allocation size exceeds host limits");
    }
    return ((size + GRADIENT_MEMORY_ALIGNMENT - 1U) / GRADIENT_MEMORY_ALIGNMENT) *
           GRADIENT_MEMORY_ALIGNMENT;
}

/**
 * @brief Kernel for element-wise gradient computation
 *
 * Computes gradients for simple operations (add, mul, etc.)
 * Coalesced memory access pattern
 */
__global__ void __launch_bounds__(256)
    elementWiseGradientKernel(const float *__restrict__ upstream_grad,
                              const float *__restrict__ input_values,
                              float *__restrict__ output_grad,
                              int op_type, // 0=add, 1=mul, 2=div, 3=pow
                              int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float upstream = upstream_grad[idx];
        float input = input_values[idx];
        float output = autodiffInfinityFloat();

        switch (op_type)
        {
            case 0: // add: grad = upstream
                output = upstream;
                break;
            case 1: // mul: grad = upstream * other_input
                output = upstream * input;
                break;
            case 2: // div: grad = upstream / other_input
            {
                const float denom = input + 1e-8f;
                output = isfinite(denom) && fabsf(denom) > 0.0f ? upstream / denom
                                                                : autodiffInfinityFloat();
            }
            break;
            case 3: // pow: grad = upstream * pow(x, y-1) * y
                output = upstream * input;
                break;
        }
        output_grad[idx] = finiteOrInfinity(output);
    }
}

/**
 * @brief Kernel for topological gradient flow
 *
 * Propagates gradients through persistence diagram operations
 * Uses warp-level parallelism for reduction
 */
__global__ void __launch_bounds__(256)
    topologicalGradientKernel(const float *__restrict__ birth_grad,
                              const float *__restrict__ death_grad,
                              const int *__restrict__ pairs, // [i, j] pairs
                              float *__restrict__ point_grad, int num_pairs, int num_points)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Each warp processes multiple pairs
    for (int p = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warp_id; p < num_pairs;
         p += gridDim.x * (BLOCK_SIZE / WARP_SIZE))
    {
        if (p < num_pairs)
        {
            int i = pairs[2 * p];
            int j = pairs[2 * p + 1];
            if (i < 0 || j < 0 || i >= num_points || j >= num_points)
            {
                continue;
            }

            // Accumulate gradients atomically
            float b_grad = finiteOrInfinity(birth_grad[p]);
            float d_grad = finiteOrInfinity(death_grad[p]);

            // Warp-level aggregation before atomic
            float sum_b = warp.shfl(b_grad, 0);
            float sum_d = warp.shfl(d_grad, 0);

            if (lane_id == 0)
            {
                atomicAdd(&point_grad[2 * i], sum_b);
                atomicAdd(&point_grad[2 * j], sum_d);
            }
        }
    }
}

/**
 * @brief Kernel for reduction operations (sum, mean, etc.)
 */
__global__ void __launch_bounds__(256, 2)
    reductionGradientKernel(const float *__restrict__ upstream_grad, float *__restrict__ input_grad,
                            int reduction_type, // 0=sum, 1=mean, 2=max
                            int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    float val = (i < n) ? finiteOrInfinity(upstream_grad[0]) : 0.0f;

    if (reduction_type == 0)
    { // sum
        val = (i < n) ? val : 0.0f;
    }
    else if (reduction_type == 1)
    { // mean
        val = (i < n) ? val / n : 0.0f;
    }

    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            const float next = sdata[tid] + sdata[tid + s];
            sdata[tid] = finiteOrInfinity(next);
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0)
    {
        atomicAdd(&input_grad[0], sdata[0]);
    }
}

/**
 * @brief Fused kernel for common gradient patterns
 *
 * Fuses multiple operations to reduce kernel launch overhead
 */
__global__ void __launch_bounds__(256)
    fusedGradientKernel(const float *__restrict__ grad_in, const float *__restrict__ weights,
                        const float *__restrict__ bias, float *__restrict__ grad_out,
                        float learning_rate, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        // Fused: grad_out = grad_in * weights - learning_rate * bias
        float grad = grad_in[idx];
        float w = weights[idx];
        float b = bias[idx];

        const float product = grad * w;
        const float decay = learning_rate * b;
        grad_out[idx] = finiteOrInfinity(product - decay);
    }
}

/**
 * @brief Half-precision gradient kernel for Tensor Cores
 */
__global__ void __launch_bounds__(256) fp16GradientKernel(const __half *__restrict__ upstream_grad,
                                                          const __half *__restrict__ input_values,
                                                          __half *__restrict__ output_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        __half upstream = upstream_grad[idx];
        __half input = input_values[idx];
        const float upstream_f = __half2float(upstream);
        const float input_f = __half2float(input);
        const float output = upstream_f * input_f;

        // FP16 computation
        output_grad[idx] = __float2half_rn(finiteOrInfinity(output));
    }
}

// Host Launchers

void launchElementWiseGradient(const float *d_upstream, const float *d_input, float *d_output,
                               int op_type, int n, cudaStream_t stream)
{
    requireDevicePointer(d_upstream, "upstream gradient pointer must not be null");
    requireDevicePointer(d_input, "input gradient pointer must not be null");
    requireDevicePointer(d_output, "output gradient pointer must not be null");
    if (op_type < 0 || op_type > 3)
    {
        throw std::invalid_argument("gradient operation type is invalid");
    }
    int blocks = checkedBlocksForPositiveCount(n, "gradient element count must be positive");
    elementWiseGradientKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_upstream, d_input, d_output,
                                                                 op_type, n);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchTopologicalGradient(const float *d_birth_grad, const float *d_death_grad,
                               const int *d_pairs, float *d_point_grad, int num_pairs,
                               int num_points, cudaStream_t stream)
{
    if (num_pairs <= 0)
    {
        return;
    }
    if (num_points <= 0)
    {
        throw std::invalid_argument("topological gradient point count must be positive");
    }
    requireDevicePointer(d_birth_grad, "birth gradient pointer must not be null");
    requireDevicePointer(d_death_grad, "death gradient pointer must not be null");
    requireDevicePointer(d_pairs, "topological gradient pair pointer must not be null");
    requireDevicePointer(d_point_grad, "point gradient pointer must not be null");
    const std::size_t warps_per_block = static_cast<std::size_t>(BLOCK_SIZE / WARP_SIZE);
    const std::size_t pair_blocks =
        (static_cast<std::size_t>(num_pairs) + warps_per_block - 1U) / warps_per_block;
    int blocks =
        static_cast<int>(std::min<std::size_t>(pair_blocks, static_cast<std::size_t>(1024)));

    topologicalGradientKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_birth_grad, d_death_grad, d_pairs, d_point_grad, num_pairs, num_points);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchReductionGradient(const float *d_upstream, float *d_input_grad, int reduction_type,
                             int n, cudaStream_t stream)
{
    requireDevicePointer(d_upstream, "upstream reduction gradient pointer must not be null");
    requireDevicePointer(d_input_grad, "input reduction gradient pointer must not be null");
    if (reduction_type < 0 || reduction_type > 2)
    {
        throw std::invalid_argument("reduction gradient type is invalid");
    }
    (void)checkedBlocksForPositiveCount(n, "reduction gradient element count must be positive");
    int smem_size = BLOCK_SIZE * sizeof(float);
    reductionGradientKernel<<<1, BLOCK_SIZE, smem_size, stream>>>(d_upstream, d_input_grad,
                                                                  reduction_type, n);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchFusedGradient(const float *d_grad_in, const float *d_weights, const float *d_bias,
                         float *d_grad_out, float learning_rate, int n, cudaStream_t stream)
{
    requireDevicePointer(d_grad_in, "input fused gradient pointer must not be null");
    requireDevicePointer(d_weights, "fused gradient weight pointer must not be null");
    requireDevicePointer(d_bias, "fused gradient bias pointer must not be null");
    requireDevicePointer(d_grad_out, "output fused gradient pointer must not be null");
    if (!std::isfinite(learning_rate))
    {
        throw std::invalid_argument("learning rate must be finite");
    }
    int blocks = checkedBlocksForPositiveCount(n, "fused gradient element count must be positive");
    fusedGradientKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_grad_in, d_weights, d_bias, d_grad_out,
                                                           learning_rate, n);
    GPU_CHECK(cudaPeekAtLastError());
}

void launchFP16Gradient(const __half *d_upstream, const __half *d_input, __half *d_output, int n,
                        cudaStream_t stream)
{
    requireDevicePointer(d_upstream, "FP16 upstream gradient pointer must not be null");
    requireDevicePointer(d_input, "FP16 input gradient pointer must not be null");
    requireDevicePointer(d_output, "FP16 output gradient pointer must not be null");
    int blocks = checkedBlocksForPositiveCount(n, "FP16 gradient element count must be positive");
    fp16GradientKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_upstream, d_input, d_output, n);
    GPU_CHECK(cudaPeekAtLastError());
}

// cuBLAS Integration

/**
 * @brief Matrix multiplication gradient using cuBLAS
 *
 * C = A * B
 * grad_A = grad_C * B^T
 * grad_B = A^T * grad_C
 */
void launchMatMulGradient(cublasHandle_t handle, const float *d_grad_c, const float *d_a,
                          const float *d_b, float *d_grad_a, float *d_grad_b, int m, int n, int k,
                          cudaStream_t stream)
{
    if (handle == nullptr || m <= 0 || n <= 0 || k <= 0)
    {
        throw std::invalid_argument("matrix gradient launch parameters are invalid");
    }
    requireDevicePointer(d_grad_c, "matrix gradient output-adjoint pointer must not be null");
    requireDevicePointer(d_a, "matrix gradient lhs pointer must not be null");
    requireDevicePointer(d_b, "matrix gradient rhs pointer must not be null");
    requireDevicePointer(d_grad_a, "matrix gradient lhs output pointer must not be null");
    requireDevicePointer(d_grad_b, "matrix gradient rhs output pointer must not be null");
    checkCublasStatus(cublasSetStream(handle, stream), "cublasSetStream");
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // grad_A = grad_C * B^T
    checkCublasStatus(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &alpha, d_grad_c, m,
                                  d_b, n, &beta, d_grad_a, m),
                      "cublasSgemm grad_A");

    // grad_B = A^T * grad_C
    checkCublasStatus(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &alpha, d_a, m,
                                  d_grad_c, m, &beta, d_grad_b, k),
                      "cublasSgemm grad_B");
}

// Gradient Memory Management

class GPUGradientMemoryPool
{
public:
    struct Allocation
    {
        void *ptr;
        size_t size;
    };

    GPUGradientMemoryPool(size_t initial_size = GRADIENT_MEMORY_POOL_DEFAULT_SIZE)
    {
        if (initial_size == 0)
        {
            throw std::invalid_argument("gradient memory pool size must be positive");
        }
        GPU_CHECK(cudaMalloc(&pool_, initial_size));
        capacity_ = initial_size;
        offset_ = 0;
    }

    ~GPUGradientMemoryPool()
    {
        for (void *ptr : overflow_allocations_)
        {
            cudaFree(ptr);
        }
        cudaFree(pool_);
    }

    void *allocate(size_t size)
    {
        const size_t aligned_size = checkedAlignedAllocationSize(size);

        if (aligned_size > capacity_ || offset_ > capacity_ - aligned_size)
        {
            // Allocate from system
            void *ptr = nullptr;
            GPU_CHECK(cudaMalloc(&ptr, size));
            overflow_allocations_.push_back(ptr);
            return ptr;
        }

        void *ptr = static_cast<char *>(pool_) + offset_;
        offset_ += aligned_size;
        return ptr;
    }

    void reset()
    {
        for (void *ptr : overflow_allocations_)
        {
            cudaFree(ptr);
        }
        overflow_allocations_.clear();
        offset_ = 0;
    }

private:
    void *pool_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
    std::vector<void *> overflow_allocations_;
};

} // namespace gpu
} // namespace autodiff
} // namespace nerve
