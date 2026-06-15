
#include "nerve/gpu/gpu_error.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <limits>

namespace nerve
{
namespace gpu
{
namespace kernels
{

constexpr double DIAGRAM_CUDA_FALLBACK_PENALTY = 1.0e300;
constexpr double DIAGRAM_CUDA_DUPLICATE_TOLERANCE = 1e-12;
constexpr size_t DIAGRAM_CUDA_BLOCK_SIZE = 256;

// Structure to hold persistence pair data on GPU
struct GPUPersistencePair
{
    double birth;
    double death;
    bool is_infinite;
};

__device__ inline double finitePenalty(double penalty)
{
    return (isfinite(penalty) && penalty >= 0.0) ? penalty : DIAGRAM_CUDA_FALLBACK_PENALTY;
}

__device__ inline double finiteCostOrPenalty(double cost, double penalty)
{
    return (isfinite(cost) && cost >= 0.0) ? cost : finitePenalty(penalty);
}

__device__ inline size_t atomicAddSizeT(size_t *address, size_t value)
{
    if constexpr (sizeof(size_t) == sizeof(unsigned long long))
    {
        return static_cast<size_t>(atomicAdd(reinterpret_cast<unsigned long long *>(address),
                                             static_cast<unsigned long long>(value)));
    }
    else
    {
        return static_cast<size_t>(
            atomicAdd(reinterpret_cast<unsigned int *>(address), static_cast<unsigned int>(value)));
    }
}

// Kernel to compute cost matrix for persistence diagram distance
// Each thread computes one element of the cost matrix
__global__ __launch_bounds__(DIAGRAM_CUDA_BLOCK_SIZE) void computeDiagramCostMatrixKernel(
    const GPUPersistencePair *pairs1, const GPUPersistencePair *pairs2, double *cost_matrix,
    size_t n1, size_t n2,
    size_t n, // n = n1 + n2 (padded matrix size)
    double large_penalty)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = n * n;

    if (idx >= total_elements)
    {
        return;
    }

    size_t row = idx / n;
    size_t col = idx % n;

    double cost = 0.0;
    const double safe_penalty = finitePenalty(large_penalty);

    // Case 1: (i < n1, j < n2) - pair-to-pair distance
    if (row < n1 && col < n2)
    {
        const GPUPersistencePair &p1 = pairs1[row];
        const GPUPersistencePair &p2 = pairs2[col];

        if (p1.is_infinite && p2.is_infinite)
        {
            cost = finiteCostOrPenalty(fabs(p1.birth - p2.birth), safe_penalty);
        }
        else if (p1.is_infinite != p2.is_infinite)
        {
            cost = safe_penalty;
        }
        else
        {
            double birth_diff = fabs(p1.birth - p2.birth);
            double death_diff = fabs(p1.death - p2.death);
            cost = finiteCostOrPenalty(fmax(birth_diff, death_diff), safe_penalty);
        }
    }
    // Case 2: (i < n1, j >= n2) - pair1 to diagonal
    else if (row < n1 && col >= n2)
    {
        // Check if this is the diagonal connection for this pair
        if (col == n2 + row)
        {
            const GPUPersistencePair &p = pairs1[row];
            if (p.is_infinite)
            {
                cost = safe_penalty;
            }
            else
            {
                cost = finiteCostOrPenalty(fabs(p.death - p.birth) * 0.5, safe_penalty);
            }
        }
        else
        {
            cost = 0.0; // Auxiliary-to-auxiliary
        }
    }
    // Case 3: (i >= n1, j < n2) - pair2 to diagonal
    else if (row >= n1 && col < n2)
    {
        // Check if this is the diagonal connection for this pair
        if (row == n1 + col)
        {
            const GPUPersistencePair &p = pairs2[col];
            if (p.is_infinite)
            {
                cost = safe_penalty;
            }
            else
            {
                cost = finiteCostOrPenalty(fabs(p.death - p.birth) * 0.5, safe_penalty);
            }
        }
        else
        {
            cost = 0.0; // Auxiliary-to-auxiliary
        }
    }
    // Case 4: (i >= n1, j >= n2) - auxiliary-to-auxiliary
    else
    {
        cost = 0.0;
    }

    cost_matrix[row * n + col] = finiteCostOrPenalty(cost, safe_penalty);
}

// Kernel to find candidate threshold values for bottleneck distance
// Uses parallel reduction to find unique distance values
__global__ __launch_bounds__(DIAGRAM_CUDA_BLOCK_SIZE) void extractCandidateThresholdsKernel(
    const double *cost_matrix, size_t n, double *candidates, size_t *candidate_count,
    size_t max_candidates)
{
    extern __shared__ double shared_mem[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    shared_mem[tid] = -1.0;

    // Each thread loads one element
    if (idx < n * n)
    {
        double val = cost_matrix[idx];
        if (val > 0 && isfinite(val))
        {
            shared_mem[tid] = val;
        }
    }

    __syncthreads();

    // Parallel reduction collects valid values to output
    // Full sorting-based deduplication can be done on host
    if (tid == 0)
    {
        size_t count = 0;
        for (size_t i = 0; i < blockDim.x && count < max_candidates; ++i)
        {
            double val = shared_mem[i];
            if (val >= 0)
            {
                // Check if already in candidates (simple linear search)
                bool found = false;
                for (size_t j = 0; j < count; ++j)
                {
                    if (fabs(candidates[j] - val) < DIAGRAM_CUDA_DUPLICATE_TOLERANCE)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    size_t pos = atomicAddSizeT(candidate_count, 1);
                    if (pos < max_candidates)
                    {
                        candidates[pos] = val;
                    }
                }
            }
        }
    }
}

// Host wrapper for cost matrix computation
void launchDiagramCostMatrixKernel(const GPUPersistencePair *d_pairs1,
                                   const GPUPersistencePair *d_pairs2, double *d_cost_matrix,
                                   size_t n1, size_t n2, size_t n, double large_penalty,
                                   cudaStream_t stream)
{
    size_t total_elements = n * n;
    size_t block_size = DIAGRAM_CUDA_BLOCK_SIZE;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    computeDiagramCostMatrixKernel<<<grid_size, block_size, 0, stream>>>(
        d_pairs1, d_pairs2, d_cost_matrix, n1, n2, n, large_penalty);
    GPU_CHECK(cudaPeekAtLastError());
}

} // namespace kernels
} // namespace gpu
} // namespace nerve
