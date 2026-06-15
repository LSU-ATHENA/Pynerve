#include "dmt_gpu.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nerve::dmt::gpu
{

__global__ void computeGradientKernel(const int *cells, size_t n_cells, const int *pairs,
                                      size_t n_pairs, int *gradient, int *critical)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int>(n_cells))
        return;

    bool has_lower = false;
    bool has_higher = false;
    int cell_val = cells[idx];

    for (size_t p = 0; p < n_pairs; ++p)
    {
        int a = pairs[2 * p];
        int b = pairs[2 * p + 1];
        if (a == static_cast<int>(idx))
        {
            has_higher = true;
            if (has_lower)
                break;
        }
        if (b == static_cast<int>(idx))
        {
            has_lower = true;
            if (has_higher)
                break;
        }
    }

    if (has_lower && has_higher)
    {
        gradient[idx] = 0;
        critical[idx] = 0;
    }
    else if (has_lower)
    {
        gradient[idx] = 1;
        critical[idx] = 0;
    }
    else if (has_higher)
    {
        gradient[idx] = -1;
        critical[idx] = 0;
    }
    else
    {
        gradient[idx] = 0;
        critical[idx] = 1;
    }
}

DiscreteGradientQuadrant computeGradientGPU(const int *d_cells, size_t n_cells, const int *d_pairs,
                                            size_t n_pairs, cudaStream_t stream)
{
    DiscreteGradientQuadrant result;
    cudaMalloc(&result.d_gradient, n_cells * sizeof(int));
    cudaMalloc(&result.d_critical, n_cells * sizeof(int));
    result.grad_size = n_cells;

    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n_cells) + 255) / 256);
    computeGradientKernel<<<grid, block, 0, stream>>>(d_cells, n_cells, d_pairs, n_pairs,
                                                      result.d_gradient, result.d_critical);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess)
    {
        freeGradientGPU(result);
        throw std::runtime_error("computeGradientKernel launch failed: " +
                                 std::string(cudaGetErrorString(launch_status)));
    }

    return result;
}

void freeGradientGPU(DiscreteGradientQuadrant &grad)
{
    if (grad.d_gradient)
        cudaFree(grad.d_gradient);
    if (grad.d_critical)
        cudaFree(grad.d_critical);
    grad.d_gradient = nullptr;
    grad.d_critical = nullptr;
    grad.grad_size = 0;
}

} // namespace nerve::dmt::gpu
