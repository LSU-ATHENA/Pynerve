#include "nerve/determinism.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace nerve::determinism
{

__global__ void blockReduceKernel(const double *input, double *output, Size n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double val = (gid < n) ? input[gid] : 0.0;
    double result = blockReduceSum<256>(val);
    if (threadIdx.x == 0)
        output[blockIdx.x] = result;
}

cudaError_t launchDeterministicReduce(const double *d_in, double *d_out, Size n, cudaStream_t s)
{
    if (n == 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    int grid = (static_cast<int>(n) + kBlock - 1) / kBlock;
    if (grid == 1)
    {
        blockReduceKernel<<<1, kBlock, kBlock * sizeof(double), s>>>(d_in, d_out, n);
        return cudaGetLastError();
    }
    double *d_tmp = nullptr;
    cudaMallocAsync(&d_tmp, grid * sizeof(double), s);
    blockReduceKernel<<<grid, kBlock, kBlock * sizeof(double), s>>>(d_in, d_tmp, n);
    blockReduceKernel<<<1, kBlock, kBlock * sizeof(double), s>>>(d_tmp, d_out, grid);
    cudaFreeAsync(d_tmp, s);
    return cudaGetLastError();
}

} // namespace nerve::determinism
