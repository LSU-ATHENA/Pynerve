#include <cuda_runtime.h>

#include <cmath>

namespace nerve::metrics::cuda
{

__global__ void bottleneckDistanceKernel(const double *dgm1, Size n1, const double *dgm2, Size n2,
                                         double *result)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double max_dist = 0.0;
    if (gid < static_cast<int>(n1))
    {
        for (Size j = 0; j < n2; ++j)
        {
            double dx = dgm1[2 * gid] - dgm2[2 * j];
            double dy = dgm1[2 * gid + 1] - dgm2[2 * j + 1];
            double d = sqrt(dx * dx + dy * dy);
            if (j == 0 || d > max_dist)
                max_dist = d;
        }
    }
    sdata[tid] = max_dist;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        result[blockIdx.x] = sdata[0];
}

__global__ void wassersteinDistanceKernel(const double *dgm1, Size n1, const double *dgm2, Size n2,
                                          double p, double *result)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= static_cast<int>(n1))
        return;

    double sum = 0.0;
    for (Size j = 0; j < n2; ++j)
    {
        double dx = dgm1[2 * gid] - dgm2[2 * j];
        double dy = dgm1[2 * gid + 1] - dgm2[2 * j + 1];
        sum += pow(sqrt(dx * dx + dy * dy), p);
    }
    result[gid] = sum;
}

cudaError_t launchBottleneckDistance(const double *d_dgm1, Size n1, const double *d_dgm2, Size n2,
                                     double *d_result, cudaStream_t stream)
{
    if (n1 == 0 || n2 == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n1) + 255) / 256);
    bottleneckDistanceKernel<<<grid, block, 0, stream>>>(d_dgm1, n1, d_dgm2, n2, d_result);
    return cudaGetLastError();
}

cudaError_t launchWassersteinDistance(const double *d_dgm1, Size n1, const double *d_dgm2, Size n2,
                                      double p, double *d_result, cudaStream_t stream)
{
    if (n1 == 0 || n2 == 0)
        return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(n1) + 255) / 256);
    wassersteinDistanceKernel<<<grid, block, 0, stream>>>(d_dgm1, n1, d_dgm2, n2, p, d_result);
    return cudaGetLastError();
}

} // namespace nerve::metrics::cuda
