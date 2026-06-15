#include "cuda/cuda_error.hpp"
#include "nerve/config.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>

#if defined(NERVE_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace nerve::streaming::gpu
{

#if defined(NERVE_HAS_CUDA)

struct MultiStreamContext
{
    int num_gpus;
    int num_streams_per_gpu;
    std::vector<int> device_ids;
    std::vector<cudaStream_t> streams;

    static cudaError_t create(int gpus, int streams_per_gpu, MultiStreamContext &ctx)
    {
        ctx.num_gpus = gpus;
        ctx.num_streams_per_gpu = streams_per_gpu;
        ctx.device_ids.resize(gpus);
        ctx.streams.resize(gpus * streams_per_gpu);

        for (int g = 0; g < gpus; ++g)
        {
            ctx.device_ids[g] = g;
            cudaSetDevice(g);
            for (int s = 0; s < streams_per_gpu; ++s)
            {
                cudaStreamCreate(&ctx.streams[g * streams_per_gpu + s]);
            }
        }
        return cudaSuccess;
    }

    static void destroy(MultiStreamContext &ctx)
    {
        for (int g = 0; g < ctx.num_gpus; ++g)
        {
            cudaSetDevice(g);
            for (int s = 0; s < ctx.num_streams_per_gpu; ++s)
            {
                cudaStreamDestroy(ctx.streams[g * ctx.num_streams_per_gpu + s]);
            }
        }
    }
};

cudaError_t scatterPoints(const double *h_points, Size n, Size dim, int num_gpus,
                          std::vector<double *> &d_points)
{
    MultiStreamContext ctx;
    auto err = MultiStreamContext::create(num_gpus, 1, ctx);
    if (err != cudaSuccess)
        return err;

    Size points_per_gpu = n / num_gpus;
    Size extra = n % num_gpus;
    d_points.resize(num_gpus);

    Size offset = 0;
    for (int g = 0; g < num_gpus; ++g)
    {
        Size count = points_per_gpu + (g < static_cast<int>(extra) ? 1 : 0);
        Size bytes = count * dim * sizeof(double);

        cudaSetDevice(g);
        cudaMalloc(&d_points[g], bytes);
        cudaMemcpyAsync(d_points[g], h_points + offset * dim, bytes, cudaMemcpyHostToDevice,
                        ctx.streams[g * ctx.num_streams_per_gpu]);
        offset += count;
    }

    for (int g = 0; g < num_gpus; ++g)
    {
        cudaSetDevice(g);
        cudaStreamSynchronize(ctx.streams[g * ctx.num_streams_per_gpu]);
    }

    MultiStreamContext::destroy(ctx);
    return cudaSuccess;
}

cudaError_t gatherResults(const std::vector<int *> &d_results, const std::vector<Size> &counts,
                          int num_gpus, int *h_result)
{
    MultiStreamContext ctx;
    auto err = MultiStreamContext::create(num_gpus, 1, ctx);
    if (err != cudaSuccess)
        return err;

    Size offset = 0;
    for (int g = 0; g < num_gpus; ++g)
    {
        Size bytes = counts[g] * sizeof(int);
        if (bytes == 0)
            continue;
        cudaSetDevice(g);
        cudaMemcpyAsync(h_result + offset, d_results[g], bytes, cudaMemcpyDeviceToHost,
                        ctx.streams[g * ctx.num_streams_per_gpu]);
        offset += counts[g];
    }

    for (int g = 0; g < num_gpus; ++g)
    {
        cudaSetDevice(g);
        cudaStreamSynchronize(ctx.streams[g * ctx.num_streams_per_gpu]);
    }

    MultiStreamContext::destroy(ctx);
    return cudaSuccess;
}

#else

cudaError_t scatterPoints(const double *, Size, Size, int, std::vector<double *> &)
{
    return cudaErrorNotSupported;
}

cudaError_t gatherResults(const std::vector<int *> &, const std::vector<Size> &, int, int *)
{
    return cudaErrorNotSupported;
}

#endif

} // namespace nerve::streaming::gpu
