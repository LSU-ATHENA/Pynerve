#include "nerve/config.hpp"
#include "nerve/persistence/cuda/cuda_multi_gpu.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace nerve::gpu::multi
{

#if defined(NERVE_HAS_CUDA)

static bool hasMultiGpuSupport()
{
    int count = 0;
    cudaGetDeviceCount(&count);
    return count >= 2;
}

MultiGpuInfo detectMultiGpuConfiguration()
{
    MultiGpuInfo info;
    info.num_gpus = 0;
    info.nvlink_available = false;

    if (!hasMultiGpuSupport())
        return info;

    cudaGetDeviceCount(&info.num_gpus);
    for (int i = 0; i < info.num_gpus; ++i)
    {
        GpuInfo gpu;
        gpu.id = i;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        gpu.name = prop.name;
        gpu.memory_mb = prop.totalGlobalMem / (1024 * 1024);
        gpu.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
        info.gpus.push_back(gpu);
    }

    info.p2p_matrix.resize(info.num_gpus * info.num_gpus, false);
    for (int i = 0; i < info.num_gpus; ++i)
    {
        cudaSetDevice(i);
        for (int j = i + 1; j < info.num_gpus; ++j)
        {
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            if (can_access)
            {
                info.p2p_matrix[i * info.num_gpus + j] = true;
                info.p2p_matrix[j * info.num_gpus + i] = true;
                info.nvlink_available = true;
            }
        }
    }

    return info;
}

std::vector<int> distributeIndices(Size total_count, int num_gpus)
{
    std::vector<int> counts(num_gpus);
    Size base = total_count / num_gpus;
    Size extra = total_count % num_gpus;
    for (int g = 0; g < num_gpus; ++g)
        counts[g] = static_cast<int>(base + (g < static_cast<int>(extra) ? 1 : 0));
    return counts;
}

std::vector<Size> distributeByTopology(Size total_count, const MultiGpuInfo &info)
{
    std::vector<Size> counts(info.num_gpus);
    if (info.num_gpus <= 1)
    {
        counts[0] = total_count;
        return counts;
    }
    Size base = total_count / info.num_gpus;
    Size extra = total_count % info.num_gpus;
    for (int g = 0; g < info.num_gpus; ++g)
        counts[g] = base + (g < static_cast<int>(extra) ? 1 : 0);
    return counts;
}

#else

MultiGpuInfo detectMultiGpuConfiguration()
{
    MultiGpuInfo info;
    info.num_gpus = 1;
    info.nvlink_available = false;
    return info;
}
std::vector<int> distributeIndices(Size, int)
{
    return {};
}
std::vector<Size> distributeByTopology(Size, const MultiGpuInfo &)
{
    return {};
}

#endif

} // namespace nerve::gpu::multi
