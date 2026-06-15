#include "nerve/gpu/consumer_config.hpp"

#include <cstdio>
#include <sstream>

namespace nerve::gpu::consumer
{
namespace
{

constexpr std::size_t kBytesPerGiB = 1024ULL * 1024ULL * 1024ULL;

} // namespace

std::string formatConfig(const ConsumerConfig &config)
{
    std::ostringstream oss;
    oss << "GPU: " << config.gpu_name << " | SM=" << config.sm_version
        << " | VRAM=" << (config.total_vram_bytes / kBytesPerGiB) << " GiB"
        << " | free=" << (config.free_vram_bytes / kBytesPerGiB) << " GiB"
        << " | tier=" << static_cast<int>(config.optimization_tier)
        << " | block=" << config.block_size << " | batch=" << config.batch_size
        << " | precision=" << config.precision_mode
        << " | max_points=" << config.max_points_full_gpu
        << " | streaming=" << (config.use_streaming ? "yes" : "no");
    return oss.str();
}

std::vector<std::string> getOptimizedGPUs()
{
    return {
        "NVIDIA RTX 4090",    "NVIDIA RTX 4080", "NVIDIA RTX 4070 Ti", "NVIDIA RTX 4070",
        "NVIDIA RTX 4060 Ti", "NVIDIA RTX 4060", "NVIDIA RTX 3090 Ti", "NVIDIA RTX 3090",
        "NVIDIA RTX 3080 Ti", "NVIDIA RTX 3080", "NVIDIA RTX 3070 Ti", "NVIDIA RTX 3070",
        "NVIDIA RTX 3060 Ti", "NVIDIA RTX 3060",
    };
}

std::vector<std::string> getSupportedGPUs()
{
    auto supported = getOptimizedGPUs();
    supported.push_back("NVIDIA RTX 2080 Ti");
    supported.push_back("NVIDIA RTX 2080");
    supported.push_back("NVIDIA RTX 2070");
    supported.push_back("NVIDIA RTX 2060");
    return supported;
}

void printGPUReport()
{
    ConsumerConfig config = detectGPU();
    if (config.isSupported())
    {
        (void)autoTune(config);
    }
    std::printf("%s\n", formatConfig(config).c_str());
}

} // namespace nerve::gpu::consumer
