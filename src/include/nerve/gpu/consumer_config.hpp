#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace nerve::gpu::consumer
{

enum class GPUArchitecture
{
    UNKNOWN = 0,
    TURING = 75,
    AMPERE = 86,
    ADA = 89,
    HOPPER = 90,
    BLACKWELL = 100
};

enum class VRAMTier
{
    MINIMAL = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3
};

enum class OptimizationTier
{
    UNSUPPORTED = 0,
    UNOPTIMIZED = 1,
    BASIC = 2,
    STANDARD = 3,
    HIGH = 4,
    MAXIMUM = 5
};

struct ConsumerConfig
{
    std::string gpu_name;
    GPUArchitecture architecture = GPUArchitecture::UNKNOWN;
    int sm_version = 0;

    std::size_t total_vram_bytes = 0;
    std::size_t free_vram_bytes = 0;
    int l2_cache_kb = 0;
    int num_sms = 0;
    int tensor_cores = 0;

    bool supports_fp8 = false;
    bool supports_tf32 = false;
    bool supports_tensor_cores = false;
    VRAMTier vram_tier = VRAMTier::MINIMAL;
    OptimizationTier optimization_tier = OptimizationTier::UNSUPPORTED;

    int block_size = 128;
    int batch_size = 1;
    std::string precision_mode = "FP32";
    std::size_t max_points_full_gpu = 0;
    std::size_t chunk_size = 0;
    bool use_streaming = true;

    [[nodiscard]] bool isOptimized() const noexcept
    {
        return optimization_tier >= OptimizationTier::BASIC;
    }

    [[nodiscard]] bool isSupported() const noexcept
    {
        return optimization_tier >= OptimizationTier::UNOPTIMIZED;
    }

    [[nodiscard]] bool needsStreaming() const noexcept { return vram_tier <= VRAMTier::LOW; }

    [[nodiscard]] bool hasTensorCores() const noexcept { return supports_tensor_cores; }
};

[[nodiscard]] ConsumerConfig detectGPU();
bool autoTune(ConsumerConfig &config);
[[nodiscard]] ConsumerConfig getPresetConfig(const std::string &gpu_name);
[[nodiscard]] std::string recommendAlgorithm(const ConsumerConfig &config, std::size_t num_points,
                                             std::size_t point_dim);
[[nodiscard]] std::string formatConfig(const ConsumerConfig &config);
[[nodiscard]] bool isGPUSupported(const std::string &gpu_name);
[[nodiscard]] std::vector<std::string> getOptimizedGPUs();
[[nodiscard]] std::vector<std::string> getSupportedGPUs();

[[nodiscard]] inline ConsumerConfig setup()
{
    auto config = detectGPU();
    if (config.isOptimized())
    {
        autoTune(config);
    }
    return config;
}

void printGPUReport();

} // namespace nerve::gpu::consumer
