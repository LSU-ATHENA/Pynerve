#include "nerve/config.hpp"
#include "nerve/gpu/consumer_config.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#if defined(NERVE_HAS_CUDA) || defined(NERVE_HAS_CUDA_RUNTIME)
#define NERVE_CONSUMER_CONFIG_HAS_CUDA 1
#include <cuda_runtime.h>
#else
#define NERVE_CONSUMER_CONFIG_HAS_CUDA 0
#endif

namespace nerve::gpu::consumer
{

namespace
{

constexpr std::size_t kBytesPerGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMemorySafetyNumerator = 7;
constexpr std::size_t kMemorySafetyDenominator = 10;
#if NERVE_CONSUMER_CONFIG_HAS_CUDA
constexpr int kCudaDeviceIndex = 0;
#endif

enum class KnownGpuModel
{
    kUnknown = 0,
    kRtx4090,
    kRtx4080,
    kRtx4070Ti,
    kRtx4070,
    kRtx4060Ti,
    kRtx4060,
    kRtx3090Ti,
    kRtx3090,
    kRtx3080Ti,
    kRtx3080,
    kRtx3070Ti,
    kRtx3070,
    kRtx3060Ti,
    kRtx3060,
    kRtx20Series,
};

struct ModelPreset
{
    KnownGpuModel model;
    const char *token;
    GPUArchitecture architecture;
    OptimizationTier tier;
    int block_size;
    int batch_size;
    const char *precision;
};

constexpr std::array<ModelPreset, 15> kModelPresets{{
    {KnownGpuModel::kRtx4090, "4090", GPUArchitecture::ADA, OptimizationTier::MAXIMUM, 512, 8,
     "TF32"},
    {KnownGpuModel::kRtx4080, "4080", GPUArchitecture::ADA, OptimizationTier::HIGH, 512, 4, "FP16"},
    {KnownGpuModel::kRtx4070Ti, "4070 TI", GPUArchitecture::ADA, OptimizationTier::HIGH, 512, 4,
     "FP16"},
    {KnownGpuModel::kRtx4070, "4070", GPUArchitecture::ADA, OptimizationTier::HIGH, 256, 4, "TF32"},
    {KnownGpuModel::kRtx4060Ti, "4060 TI", GPUArchitecture::ADA, OptimizationTier::STANDARD, 256, 2,
     "TF32"},
    {KnownGpuModel::kRtx4060, "4060", GPUArchitecture::ADA, OptimizationTier::STANDARD, 128, 1,
     "FP32"},
    {KnownGpuModel::kRtx3090Ti, "3090 TI", GPUArchitecture::AMPERE, OptimizationTier::HIGH, 512, 4,
     "FP16"},
    {KnownGpuModel::kRtx3090, "3090", GPUArchitecture::AMPERE, OptimizationTier::HIGH, 512, 4,
     "FP16"},
    {KnownGpuModel::kRtx3080Ti, "3080 TI", GPUArchitecture::AMPERE, OptimizationTier::STANDARD, 512,
     4, "FP16"},
    {KnownGpuModel::kRtx3080, "3080", GPUArchitecture::AMPERE, OptimizationTier::STANDARD, 256, 4,
     "TF32"},
    {KnownGpuModel::kRtx3070Ti, "3070 TI", GPUArchitecture::AMPERE, OptimizationTier::STANDARD, 256,
     2, "TF32"},
    {KnownGpuModel::kRtx3070, "3070", GPUArchitecture::AMPERE, OptimizationTier::STANDARD, 256, 2,
     "TF32"},
    {KnownGpuModel::kRtx3060Ti, "3060 TI", GPUArchitecture::AMPERE, OptimizationTier::BASIC, 256, 1,
     "FP32"},
    {KnownGpuModel::kRtx3060, "3060", GPUArchitecture::AMPERE, OptimizationTier::BASIC, 128, 1,
     "FP32"},
    {KnownGpuModel::kRtx20Series, "20", GPUArchitecture::TURING, OptimizationTier::UNOPTIMIZED, 128,
     1, "FP32"},
}};

[[nodiscard]] std::string toUpper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

#if NERVE_CONSUMER_CONFIG_HAS_CUDA
[[nodiscard]] GPUArchitecture architectureFromSM(const int sm_version)
{
    if (sm_version >= static_cast<int>(GPUArchitecture::BLACKWELL))
    {
        return GPUArchitecture::BLACKWELL;
    }
    if (sm_version >= static_cast<int>(GPUArchitecture::HOPPER))
    {
        return GPUArchitecture::HOPPER;
    }
    if (sm_version >= static_cast<int>(GPUArchitecture::ADA))
    {
        return GPUArchitecture::ADA;
    }
    if (sm_version >= static_cast<int>(GPUArchitecture::AMPERE))
    {
        return GPUArchitecture::AMPERE;
    }
    if (sm_version >= static_cast<int>(GPUArchitecture::TURING))
    {
        return GPUArchitecture::TURING;
    }
    return GPUArchitecture::UNKNOWN;
}

[[nodiscard]] VRAMTier vramTierFromBytes(const std::size_t total_vram_bytes)
{
    const std::size_t vram_gib = total_vram_bytes / kBytesPerGiB;
    if (vram_gib >= 24)
    {
        return VRAMTier::HIGH;
    }
    if (vram_gib >= 12)
    {
        return VRAMTier::MEDIUM;
    }
    if (vram_gib >= 8)
    {
        return VRAMTier::LOW;
    }
    return VRAMTier::MINIMAL;
}

[[nodiscard]] KnownGpuModel parseGpuModel(const std::string &gpu_name_upper)
{
    for (const auto &preset : kModelPresets)
    {
        if (gpu_name_upper.find(preset.token) != std::string::npos)
        {
            return preset.model;
        }
    }
    return KnownGpuModel::kUnknown;
}

[[nodiscard]] const ModelPreset *findPreset(const KnownGpuModel model)
{
    for (const auto &preset : kModelPresets)
    {
        if (preset.model == model)
        {
            return &preset;
        }
    }
    return nullptr;
}

[[nodiscard]] int estimateTensorCores(const int num_sms, const GPUArchitecture architecture)
{
    if (num_sms <= 0)
    {
        return 0;
    }
    switch (architecture)
    {
        case GPUArchitecture::ADA:
            return num_sms * 4;
        case GPUArchitecture::AMPERE:
            return num_sms * 4;
        case GPUArchitecture::TURING:
            return num_sms * 2;
        default:
            return 0;
    }
}

[[nodiscard]] std::size_t estimateMaxPointsFullGpu(const ConsumerConfig &config)
{
    if (config.free_vram_bytes == 0)
    {
        return 0;
    }
    const long double bytes_budget = static_cast<long double>(config.free_vram_bytes) *
                                     static_cast<long double>(kMemorySafetyNumerator) /
                                     static_cast<long double>(kMemorySafetyDenominator);

    const long double bytes_per_pair = static_cast<long double>(sizeof(float));
    const long double bounded_pairs = bytes_budget / bytes_per_pair;
    const long double estimated_n = std::sqrt(2.0L * bounded_pairs);
    if (estimated_n <= 0.0L)
    {
        return 0;
    }
    constexpr long double kUpperBound = 1.0e9L;
    const long double clamped = std::min(estimated_n, kUpperBound);
    return static_cast<std::size_t>(clamped);
}
#endif

[[nodiscard]] ConsumerConfig makeUnsupportedConfig()
{
    ConsumerConfig config{};
    config.gpu_name = "UNSUPPORTED";
    config.architecture = GPUArchitecture::UNKNOWN;
    config.sm_version = 0;
    config.total_vram_bytes = 0;
    config.free_vram_bytes = 0;
    config.l2_cache_kb = 0;
    config.num_sms = 0;
    config.tensor_cores = 0;
    config.supports_fp8 = false;
    config.supports_tf32 = false;
    config.supports_tensor_cores = false;
    config.vram_tier = VRAMTier::MINIMAL;
    config.optimization_tier = OptimizationTier::UNSUPPORTED;
    config.block_size = 128;
    config.batch_size = 1;
    config.precision_mode = "FP32";
    config.max_points_full_gpu = 0;
    config.chunk_size = 0;
    config.use_streaming = true;
    return config;
}

[[nodiscard]] ConsumerConfig presetFromName(const std::string &gpu_name)
{
    ConsumerConfig config = makeUnsupportedConfig();
    config.gpu_name = gpu_name.empty() ? "UNKNOWN GPU" : gpu_name;
    config.optimization_tier = OptimizationTier::UNOPTIMIZED;
    config.architecture = GPUArchitecture::UNKNOWN;

    const std::string upper = toUpper(gpu_name);
    const KnownGpuModel model = parseGpuModel(upper);
    const ModelPreset *preset = findPreset(model);
    if (preset == nullptr)
    {
        return config;
    }
    config.architecture = preset->architecture;
    config.optimization_tier = preset->tier;
    config.block_size = preset->block_size;
    config.batch_size = preset->batch_size;
    config.precision_mode = preset->precision;
    return config;
}

} // namespace

ConsumerConfig detectGPU()
{
    ConsumerConfig config = makeUnsupportedConfig();

#if NERVE_CONSUMER_CONFIG_HAS_CUDA
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        config.gpu_name = "CUDA device missing";
        return config;
    }

    const int device_index = std::min(kCudaDeviceIndex, device_count - 1);
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device_index) != cudaSuccess)
    {
        config.gpu_name = "CUDA device query failed";
        return config;
    }

    int previous_device = 0;
    (void)cudaGetDevice(&previous_device);
    const cudaError_t set_device_result = cudaSetDevice(device_index);
    if (set_device_result != cudaSuccess)
    {
        config.gpu_name = "CUDA device selection failed";
        return config;
    }

    std::size_t free_mem = 0;
    std::size_t total_mem = static_cast<std::size_t>(props.totalGlobalMem);
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess)
    {
        free_mem = total_mem;
    }

    config.gpu_name = props.name;
    config.sm_version = props.major * 10 + props.minor;
    config.architecture = architectureFromSM(config.sm_version);
    config.total_vram_bytes = total_mem;
    config.free_vram_bytes = free_mem;
    config.l2_cache_kb = static_cast<int>(props.l2CacheSize / 1024);
    config.num_sms = props.multiProcessorCount;
    config.tensor_cores = estimateTensorCores(props.multiProcessorCount, config.architecture);
    config.supports_fp8 = config.architecture >= GPUArchitecture::HOPPER;
    config.supports_tf32 = config.architecture >= GPUArchitecture::AMPERE;
    config.supports_tensor_cores = config.tensor_cores > 0;
    config.vram_tier = vramTierFromBytes(config.total_vram_bytes);
    config.max_points_full_gpu = estimateMaxPointsFullGpu(config);
    config.chunk_size = std::max<std::size_t>(config.max_points_full_gpu / 4, 4096U);
    config.use_streaming = config.needsStreaming();

    const ConsumerConfig preset = presetFromName(config.gpu_name);
    config.optimization_tier = preset.optimization_tier;
    config.block_size = preset.block_size;
    config.batch_size = preset.batch_size;
    config.precision_mode = preset.precision_mode;

    if (config.optimization_tier == OptimizationTier::UNSUPPORTED &&
        config.architecture >= GPUArchitecture::TURING)
    {
        config.optimization_tier = OptimizationTier::UNOPTIMIZED;
    }
    if (config.supports_fp8 && config.precision_mode == "FP32")
    {
        config.precision_mode = "TF32";
    }

    (void)cudaSetDevice(previous_device);
    return config;
#else
    config.gpu_name = "CUDA support not built";
    return config;
#endif
}

bool autoTune(ConsumerConfig &config)
{
    if (!config.isSupported())
    {
        return false;
    }
    if (config.block_size <= 0)
    {
        config.block_size = 128;
    }
    if (config.batch_size <= 0)
    {
        config.batch_size = 1;
    }

    std::array<int, 3> candidates{128, 256, 512};
    int best_block = candidates.front();
    double best_score = std::numeric_limits<double>::lowest();
    for (int block : candidates)
    {
        const double warp_utilization = static_cast<double>(block) / 512.0;
        const double sm_scale = static_cast<double>(std::max(config.num_sms, 1));
        const double memory_weight = config.use_streaming ? 0.8 : 1.1;
        const double score = warp_utilization * std::log2(sm_scale + 1.0) * memory_weight;
        if (score > best_score)
        {
            best_score = score;
            best_block = block;
        }
    }
    config.block_size = best_block;

    const std::size_t free_gib = config.free_vram_bytes / kBytesPerGiB;
    if (free_gib >= 20)
    {
        config.batch_size = 8;
    }
    else if (free_gib >= 12)
    {
        config.batch_size = 4;
    }
    else if (free_gib >= 8)
    {
        config.batch_size = 2;
    }
    else
    {
        config.batch_size = 1;
    }

    if (config.supports_fp8 && config.optimization_tier >= OptimizationTier::HIGH)
    {
        config.precision_mode = "FP8";
    }
    else if (config.supports_tf32)
    {
        config.precision_mode = "TF32";
    }
    else
    {
        config.precision_mode = "FP32";
    }
    return true;
}

ConsumerConfig getPresetConfig(const std::string &gpu_name)
{
    ConsumerConfig config = presetFromName(gpu_name);
    config.total_vram_bytes = 0;
    config.free_vram_bytes = 0;
    config.l2_cache_kb = 0;
    config.num_sms = 0;
    config.tensor_cores = 0;
    config.supports_fp8 = config.architecture >= GPUArchitecture::HOPPER;
    config.supports_tf32 = config.architecture >= GPUArchitecture::AMPERE;
    config.supports_tensor_cores = config.architecture >= GPUArchitecture::TURING;
    config.vram_tier = VRAMTier::LOW;
    config.max_points_full_gpu = 0;
    config.chunk_size = 0;
    config.use_streaming = true;
    return config;
}

std::string recommendAlgorithm(const ConsumerConfig &config, const std::size_t num_points,
                               const std::size_t point_dim)
{
    if (num_points == 0 || point_dim == 0 || !config.isSupported())
    {
        return "unsupported";
    }
    const long double point_bytes = static_cast<long double>(num_points) *
                                    static_cast<long double>(point_dim) *
                                    static_cast<long double>(sizeof(float));
    const long double pairwise_bytes =
        (static_cast<long double>(num_points) * static_cast<long double>(num_points - 1) / 2.0L) *
        static_cast<long double>(sizeof(float));
    const long double estimated_total = point_bytes + pairwise_bytes;
    const long double free_budget = static_cast<long double>(config.free_vram_bytes) *
                                    static_cast<long double>(kMemorySafetyNumerator) /
                                    static_cast<long double>(kMemorySafetyDenominator);

    if (estimated_total <= free_budget && !config.needsStreaming())
    {
        return "full-gpu";
    }
    if (estimated_total <= static_cast<long double>(config.total_vram_bytes))
    {
        return config.supports_tensor_cores ? "chunked-tensorcore" : "chunked-gpu";
    }
    return "streaming-hybrid";
}

bool isGPUSupported(const std::string &gpu_name)
{
    const std::string upper = toUpper(gpu_name);
    if (upper.find("NVIDIA") == std::string::npos || upper.find("RTX") == std::string::npos)
    {
        return false;
    }
    return parseGpuModel(upper) != KnownGpuModel::kUnknown;
}

} // namespace nerve::gpu::consumer
