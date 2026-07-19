
#include "nerve/gpu/tuning_cache.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>

namespace nerve::gpu::tuning
{

using json = nlohmann::json;

GpuTuningDatabase &GpuTuningDatabase::instance()
{
    static GpuTuningDatabase instance;
    return instance;
}

void GpuTuningDatabase::initialize(const std::string &cacheDir)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (cacheDir.empty())
    {
        cacheDir_ = getDefaultCachePath();
    }
    else
    {
        cacheDir_ = cacheDir;
    }

    std::filesystem::create_directories(cacheDir_);
    load();
}

std::string GpuTuningDatabase::getDefaultCachePath() const
{
    const char *home = std::getenv("HOME");
    if (home)
    {
        return std::string(home) + "/.cache/nerve/tuning/";
    }
    return "/tmp/nerve_tuning/";
}

std::string GpuTuningDatabase::getCachePath() const
{
    return cacheDir_ + "tuning_cache.json";
}

bool GpuTuningDatabase::load()
{
    return load(getCachePath());
}

bool GpuTuningDatabase::load(const std::string &path)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::ifstream file(path);
    if (!file.is_open())
    {
        return false;
    }

    try
    {
        json j;
        file >> j;

        std::string version = j.value("version", "1.0.0");
        if (version != "2.0.0")
        {
            // Version mismatch - invalidate
            entries_.clear();
            return false;
        }

        entries_.clear();
        auto &entries = j["entries"];
        for (auto &[gpuKey, workloads] : entries.items())
        {
            for (auto &[workloadKey, configJson] : workloads.items())
            {
                TunedConfig config;
                config.blockSize = configJson.value("blockSize", 256);
                config.tileSize = configJson.value("tileSize", 64);
                config.clusterSize = configJson.value("clusterSize", 4);
                config.numStages = configJson.value("numStages", 3);
                config.useWGMMA = configJson.value("useWGMMA", true);
                config.useTMA = configJson.value("useTMA", true);
                config.usePTXOpts = configJson.value("usePTXOpts", true);
                config.useFP8 = configJson.value("useFP8", false);
                config.useFP4 = configJson.value("useFP4", false);
                config.measuredTime = configJson.value("measuredTime", 0.0f);
                config.modelConfidence = configJson.value("modelConfidence", 0.0f);
                config.version = configJson.value("version", "2.0.0");
                config.timestamp = configJson.value("timestamp", 0);
                config.tcgen05Shape = configJson.value("tcgen05Shape", 0);
                config.warpProducerCount = configJson.value("warpProducerCount", 4);
                config.warpConsumerCount = configJson.value("warpConsumerCount", 28);
                config.l2PromotionSize = configJson.value("l2PromotionSize", 128);
                config.smemSwizzle = configJson.value("smemSwizzle", 32);

                entries_[gpuKey][workloadKey] = config;
            }
        }

        dirty_ = false;
        return true;
    }
    catch (const std::exception &e)
    {
        entries_.clear();
        return false;
    }
}

bool GpuTuningDatabase::save() const
{
    return save(getCachePath());
}

bool GpuTuningDatabase::save(const std::string &path) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!dirty_)
    {
        return true;
    }

    json j;
    j["version"] = "2.0.0";
    j["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();

    json entries;
    for (const auto &[gpuKey, workloads] : entries_)
    {
        json gpuEntry;
        for (const auto &[workloadKey, config] : workloads)
        {
            json configJson;
            configJson["blockSize"] = config.blockSize;
            configJson["tileSize"] = config.tileSize;
            configJson["clusterSize"] = config.clusterSize;
            configJson["numStages"] = config.numStages;
            configJson["useWGMMA"] = config.useWGMMA;
            configJson["useTMA"] = config.useTMA;
            configJson["usePTXOpts"] = config.usePTXOpts;
            configJson["useFP8"] = config.useFP8;
            configJson["useFP4"] = config.useFP4;
            configJson["measuredTime"] = config.measuredTime;
            configJson["modelConfidence"] = config.modelConfidence;
            configJson["version"] = config.version;
            configJson["timestamp"] = config.timestamp;
            configJson["tcgen05Shape"] = config.tcgen05Shape;
            configJson["warpProducerCount"] = config.warpProducerCount;
            configJson["warpConsumerCount"] = config.warpConsumerCount;
            configJson["l2PromotionSize"] = config.l2PromotionSize;
            configJson["smemSwizzle"] = config.smemSwizzle;
            gpuEntry[workloadKey] = configJson;
        }
        entries[gpuKey] = gpuEntry;
    }
    j["entries"] = entries;

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream file(path);
    if (!file.is_open())
    {
        return false;
    }

    file << j.dump(2);
    dirty_ = false;
    return true;
}

std::optional<TunedConfig> GpuTuningDatabase::lookup(const GpuSignature &gpu,
                                                     const WorkloadFingerprint &workload) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto gpuIt = entries_.find(gpu.toKey());
    if (gpuIt == entries_.end())
    {
        return std::nullopt;
    }

    auto workloadIt = gpuIt->second.find(workload.toKey());
    if (workloadIt == gpuIt->second.end())
    {
        return std::nullopt;
    }

    return workloadIt->second;
}

void GpuTuningDatabase::store(const GpuSignature &gpu, const WorkloadFingerprint &workload,
                              const TunedConfig &config)
{
    std::lock_guard<std::mutex> lock(mutex_);

    TunedConfig configWithTimestamp = config;
    configWithTimestamp.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    entries_[gpu.toKey()][workload.toKey()] = configWithTimestamp;
    dirty_ = true;
}

void GpuTuningDatabase::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    dirty_ = true;
}

GpuTuningDatabase::Stats GpuTuningDatabase::getStats() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    Stats stats;
    stats.uniqueGpus = entries_.size();

    for (const auto &[gpuKey, workloads] : entries_)
    {
        stats.cacheSizeBytes += gpuKey.size();
        stats.totalEntries += workloads.size();
        for (const auto &[workloadKey, config] : workloads)
        {
            stats.cacheSizeBytes += workloadKey.size() + sizeof(config) + config.version.size();
        }
    }

    std::set<std::string> uniqueWorkloads;
    for (const auto &gpuEntry : entries_)
    {
        for (const auto &workloadEntry : gpuEntry.second)
        {
            uniqueWorkloads.insert(workloadEntry.first);
        }
    }
    stats.uniqueWorkloads = uniqueWorkloads.size();

    return stats;
}

void GpuTuningDatabase::invalidateOldVersions(const std::string &currentVersion)
{
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &[gpuKey, workloads] : entries_)
    {
        for (auto it = workloads.begin(); it != workloads.end();)
        {
            if (it->second.version != currentVersion)
            {
                it = workloads.erase(it);
                dirty_ = true;
            }
            else
            {
                ++it;
            }
        }
    }
}

GpuSignature GpuTuningDatabase::detectCurrentGpu(int deviceId)
{
    GpuSignature sig;

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess)
    {
        return sig;
    }

    sig.computeCapability = prop.major * 10 + prop.minor;
    sig.smCount = prop.multiProcessorCount;
#if CUDART_VERSION >= 13000
    {
        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, deviceId);
        sig.clockRate = clock_khz;
    }
#else
    sig.clockRate = prop.clockRate;
#endif
    sig.totalMemory = prop.totalGlobalMem;
    sig.name = prop.name;

    // Get UUID if available (CUDA 10.2+)
#if CUDART_VERSION >= 10020 && CUDART_VERSION < 12000
    {
        cudaUUID_t uuid;
        err = cudaDeviceGetUuid(&uuid, deviceId);
        if (err == cudaSuccess)
        {
            std::stringstream ss;
            for (int i = 0; i < 16; ++i)
            {
                ss << std::hex << std::setw(2) << std::setfill('0')
                   << static_cast<unsigned int>(uuid.bytes[i]);
            }
            sig.uuid = ss.str();
        }
    }
#endif

    return sig;
}

std::vector<GpuSignature> GpuTuningDatabase::detectAllGpus()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    std::vector<GpuSignature> gpus;
    for (int i = 0; i < deviceCount; ++i)
    {
        gpus.push_back(detectCurrentGpu(i));
    }

    return gpus;
}

// JSON serialization helpers
std::string configToJson(const TunedConfig &config)
{
    json j;
    j["blockSize"] = config.blockSize;
    j["tileSize"] = config.tileSize;
    j["clusterSize"] = config.clusterSize;
    j["numStages"] = config.numStages;
    j["useWGMMA"] = config.useWGMMA;
    j["useTMA"] = config.useTMA;
    j["usePTXOpts"] = config.usePTXOpts;
    j["useFP8"] = config.useFP8;
    j["useFP4"] = config.useFP4;
    j["measuredTime"] = config.measuredTime;
    j["modelConfidence"] = config.modelConfidence;
    j["version"] = config.version;
    j["timestamp"] = config.timestamp;
    return j.dump();
}

TunedConfig configFromJson(const std::string &jsonStr)
{
    TunedConfig config;
    try
    {
        json j = json::parse(jsonStr);
        config.blockSize = j.value("blockSize", 256);
        config.tileSize = j.value("tileSize", 64);
        config.clusterSize = j.value("clusterSize", 4);
        config.numStages = j.value("numStages", 3);
        config.useWGMMA = j.value("useWGMMA", true);
        config.useTMA = j.value("useTMA", true);
        config.usePTXOpts = j.value("usePTXOpts", true);
        config.useFP8 = j.value("useFP8", false);
        config.useFP4 = j.value("useFP4", false);
        config.measuredTime = j.value("measuredTime", 0.0f);
        config.modelConfidence = j.value("modelConfidence", 0.0f);
        config.version = j.value("version", "2.0.0");
        config.timestamp = j.value("timestamp", 0);
    }
    catch (...)
    {
        // Return default config on parse error
    }
    return config;
}

std::string signatureToJson(const GpuSignature &sig)
{
    json j;
    j["computeCapability"] = sig.computeCapability;
    j["smCount"] = sig.smCount;
    j["clockRate"] = sig.clockRate;
    j["totalMemory"] = sig.totalMemory;
    j["uuid"] = sig.uuid;
    j["name"] = sig.name;
    return j.dump();
}

GpuSignature signatureFromJson(const std::string &jsonStr)
{
    GpuSignature sig;
    try
    {
        json j = json::parse(jsonStr);
        sig.computeCapability = j.value("computeCapability", 0);
        sig.smCount = j.value("smCount", 0);
        sig.clockRate = j.value("clockRate", 0);
        sig.totalMemory = j.value("totalMemory", 0);
        sig.uuid = j.value("uuid", "");
        sig.name = j.value("name", "");
    }
    catch (...)
    {
        // Return empty sig on parse error
    }
    return sig;
}

std::string workloadToJson(const WorkloadFingerprint &workload)
{
    json j;
    j["nPoints"] = workload.nPoints;
    j["pointDim"] = workload.pointDim;
    j["sparsityEstimate"] = workload.sparsityEstimate;
    j["problemType"] = workload.problemType;
    j["flags"] = workload.flags;
    return j.dump();
}

WorkloadFingerprint workloadFromJson(const std::string &jsonStr)
{
    WorkloadFingerprint workload;
    try
    {
        json j = json::parse(jsonStr);
        workload.nPoints = j.value("nPoints", 0);
        workload.pointDim = j.value("pointDim", 0);
        workload.sparsityEstimate = j.value("sparsityEstimate", 0.5f);
        workload.problemType = j.value("problemType", 0);
        workload.flags = j.value("flags", 0);
    }
    catch (...)
    {
        // Return default workload on parse error
    }
    return workload;
}

} // namespace nerve::gpu::tuning
