
#pragma once

#include "nerve/types.hpp"

#include <compare>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace nerve::gpu::tuning
{

struct GpuSignature
{
    int computeCapability = 0;
    int smCount = 0;
    int clockRate = 0;
    size_t totalMemory = 0;
    std::string uuid;
    std::string name;

    std::string toKey() const
    {
        return std::to_string(computeCapability) + "_" + std::to_string(smCount) + "_" +
               std::to_string(clockRate) + "_" + std::to_string(totalMemory) + "_" + uuid;
    }

    [[nodiscard]] auto operator<=>(const GpuSignature &other) const = default;
    [[nodiscard]] bool operator==(const GpuSignature &other) const = default;
};

struct WorkloadFingerprint
{
    uint32_t nPoints = 0;
    uint32_t pointDim = 0;
    float sparsityEstimate = 0.5f;
    uint32_t problemType = 0;
    uint32_t flags = 0;

    [[nodiscard]] auto operator<=>(const WorkloadFingerprint &other) const = default;
    [[nodiscard]] bool operator==(const WorkloadFingerprint &other) const = default;

    std::string toKey() const
    {
        return std::to_string(nPoints) + "_" + std::to_string(pointDim) + "_" +
               std::to_string(static_cast<int>(sparsityEstimate * 100)) + "_" +
               std::to_string(problemType) + "_" + std::to_string(flags);
    }
};

struct TunedConfig
{
    int blockSize = 256;
    int tileSize = 64;
    int clusterSize = 4;
    int numStages = 3;
    bool useWGMMA = true;
    bool useTMA = true;
    bool usePTXOpts = true;
    bool useFP8 = false;
    bool useFP4 = false;
    float measuredTime = 0.0f;
    float modelConfidence = 0.0f;
    std::string version = "2.0.0";
    int64_t timestamp = 0;

    // Extended Blackwell-specific fields
    int tcgen05Shape = 0;
    int warpProducerCount = 4;
    int warpConsumerCount = 28;
    int l2PromotionSize = 128;
    int smemSwizzle = 32;

    // Cluster configuration
    bool useCluster = false;
    bool useNonPortableCluster = false;
    bool useTMAMulticast = false;

    [[nodiscard]] auto operator<=>(const TunedConfig &other) const = default;
    [[nodiscard]] bool operator==(const TunedConfig &other) const = default;
};

class GpuTuningDatabase
{
public:
    static GpuTuningDatabase &instance();

    /// Initialize with cache directory (default: ~/.cache/nerve/tuning/)
    void initialize(const std::string &cacheDir = "");

    /// Load database from disk
    bool load();
    bool load(const std::string &path);

    /// Save database to disk
    bool save() const;
    bool save(const std::string &path) const;

    /// Lookup cached configuration
    std::optional<TunedConfig> lookup(const GpuSignature &gpu,
                                      const WorkloadFingerprint &workload) const;

    /// Store configuration in cache
    void store(const GpuSignature &gpu, const WorkloadFingerprint &workload,
               const TunedConfig &config);

    /// Clear all entries
    void clear();

    /// Get cache statistics
    struct Stats
    {
        size_t totalEntries = 0;
        size_t uniqueGpus = 0;
        size_t uniqueWorkloads = 0;
        size_t cacheSizeBytes = 0;

        [[nodiscard]] auto operator<=>(const Stats &other) const = default;
        [[nodiscard]] bool operator==(const Stats &other) const = default;
    };
    Stats getStats() const;

    /// Invalidate entries with version mismatch
    void invalidateOldVersions(const std::string &currentVersion);

    /// Detect GPU signature for current device
    static GpuSignature detectCurrentGpu(int deviceId = 0);
    static std::vector<GpuSignature> detectAllGpus();

private:
    GpuTuningDatabase() = default;
    ~GpuTuningDatabase() = default;
    GpuTuningDatabase(const GpuTuningDatabase &) = delete;
    GpuTuningDatabase &operator=(const GpuTuningDatabase &) = delete;

    std::string getDefaultCachePath() const;
    std::string getCachePath() const;

    mutable std::mutex mutex_;
    std::string cacheDir_;
    std::map<std::string, std::map<std::string, TunedConfig>> entries_;
    mutable bool dirty_ = false;
};

std::string configToJson(const TunedConfig &config);
TunedConfig configFromJson(const std::string &json);
std::string signatureToJson(const GpuSignature &sig);
GpuSignature signatureFromJson(const std::string &json);
std::string workloadToJson(const WorkloadFingerprint &workload);
WorkloadFingerprint workloadFromJson(const std::string &json);

} // namespace nerve::gpu::tuning
