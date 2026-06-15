
#pragma once

#include "nerve/config.hpp"
#include "nerve/gpu/hybrid_tuning.hpp"
#include "nerve/gpu/tuning_cache.hpp"

#include <vector>
#if defined(NERVE_HAS_MPI)
#include <mpi.h>
#endif

namespace nerve::gpu::tuning
{

#if defined(NERVE_HAS_MPI)

struct GpuTopology
{
    int worldSize;
    int worldRank;
    int localRank;
    int localSize;
    int numNodes;
    int nodeId;

    bool isSameNode(int peerRank) const { return (peerRank / localSize) == nodeId; }
};

struct NodeGpuConfig
{
    int nodeId;
    std::vector<GpuSignature> gpus;
    std::vector<TunedConfig> configs;
};

class DistributedTuningCoordinator
{
public:
    explicit DistributedTuningCoordinator(MPI_Comm comm = MPI_COMM_WORLD);
    ~DistributedTuningCoordinator();

    /// Initialize topology detection
    void initialize();

    /// Single-node tuning then broadcast to all ranks
    std::vector<TunedConfig> tuneAndBroadcast(const WorkloadFingerprint &workload,
                                              bool usePersistence = true);

    /// Heterogeneous tuning: each unique GPU type is tuned once
    std::vector<TunedConfig> tuneHeterogeneous(const WorkloadFingerprint &workload,
                                               bool usePersistence = true);

    /// Collective tuning with workload-aware distribution
    std::vector<TunedConfig> tuneCollective(const WorkloadFingerprint &workload,
                                            const std::vector<uint32_t> &workloadDistribution,
                                            bool usePersistence = true);

    /// Gather configs from all ranks for comparison
    std::vector<NodeGpuConfig> gatherAllConfigs();

    /// Check if all GPUs are identical (homogeneous system)
    bool isHomogeneous() const;

    /// Get topology info
    const GpuTopology &topology() const { return topology_; }

    /// Get local GPU signatures
    const std::vector<GpuSignature> &localGpus() const { return localGpus_; }

private:
    MPI_Comm comm_;
    GpuTopology topology_;
    std::vector<GpuSignature> localGpus_;
    std::vector<TunedConfig> localConfigs_;
    HybridTuner tuner_;
    bool initialized_ = false;

    void detectTopology();
    void detectLocalGpus();

    // MPI helper functions
    void broadcastConfig(TunedConfig &config, int root);
    void gatherSignatures(std::vector<GpuSignature> &allSignatures);
    void scatterConfigs(const std::vector<TunedConfig> &allConfigs);

    // Find unique GPU types across all nodes
    std::vector<GpuSignature> findUniqueGpuTypes();

    // Assign tuning work to ranks
    std::vector<int> assignTuningWork(const std::vector<GpuSignature> &uniqueTypes);
};

class NvLinkTopology
{
public:
    static NvLinkTopology detect();

    int numGpus() const { return numGpus_; }
    bool hasP2P(int src, int dst) const;
    float getBandwidthGBps(int src, int dst) const;
    bool isFullyConnected() const;

    /// Get optimal peer for data transfer
    int getOptimalPeer(int src, const std::vector<bool> &candidates) const;

private:
    int numGpus_ = 0;
    std::vector<std::vector<bool>> p2pMatrix_;
    std::vector<std::vector<float>> bandwidthMatrix_;
};

class MultiGpuWorkDistributor
{
public:
    struct WorkDistribution
    {
        std::vector<uint32_t> rowRanges; // Start row for each GPU
        std::vector<uint32_t> rowCounts; // Number of rows for each GPU
        float expectedImbalance;         // Load imbalance metric
    };

    /// Distribute work based on NVLink topology and GPU performance
    static WorkDistribution distributeByTopology(uint32_t totalWork, const NvLinkTopology &topology,
                                                 const std::vector<TunedConfig> &configs);

    /// Distribute with bandwidth-aware chunking
    static WorkDistribution distributeBandwidthAware(uint32_t totalWork,
                                                     const NvLinkTopology &topology,
                                                     const std::vector<GpuSignature> &gpus);
};

class MultiGpuSynchronizer
{
public:
    explicit MultiGpuSynchronizer(const std::vector<int> &deviceIds);
    ~MultiGpuSynchronizer();
    MultiGpuSynchronizer(const MultiGpuSynchronizer &) = delete;
    MultiGpuSynchronizer &operator=(const MultiGpuSynchronizer &) = delete;
    MultiGpuSynchronizer(MultiGpuSynchronizer &&) = delete;
    MultiGpuSynchronizer &operator=(MultiGpuSynchronizer &&) = delete;

    /// Cross-GPU barrier
    void barrier();

    /// All-gather operation using NVLink P2P
    void allGather(void *localData, size_t localSize, std::vector<void *> &allData,
                   const NvLinkTopology &topology);

    /// Broadcast from root GPU
    void broadcast(void *data, size_t size, int rootDevice, const NvLinkTopology &topology);

private:
    std::vector<int> deviceIds_;
    std::vector<cudaEvent_t> events_;
};

#endif // NERVE_HAS_MPI

} // namespace nerve::gpu::tuning
