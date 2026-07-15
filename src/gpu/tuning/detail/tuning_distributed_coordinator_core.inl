DistributedTuningCoordinator::DistributedTuningCoordinator(MPI_Comm comm)
    : comm_(comm)
    , initialized_(false)
{}

DistributedTuningCoordinator::~DistributedTuningCoordinator() = default;

void DistributedTuningCoordinator::initialize()
{
    detectTopology();
    detectLocalGpus();
    initialized_ = true;
}

void DistributedTuningCoordinator::detectTopology()
{
    MPI_Comm_size(comm_, &topology_.worldSize);
    MPI_Comm_rank(comm_, &topology_.worldRank);

    // Get local rank (rank within node)
    MPI_Comm localComm;
    MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &topology_.localRank);
    MPI_Comm_size(localComm, &topology_.localSize);
    MPI_Comm_free(&localComm);

    // Calculate node ID
    topology_.numNodes = topology_.worldSize / topology_.localSize;
    topology_.nodeId = topology_.worldRank / topology_.localSize;
}

void DistributedTuningCoordinator::detectLocalGpus()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0)
    {
        throw std::runtime_error("No CUDA devices detected for distributed tuning");
    }

    // In typical MPI setups, localRank maps to GPU ID
    int gpuId = topology_.localRank % deviceCount;
    cudaSetDevice(gpuId);

    // Detect all GPUs on this node
    for (int i = 0; i < deviceCount; ++i)
    {
        localGpus_.push_back(GpuTuningDatabase::detectCurrentGpu(i));
    }

    // Set configs vector size
    localConfigs_.resize(localGpus_.size());
}

std::vector<TunedConfig>
DistributedTuningCoordinator::tuneAndBroadcast(const WorkloadFingerprint &workload,
                                               bool usePersistence)
{
    if (!initialized_)
    {
        initialize();
    }

    std::vector<TunedConfig> allConfigs;

    if (topology_.worldRank == 0)
    {
        // Rank 0 tunes for its GPUs
        GpuHardwareSpecs specs = detectHardwareSpecs(0);

        for (size_t i = 0; i < localGpus_.size(); ++i)
        {
            if (usePersistence)
            {
                auto cached = GpuTuningDatabase::instance().lookup(localGpus_[i], workload);
                if (cached)
                {
                    localConfigs_[i] = *cached;
                }
                else
                {
                    localConfigs_[i] =
                        tuner_.tune(specs, analyzeWorkload(workload.nPoints, workload.pointDim,
                                                           workload.sparsityEstimate));
                    GpuTuningDatabase::instance().store(localGpus_[i], workload, localConfigs_[i]);
                }
            }
            else
            {
                localConfigs_[i] =
                    tuner_.tune(specs, analyzeWorkload(workload.nPoints, workload.pointDim,
                                                       workload.sparsityEstimate));
            }
        }

        // Gather all signatures to determine what needs to be tuned
        std::vector<GpuSignature> allSignatures;
        gatherSignatures(allSignatures);

        // Find unique GPU types
        std::vector<GpuSignature> uniqueTypes;
        std::vector<int> tuningAssignments;

        for (const auto &sig : allSignatures)
        {
            auto it = std::ranges::find_if(uniqueTypes, [&sig](const GpuSignature &existing) {
                return existing.computeCapability == sig.computeCapability &&
                       existing.smCount == sig.smCount && existing.clockRate == sig.clockRate;
            });

            if (it == uniqueTypes.end())
            {
                uniqueTypes.push_back(sig);
                tuningAssignments.push_back(uniqueTypes.size() - 1);
            }
            else
            {
                tuningAssignments.push_back(std::distance(uniqueTypes.begin(), it));
            }
        }

        // Tune for each unique type (if not already tuned)
        std::vector<TunedConfig> proxyConfigs;
        for (size_t i = 0; i < uniqueTypes.size(); ++i)
        {
            bool alreadyTuned = false;
            for (const auto &localSig : localGpus_)
            {
                if (localSig.computeCapability == uniqueTypes[i].computeCapability &&
                    localSig.smCount == uniqueTypes[i].smCount)
                {
                    alreadyTuned = true;
                    break;
                }
            }

            if (!alreadyTuned)
            {
                // Need to tune for this type - use rank 0's hardware as proxy
                // Remote tuning: use existing config from similar GPU type
                // Scale parameters based on SM count ratio

                TunedConfig baseConfig;
                if (!localConfigs_.empty())
                {
                    baseConfig = localConfigs_[0];
                }

                // Adjust config for target GPU's SM count
                const int baseSmCount =
                    localGpus_.empty() ? 1 : std::max(1, localGpus_.front().smCount);
                float smRatio =
                    static_cast<float>(uniqueTypes[i].smCount) / static_cast<float>(baseSmCount);

                TunedConfig adjustedConfig = baseConfig;
                adjustedConfig.blockSize = std::min(
                    512,
                    std::max(128, static_cast<int>(baseConfig.blockSize * std::sqrt(smRatio))));
                adjustedConfig.tileSize =
                    std::min(256, std::max(32, static_cast<int>(baseConfig.tileSize * smRatio)));
                adjustedConfig.clusterSize =
                    std::min(8, std::max(1, static_cast<int>(baseConfig.clusterSize * smRatio)));
                adjustedConfig.modelConfidence = 0.5f; // Lower confidence for proxy tuning

                proxyConfigs.push_back(adjustedConfig);
                alreadyTuned = true;
            }
        }

        // Build all configs
        allConfigs.resize(topology_.worldSize);

        // Copy rank 0's configs
        for (size_t i = 0; i < localConfigs_.size() && i < allConfigs.size(); ++i)
        {
            allConfigs[i] = localConfigs_[i];
        }

        // Fill remaining ranks with proxy-tuned configs for unmatched GPU types.
        size_t configCursor = std::min(localConfigs_.size(), allConfigs.size());
        for (const auto &proxy : proxyConfigs)
        {
            if (configCursor >= allConfigs.size())
            {
                break;
            }
            allConfigs[configCursor++] = proxy;
        }

        // Scatter configs to all ranks
        scatterConfigs(allConfigs);
    }
    else
    {
        // Other ranks receive configs from rank 0
        scatterConfigs(allConfigs);
    }

    if (localConfigs_.empty())
    {
        localConfigs_.resize(localGpus_.size(), TunedConfig{});
    }
    if (!localConfigs_.empty())
    {
        const TunedConfig rankConfig = localConfigs_.front();
        localConfigs_.assign(localGpus_.size(), rankConfig);
    }

    struct ConfigBuffer
    {
        int blockSize;
        int tileSize;
        int clusterSize;
        int numStages;
        int useWGMMA;
        int useTMA;
        int useFP8;
        int useFP4;
        int usePTXOpts;
        float measuredTime;
        float modelConfidence;
    };

    ConfigBuffer localConfig{};
    if (!localConfigs_.empty())
    {
        const TunedConfig &src = localConfigs_.front();
        localConfig.blockSize = src.blockSize;
        localConfig.tileSize = src.tileSize;
        localConfig.clusterSize = src.clusterSize;
        localConfig.numStages = src.numStages;
        localConfig.useWGMMA = src.useWGMMA ? 1 : 0;
        localConfig.useTMA = src.useTMA ? 1 : 0;
        localConfig.useFP8 = src.useFP8 ? 1 : 0;
        localConfig.useFP4 = src.useFP4 ? 1 : 0;
        localConfig.usePTXOpts = src.usePTXOpts ? 1 : 0;
        localConfig.measuredTime = src.measuredTime;
        localConfig.modelConfidence = src.modelConfidence;
    }

    std::vector<ConfigBuffer> gathered(topology_.worldSize);
    MPI_Allgather(&localConfig, sizeof(ConfigBuffer), MPI_BYTE, gathered.data(),
                  sizeof(ConfigBuffer), MPI_BYTE, comm_);

    allConfigs.resize(topology_.worldSize);
    for (int rank = 0; rank < topology_.worldSize; ++rank)
    {
        TunedConfig cfg{};
        cfg.blockSize = gathered[rank].blockSize;
        cfg.tileSize = gathered[rank].tileSize;
        cfg.clusterSize = gathered[rank].clusterSize;
        cfg.numStages = gathered[rank].numStages;
        cfg.useWGMMA = gathered[rank].useWGMMA != 0;
        cfg.useTMA = gathered[rank].useTMA != 0;
        cfg.useFP8 = gathered[rank].useFP8 != 0;
        cfg.useFP4 = gathered[rank].useFP4 != 0;
        cfg.usePTXOpts = gathered[rank].usePTXOpts != 0;
        cfg.measuredTime = gathered[rank].measuredTime;
        cfg.modelConfidence = gathered[rank].modelConfidence;
        allConfigs[rank] = cfg;
    }

    return allConfigs;
}

std::vector<TunedConfig>
DistributedTuningCoordinator::tuneHeterogeneous(const WorkloadFingerprint &workload,
                                                bool usePersistence)
{
    if (!initialized_)
    {
        initialize();
    }

    // Gather all GPU signatures
    std::vector<GpuSignature> allSignatures;
    gatherSignatures(allSignatures);

    // Find unique GPU types
    std::vector<GpuSignature> uniqueTypes = findUniqueGpuTypes();

    // Assign tuning work
    std::vector<int> tuningAssignments = assignTuningWork(uniqueTypes);

    // Each rank tunes for its assigned types
    std::map<int, TunedConfig> tunedConfigs;

    for (size_t i = 0; i < tuningAssignments.size(); ++i)
    {
        if (tuningAssignments[i] == topology_.worldRank)
        {
            // This rank tunes for unique type i
            GpuSignature &sig = uniqueTypes[i];

            TunedConfig config;
            if (usePersistence)
            {
                auto cached = GpuTuningDatabase::instance().lookup(sig, workload);
                if (cached)
                {
                    config = *cached;
                }
                else
                {
                    GpuHardwareSpecs specs;
                    specs.computeCapability = sig.computeCapability;
                    specs.smCount = sig.smCount;
                    specs.clockRate = sig.clockRate;
                    // Fill in other specs based on architecture

                    config = tuner_.tune(specs, analyzeWorkload(workload.nPoints, workload.pointDim,
                                                                workload.sparsityEstimate));
                    GpuTuningDatabase::instance().store(sig, workload, config);
                }
            }
            else
            {
                GpuHardwareSpecs specs;
                specs.computeCapability = sig.computeCapability;
                specs.smCount = sig.smCount;
                specs.clockRate = sig.clockRate;

                config = tuner_.tune(specs, analyzeWorkload(workload.nPoints, workload.pointDim,
                                                            workload.sparsityEstimate));
            }

            tunedConfigs[i] = config;
        }
    }

    // Gather all tuned configs
    // Serialize configs for MPI transfer using broadcastConfig

    std::vector<TunedConfig> allConfigs(uniqueTypes.size());

    // Broadcast tuned configs
    for (size_t i = 0; i < uniqueTypes.size(); ++i)
    {
        int tunerRank = tuningAssignments[i];
        TunedConfig config;

        if (topology_.worldRank == tunerRank)
        {
            config = tunedConfigs[i];
        }

        broadcastConfig(config, tunerRank);
        allConfigs[i] = config;
    }

    // Map unique configs back to each GPU
    std::vector<TunedConfig> result;
    for (const auto &sig : localGpus_)
    {
        auto it = std::ranges::find_if(uniqueTypes, [&sig](const GpuSignature &type) {
            return type.computeCapability == sig.computeCapability && type.smCount == sig.smCount &&
                   type.clockRate == sig.clockRate;
        });

        if (it != uniqueTypes.end())
        {
            size_t idx = std::distance(uniqueTypes.begin(), it);
            result.push_back(allConfigs[idx]);
        }
        else
        {
            // Default - should not happen
            result.push_back(TunedConfig{});
        }
    }

    localConfigs_ = result;
    return result;
}

std::vector<TunedConfig>
DistributedTuningCoordinator::tuneCollective(const WorkloadFingerprint &workload,
                                             const std::vector<uint32_t> &workloadDistribution,
                                             bool usePersistence)
{
    if (!initialized_)
    {
        initialize();
    }

    const uint64_t localWork =
        std::accumulate(workloadDistribution.begin(), workloadDistribution.end(), uint64_t{0});

    std::vector<uint64_t> totalWorkPerRank(topology_.worldSize, 0);
    MPI_Allgather(const_cast<uint64_t *>(&localWork), 1, MPI_UINT64_T, totalWorkPerRank.data(), 1,
                  MPI_UINT64_T, comm_);

    const uint64_t globalWork =
        std::accumulate(totalWorkPerRank.begin(), totalWorkPerRank.end(), uint64_t{0});
    const float averageWork = topology_.worldSize > 0 ? static_cast<float>(globalWork) /
                                                            static_cast<float>(topology_.worldSize)
                                                      : 0.0f;
    const float loadScale =
        averageWork > 0.0f ? std::clamp(static_cast<float>(localWork) / averageWork, 0.5f, 2.0f)
                           : 1.0f;

    auto baseConfigs = tuneHeterogeneous(workload, usePersistence);
    if (baseConfigs.empty())
    {
        localConfigs_.assign(localGpus_.size(), TunedConfig{});
    }
    else
    {
        localConfigs_.resize(localGpus_.size());
        for (size_t i = 0; i < localGpus_.size(); ++i)
        {
            const TunedConfig &base = baseConfigs[std::min(i, baseConfigs.size() - 1)];
            TunedConfig tuned = base;
            tuned.blockSize = std::clamp(
                static_cast<int>(std::lround(base.blockSize * std::sqrt(loadScale))), 64, 512);
            tuned.tileSize =
                std::clamp(static_cast<int>(std::lround(base.tileSize * loadScale)), 16, 256);
            tuned.clusterSize =
                std::clamp(static_cast<int>(std::lround(base.clusterSize * loadScale)), 1, 8);
            tuned.modelConfidence = std::clamp(
                base.modelConfidence * (0.8f + 0.2f / std::max(loadScale, 0.5f)), 0.0f, 1.0f);
            localConfigs_[i] = tuned;
        }
    }

    struct ConfigBuffer
    {
        int blockSize;
        int tileSize;
        int clusterSize;
        int numStages;
        int useWGMMA;
        int useTMA;
        int useFP8;
        int useFP4;
        float measuredTime;
        float modelConfidence;
    };

    ConfigBuffer localConfig{};
    if (!localConfigs_.empty())
    {
        const TunedConfig &src = localConfigs_.front();
        localConfig.blockSize = src.blockSize;
        localConfig.tileSize = src.tileSize;
        localConfig.clusterSize = src.clusterSize;
        localConfig.numStages = src.numStages;
        localConfig.useWGMMA = src.useWGMMA ? 1 : 0;
        localConfig.useTMA = src.useTMA ? 1 : 0;
        localConfig.useFP8 = src.useFP8 ? 1 : 0;
        localConfig.useFP4 = src.useFP4 ? 1 : 0;
        localConfig.measuredTime = src.measuredTime;
        localConfig.modelConfidence = src.modelConfidence;
    }

    std::vector<ConfigBuffer> rankConfigs(topology_.worldSize);
    MPI_Allgather(&localConfig, sizeof(ConfigBuffer), MPI_BYTE, rankConfigs.data(),
                  sizeof(ConfigBuffer), MPI_BYTE, comm_);

    std::vector<TunedConfig> result(topology_.worldSize);
    for (int rank = 0; rank < topology_.worldSize; ++rank)
    {
        TunedConfig cfg{};
        cfg.blockSize = rankConfigs[rank].blockSize;
        cfg.tileSize = rankConfigs[rank].tileSize;
        cfg.clusterSize = rankConfigs[rank].clusterSize;
        cfg.numStages = rankConfigs[rank].numStages;
        cfg.useWGMMA = rankConfigs[rank].useWGMMA != 0;
        cfg.useTMA = rankConfigs[rank].useTMA != 0;
        cfg.useFP8 = rankConfigs[rank].useFP8 != 0;
        cfg.useFP4 = rankConfigs[rank].useFP4 != 0;
        cfg.measuredTime = rankConfigs[rank].measuredTime;
        cfg.modelConfidence = rankConfigs[rank].modelConfidence;
        result[rank] = cfg;
    }

    return result;
}
