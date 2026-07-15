// NVLink topology detection
namespace
{

void throwOnCudaFailure(cudaError_t status, const char *message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

NvLinkTopology NvLinkTopology::detect()
{
    NvLinkTopology topo;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    topo.numGpus_ = deviceCount;

    topo.p2pMatrix_.resize(deviceCount, std::vector<bool>(deviceCount, false));
    topo.bandwidthMatrix_.resize(deviceCount, std::vector<float>(deviceCount, 0.0f));

    for (int i = 0; i < deviceCount; ++i)
    {
        for (int j = 0; j < deviceCount; ++j)
        {
            if (i == j)
            {
                topo.p2pMatrix_[i][j] = true;
                topo.bandwidthMatrix_[i][j] = 0.0f;
                continue;
            }

            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            topo.p2pMatrix_[i][j] = (canAccess != 0);

            // NVLink bandwidth (approximate)
            // NVLink 4: 900 GB/s
            // NVLink 5: 1800 GB/s (Blackwell)
            if (canAccess)
            {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                int cc = prop.major * 10 + prop.minor;

                if (cc >= 100)
                {
                    topo.bandwidthMatrix_[i][j] = 1800.0f; // NVLink 5
                }
                else if (cc >= 90)
                {
                    topo.bandwidthMatrix_[i][j] = 900.0f; // NVLink 4
                }
                else
                {
                    topo.bandwidthMatrix_[i][j] = 600.0f; // NVLink 3
                }
            }
        }
    }

    return topo;
}

bool NvLinkTopology::hasP2P(int src, int dst) const
{
    if (src < 0 || src >= numGpus_ || dst < 0 || dst >= numGpus_)
    {
        return false;
    }
    return p2pMatrix_[src][dst];
}

float NvLinkTopology::getBandwidthGBps(int src, int dst) const
{
    if (src < 0 || src >= numGpus_ || dst < 0 || dst >= numGpus_)
    {
        return 0.0f;
    }
    return bandwidthMatrix_[src][dst];
}

bool NvLinkTopology::isFullyConnected() const
{
    for (int i = 0; i < numGpus_; ++i)
    {
        for (int j = 0; j < numGpus_; ++j)
        {
            if (i != j && !p2pMatrix_[i][j])
            {
                return false;
            }
        }
    }
    return true;
}

int NvLinkTopology::getOptimalPeer(int src, const std::vector<bool> &candidates) const
{
    if (src < 0 || src >= numGpus_ || candidates.size() < static_cast<size_t>(numGpus_))
    {
        return -1;
    }
    float maxBandwidth = 0.0f;
    int bestPeer = -1;

    for (int i = 0; i < numGpus_; ++i)
    {
        if (candidates[i] && p2pMatrix_[src][i])
        {
            float bw = bandwidthMatrix_[src][i];
            if (bw > maxBandwidth)
            {
                maxBandwidth = bw;
                bestPeer = i;
            }
        }
    }

    return bestPeer;
}

// Multi-GPU work distributor

MultiGpuWorkDistributor::WorkDistribution
MultiGpuWorkDistributor::distributeByTopology(uint32_t totalWork, const NvLinkTopology &topology,
                                              const std::vector<TunedConfig> &configs)
{
    WorkDistribution dist;
    int numGpus = topology.numGpus();

    dist.rowRanges.resize(numGpus);
    dist.rowCounts.resize(numGpus);
    dist.expectedImbalance = 0.0f;
    if (numGpus <= 0 || configs.size() < static_cast<size_t>(numGpus))
    {
        return dist;
    }

    // Calculate relative performance of each GPU
    std::vector<float> performance(numGpus);
    for (int i = 0; i < numGpus; ++i)
    {
        // Use measured benchmark time as primary metric, proxy to tile*block proxy
        if (configs[i].measuredTime > 0)
        {
            performance[i] = 1.0f / configs[i].measuredTime; // Higher is better
        }
        else
        {
            performance[i] = configs[i].tileSize * configs[i].blockSize;
        }
    }

    float totalPerformance = 0.0f;
    for (float p : performance)
        totalPerformance += p;
    if (!(totalPerformance > 0.0f) || !std::isfinite(totalPerformance))
    {
        return dist;
    }

    // Distribute work proportionally
    uint32_t currentStart = 0;
    float expectedImbalance = 0.0f;

    for (int i = 0; i < numGpus; ++i)
    {
        float fraction = performance[i] / totalPerformance;
        uint32_t count = static_cast<uint32_t>(totalWork * fraction);

        // Last GPU gets remainder
        if (i == numGpus - 1)
        {
            count = totalWork - currentStart;
        }

        dist.rowRanges[i] = currentStart;
        dist.rowCounts[i] = count;
        currentStart += count;

        // Calculate imbalance
        float idealFraction = 1.0f / numGpus;
        float imbalance = std::abs(fraction - idealFraction) / idealFraction;
        expectedImbalance = std::max(expectedImbalance, imbalance);
    }

    dist.expectedImbalance = expectedImbalance;
    return dist;
}

MultiGpuWorkDistributor::WorkDistribution MultiGpuWorkDistributor::distributeBandwidthAware(
    uint32_t totalWork, const NvLinkTopology &topology, const std::vector<GpuSignature> &gpus)
{
    WorkDistribution dist;
    int numGpus = topology.numGpus();

    dist.rowRanges.resize(numGpus);
    dist.rowCounts.resize(numGpus);
    dist.expectedImbalance = 0.0f;
    if (numGpus <= 0 || gpus.size() < static_cast<size_t>(numGpus))
    {
        return dist;
    }

    // Use memory bandwidth as proxy for performance
    std::vector<float> bandwidths(numGpus);
    for (int i = 0; i < numGpus; ++i)
    {
        if (gpus[i].computeCapability >= 100)
        {
            bandwidths[i] = 8000.0f; // B200
        }
        else if (gpus[i].computeCapability >= 90)
        {
            bandwidths[i] = 3350.0f; // H100
        }
        else
        {
            bandwidths[i] = 2000.0f; // A100
        }
    }

    float totalBandwidth = 0.0f;
    for (float bw : bandwidths)
        totalBandwidth += bw;
    if (!(totalBandwidth > 0.0f) || !std::isfinite(totalBandwidth))
    {
        return dist;
    }

    uint32_t currentStart = 0;
    for (int i = 0; i < numGpus; ++i)
    {
        float fraction = bandwidths[i] / totalBandwidth;
        uint32_t count = static_cast<uint32_t>(totalWork * fraction);

        if (i == numGpus - 1)
        {
            count = totalWork - currentStart;
        }

        dist.rowRanges[i] = currentStart;
        dist.rowCounts[i] = count;
        currentStart += count;
    }

    dist.expectedImbalance = 0.0f;
    return dist;
}

// Multi-GPU synchronizer

MultiGpuSynchronizer::MultiGpuSynchronizer(const std::vector<int> &deviceIds)
    : deviceIds_(deviceIds)
{
    events_.resize(deviceIds_.size());
    for (size_t i = 0; i < deviceIds_.size(); ++i)
    {
        const auto cleanupCreatedEvents = [&]() {
            for (size_t j = 0; j < i; ++j)
            {
                if (events_[j] != nullptr)
                {
                    (void)cudaSetDevice(deviceIds_[j]);
                    (void)cudaEventDestroy(events_[j]);
                    events_[j] = nullptr;
                }
            }
        };
        if (deviceIds_[i] < 0)
        {
            cleanupCreatedEvents();
            throw std::invalid_argument("multi-GPU device id must be non-negative");
        }
        cudaError_t status = cudaSetDevice(deviceIds_[i]);
        if (status != cudaSuccess)
        {
            cleanupCreatedEvents();
            throw std::runtime_error("cudaSetDevice failed");
        }
        status = cudaEventCreate(&events_[i]);
        if (status != cudaSuccess)
        {
            cleanupCreatedEvents();
            throw std::runtime_error("cudaEventCreate failed");
        }
    }
}

MultiGpuSynchronizer::~MultiGpuSynchronizer()
{
    for (size_t i = 0; i < events_.size() && i < deviceIds_.size(); ++i)
    {
        if (events_[i] != nullptr)
        {
            (void)cudaSetDevice(deviceIds_[i]);
            (void)cudaEventDestroy(events_[i]);
        }
    }
}

void MultiGpuSynchronizer::barrier()
{
    // Record events on all devices
    for (size_t i = 0; i < deviceIds_.size(); ++i)
    {
        throwOnCudaFailure(cudaSetDevice(deviceIds_[i]), "cudaSetDevice failed");
        throwOnCudaFailure(cudaEventRecord(events_[i]), "cudaEventRecord failed");
    }

    // Wait for all events
    for (size_t i = 0; i < deviceIds_.size(); ++i)
    {
        throwOnCudaFailure(cudaEventSynchronize(events_[i]), "cudaEventSynchronize failed");
    }
}

void MultiGpuSynchronizer::allGather(void *localData, size_t localSize,
                                     std::vector<void *> &allData, const NvLinkTopology &topology)
{
    if (localSize == 0)
    {
        return;
    }
    if (localData == nullptr)
    {
        throw std::invalid_argument("multi-GPU allGather localData cannot be null");
    }
    if (allData.size() < deviceIds_.size())
    {
        throw std::invalid_argument("multi-GPU allGather output buffer count is too small");
    }
    for (size_t i = 0; i < deviceIds_.size(); ++i)
    {
        if (allData[i] == nullptr)
        {
            throw std::invalid_argument("multi-GPU allGather output buffers cannot be null");
        }
    }

    // Simple implementation: each GPU copies to all others
    for (size_t src = 0; src < deviceIds_.size(); ++src)
    {
        for (size_t dst = 0; dst < deviceIds_.size(); ++dst)
        {
            if (src == dst)
                continue;

            const int srcDevice = deviceIds_[src];
            const int dstDevice = deviceIds_[dst];
            if (topology.hasP2P(srcDevice, dstDevice))
            {
                throwOnCudaFailure(cudaSetDevice(dstDevice), "cudaSetDevice failed");
                throwOnCudaFailure(
                    cudaMemcpyPeer(allData[dst], dstDevice, localData, srcDevice, localSize),
                    "cudaMemcpyPeer failed");
            }
        }
    }
}

void MultiGpuSynchronizer::broadcast(void *data, size_t size, int rootDevice,
                                     const NvLinkTopology &topology)
{
    if (size == 0)
    {
        return;
    }
    if (data == nullptr)
    {
        throw std::invalid_argument("multi-GPU broadcast data cannot be null");
    }
    if (rootDevice < 0 || rootDevice >= static_cast<int>(deviceIds_.size()))
    {
        throw std::invalid_argument("multi-GPU broadcast root device is out of range");
    }
    const int rootCudaDevice = deviceIds_[static_cast<size_t>(rootDevice)];

    for (size_t i = 0; i < deviceIds_.size(); ++i)
    {
        if (static_cast<int>(i) == rootDevice)
            continue;

        const int dstDevice = deviceIds_[i];
        if (topology.hasP2P(rootCudaDevice, dstDevice))
        {
            throwOnCudaFailure(cudaSetDevice(dstDevice), "cudaSetDevice failed");
            throwOnCudaFailure(cudaMemcpyPeer(data, dstDevice, data, rootCudaDevice, size),
                               "cudaMemcpyPeer failed");
        }
    }
}
