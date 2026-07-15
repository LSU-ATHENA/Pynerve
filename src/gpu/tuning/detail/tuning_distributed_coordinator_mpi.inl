std::vector<NodeGpuConfig> DistributedTuningCoordinator::gatherAllConfigs()
{
    // Gather all configs from all ranks
    std::vector<NodeGpuConfig> result(topology_.numNodes);

    // Serialize local config
    NodeGpuConfig localConfig;
    localConfig.nodeId = topology_.nodeId;
    localConfig.gpus = localGpus_;
    localConfig.configs = localConfigs_;

    // Calculate serialized buffer size
    size_t sigSize = sizeof(GpuSignature);
    size_t configSize = sizeof(TunedConfig);
    size_t bufferSize = sizeof(int) +                      // nodeId
                        sizeof(int) +                      // numGpus
                        localGpus_.size() * sigSize +      // GpuSignature array
                        localConfigs_.size() * configSize; // TunedConfig array

    std::vector<char> sendBuffer(bufferSize);
    size_t offset = 0;

    // Pack local config
    *reinterpret_cast<int *>(&sendBuffer[offset]) = localConfig.nodeId;
    offset += sizeof(int);

    int numGpus = static_cast<int>(localGpus_.size());
    *reinterpret_cast<int *>(&sendBuffer[offset]) = numGpus;
    offset += sizeof(int);

    // Pack GPU signatures
    memcpy(&sendBuffer[offset], localGpus_.data(), localGpus_.size() * sigSize);
    offset += localGpus_.size() * sigSize;

    // Pack configs
    memcpy(&sendBuffer[offset], localConfigs_.data(), localConfigs_.size() * configSize);

    // Gather buffer sizes at rank 0
    int sendCount = static_cast<int>(bufferSize);
    std::vector<int> recvCounts;
    std::vector<int> displs;

    if (topology_.worldRank == 0)
    {
        recvCounts.resize(topology_.worldSize);
        displs.resize(topology_.worldSize);
    }

    MPI_Gather(&sendCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, comm_);

    // Calculate displacements
    if (topology_.worldRank == 0)
    {
        displs[0] = 0;
        for (int i = 1; i < topology_.worldSize; ++i)
        {
            displs[i] = displs[i - 1] + recvCounts[i - 1];
        }
    }

    // Gather all configs at rank 0
    std::vector<char> recvBuffer;
    if (topology_.worldRank == 0)
    {
        int totalSize = displs.back() + recvCounts.back();
        recvBuffer.resize(totalSize);
    }

    MPI_Gatherv(sendBuffer.data(), sendCount, MPI_BYTE, recvBuffer.data(), recvCounts.data(),
                displs.data(), MPI_BYTE, 0, comm_);

    // Unpack at rank 0
    if (topology_.worldRank == 0)
    {
        for (int rank = 0; rank < topology_.worldSize; ++rank)
        {
            int nodeId = rank / topology_.localSize;
            size_t pos = displs[rank];

            // Unpack node ID
            result[nodeId].nodeId = *reinterpret_cast<int *>(&recvBuffer[pos]);
            pos += sizeof(int);

            // Unpack num GPUs
            int recvNumGpus = *reinterpret_cast<int *>(&recvBuffer[pos]);
            pos += sizeof(int);

            // Unpack GPU signatures
            result[nodeId].gpus.resize(recvNumGpus);
            memcpy(result[nodeId].gpus.data(), &recvBuffer[pos], recvNumGpus * sigSize);
            pos += recvNumGpus * sigSize;

            // Unpack configs
            result[nodeId].configs.resize(recvNumGpus);
            memcpy(result[nodeId].configs.data(), &recvBuffer[pos], recvNumGpus * configSize);
        }
    }

    // Broadcast result to all ranks
    // First broadcast the number of nodes
    int numNodes = topology_.numNodes;
    MPI_Bcast(&numNodes, 1, MPI_INT, 0, comm_);

    if (topology_.worldRank != 0)
    {
        result.resize(numNodes);
    }

    // Broadcast each node's config
    for (int i = 0; i < numNodes; ++i)
    {
        if (topology_.worldRank == 0)
        {
            int recvNumGpus = static_cast<int>(result[i].gpus.size());
            MPI_Bcast(&result[i].nodeId, 1, MPI_INT, 0, comm_);
            MPI_Bcast(&recvNumGpus, 1, MPI_INT, 0, comm_);

            if (recvNumGpus > 0)
            {
                result[i].gpus.resize(recvNumGpus);
                result[i].configs.resize(recvNumGpus);
                MPI_Bcast(result[i].gpus.data(), recvNumGpus * sigSize, MPI_BYTE, 0, comm_);
                MPI_Bcast(result[i].configs.data(), recvNumGpus * configSize, MPI_BYTE, 0, comm_);
            }
        }
        else
        {
            int recvNodeId, recvNumGpus;
            MPI_Bcast(&recvNodeId, 1, MPI_INT, 0, comm_);
            MPI_Bcast(&recvNumGpus, 1, MPI_INT, 0, comm_);

            if (recvNumGpus > 0)
            {
                result[i].nodeId = recvNodeId;
                result[i].gpus.resize(recvNumGpus);
                result[i].configs.resize(recvNumGpus);
                MPI_Bcast(result[i].gpus.data(), recvNumGpus * sigSize, MPI_BYTE, 0, comm_);
                MPI_Bcast(result[i].configs.data(), recvNumGpus * configSize, MPI_BYTE, 0, comm_);
            }
        }
    }

    return result;
}

bool DistributedTuningCoordinator::isHomogeneous() const
{
    // Gather all signatures
    std::vector<GpuSignature> allSignatures;

    if (!initialized_)
    {
        const_cast<DistributedTuningCoordinator *>(this)->initialize();
    }

    const_cast<DistributedTuningCoordinator *>(this)->gatherSignatures(allSignatures);

    if (allSignatures.empty())
    {
        return true;
    }

    // Check if all GPUs are the same
    const auto &first = allSignatures[0];
    for (const auto &sig : allSignatures)
    {
        if (sig.computeCapability != first.computeCapability || sig.smCount != first.smCount ||
            sig.clockRate != first.clockRate)
        {
            return false;
        }
    }

    return true;
}

void DistributedTuningCoordinator::broadcastConfig(TunedConfig &config, int root)
{
    // Pack config into buffer for MPI broadcast

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
    } buffer;

    if (topology_.worldRank == root)
    {
        buffer.blockSize = config.blockSize;
        buffer.tileSize = config.tileSize;
        buffer.clusterSize = config.clusterSize;
        buffer.numStages = config.numStages;
        buffer.useWGMMA = config.useWGMMA ? 1 : 0;
        buffer.useTMA = config.useTMA ? 1 : 0;
        buffer.useFP8 = config.useFP8 ? 1 : 0;
        buffer.useFP4 = config.useFP4 ? 1 : 0;
        buffer.measuredTime = config.measuredTime;
        buffer.modelConfidence = config.modelConfidence;
    }

    MPI_Bcast(&buffer, sizeof(buffer), MPI_BYTE, root, comm_);

    if (topology_.worldRank != root)
    {
        config.blockSize = buffer.blockSize;
        config.tileSize = buffer.tileSize;
        config.clusterSize = buffer.clusterSize;
        config.numStages = buffer.numStages;
        config.useWGMMA = buffer.useWGMMA != 0;
        config.useTMA = buffer.useTMA != 0;
        config.useFP8 = buffer.useFP8 != 0;
        config.useFP4 = buffer.useFP4 != 0;
        config.measuredTime = buffer.measuredTime;
        config.modelConfidence = buffer.modelConfidence;
    }
}

void DistributedTuningCoordinator::gatherSignatures(std::vector<GpuSignature> &allSignatures)
{
    // Serialize signature
    std::vector<int> localBuffer;

    for (const auto &sig : localGpus_)
    {
        localBuffer.push_back(sig.computeCapability);
        localBuffer.push_back(sig.smCount);
        localBuffer.push_back(sig.clockRate);
        localBuffer.push_back(static_cast<int>(sig.totalMemory >> 32));
        localBuffer.push_back(static_cast<int>(sig.totalMemory & 0xFFFFFFFF));
    }

    int localCount = static_cast<int>(localGpus_.size());
    std::vector<int> allCounts(topology_.worldSize);

    MPI_Gather(&localCount, 1, MPI_INT, allCounts.data(), 1, MPI_INT, 0, comm_);

    // Calculate recv counts and displacements for Gatherv
    std::vector<int> recvCounts(topology_.worldSize);
    std::vector<int> displs(topology_.worldSize);
    if (topology_.worldRank == 0)
    {
        int total = 0;
        for (int i = 0; i < topology_.worldSize; ++i)
        {
            recvCounts[i] = allCounts[i] * 5; // 5 ints per signature
            displs[i] = total;
            total += recvCounts[i];
        }
        allSignatures.resize(total / 5); // Convert back to signature count
    }

    // Gather all signatures using Gatherv
    int localBufSize = localCount * 5;
    std::vector<int> recvBuffer;
    if (topology_.worldRank == 0)
    {
        int total = 0;
        for (int c : allCounts)
            total += c * 5;
        recvBuffer.resize(total);
    }

    MPI_Gatherv(localBuffer.data(), localBufSize, MPI_INT, recvBuffer.data(),
                recvCounts.data(), // recv_counts per rank
                displs.data(),     // displacements
                MPI_INT, 0, comm_);

    // Reconstruct signatures on rank 0
    if (topology_.worldRank == 0)
    {
        int sigIdx = 0;
        for (int rank = 0; rank < topology_.worldSize; ++rank)
        {
            int count = allCounts[rank];
            int offset = displs[rank];
            for (int i = 0; i < count; ++i)
            {
                GpuSignature sig;
                sig.computeCapability = recvBuffer[offset + i * 5 + 0];
                sig.smCount = recvBuffer[offset + i * 5 + 1];
                sig.clockRate = recvBuffer[offset + i * 5 + 2];
                uint64_t high = static_cast<uint64_t>(recvBuffer[offset + i * 5 + 3]);
                uint64_t low = static_cast<uint64_t>(recvBuffer[offset + i * 5 + 4]);
                sig.totalMemory = (high << 32) | low;
                allSignatures[sigIdx++] = sig;
            }
        }
    }
}

void DistributedTuningCoordinator::scatterConfigs(const std::vector<TunedConfig> &allConfigs)
{
    // Pack configs
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

    std::vector<ConfigBuffer> buffers;

    if (topology_.worldRank == 0)
    {
        buffers.resize(allConfigs.size());
        for (size_t i = 0; i < allConfigs.size(); ++i)
        {
            buffers[i].blockSize = allConfigs[i].blockSize;
            buffers[i].tileSize = allConfigs[i].tileSize;
            buffers[i].clusterSize = allConfigs[i].clusterSize;
            buffers[i].numStages = allConfigs[i].numStages;
            buffers[i].useWGMMA = allConfigs[i].useWGMMA ? 1 : 0;
            buffers[i].useTMA = allConfigs[i].useTMA ? 1 : 0;
            buffers[i].useFP8 = allConfigs[i].useFP8 ? 1 : 0;
            buffers[i].useFP4 = allConfigs[i].useFP4 ? 1 : 0;
            buffers[i].usePTXOpts = allConfigs[i].usePTXOpts ? 1 : 0;
            buffers[i].measuredTime = allConfigs[i].measuredTime;
            buffers[i].modelConfidence = allConfigs[i].modelConfidence;
        }
    }
    else
    {
        buffers.resize(1); // Receive buffer
    }

    // Scatter to all ranks
    MPI_Scatter(buffers.data(), sizeof(ConfigBuffer), MPI_BYTE, buffers.data(),
                sizeof(ConfigBuffer), MPI_BYTE, 0, comm_);

    // Unpack
    if (localConfigs_.empty())
    {
        localConfigs_.resize(1);
    }
    localConfigs_[0].blockSize = buffers[0].blockSize;
    localConfigs_[0].tileSize = buffers[0].tileSize;
    localConfigs_[0].clusterSize = buffers[0].clusterSize;
    localConfigs_[0].numStages = buffers[0].numStages;
    localConfigs_[0].useWGMMA = buffers[0].useWGMMA != 0;
    localConfigs_[0].useTMA = buffers[0].useTMA != 0;
    localConfigs_[0].useFP8 = buffers[0].useFP8 != 0;
    localConfigs_[0].useFP4 = buffers[0].useFP4 != 0;
    localConfigs_[0].usePTXOpts = buffers[0].usePTXOpts != 0;
    localConfigs_[0].measuredTime = buffers[0].measuredTime;
    localConfigs_[0].modelConfidence = buffers[0].modelConfidence;
}

std::vector<GpuSignature> DistributedTuningCoordinator::findUniqueGpuTypes()
{
    std::vector<GpuSignature> allSignatures;
    gatherSignatures(allSignatures);

    std::vector<GpuSignature> unique;

    for (const auto &sig : allSignatures)
    {
        auto it = std::ranges::find_if(unique, [&sig](const GpuSignature &existing) {
            return existing.computeCapability == sig.computeCapability &&
                   existing.smCount == sig.smCount && existing.clockRate == sig.clockRate;
        });

        if (it == unique.end())
        {
            unique.push_back(sig);
        }
    }

    return unique;
}

std::vector<int>
DistributedTuningCoordinator::assignTuningWork(const std::vector<GpuSignature> &uniqueTypes)
{
    std::vector<int> assignments(uniqueTypes.size());

    // Round-robin assignment
    for (size_t i = 0; i < uniqueTypes.size(); ++i)
    {
        assignments[i] = static_cast<int>(i % topology_.worldSize);
    }

    return assignments;
}
