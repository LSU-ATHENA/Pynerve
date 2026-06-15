
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_engine.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace nerve::persistence::adaptive_acceleration::streaming
{

struct StreamingConfig
{
    std::size_t chunk_size = 0;
    std::size_t memory_limit_mb = 8192;
    bool enable_compression = true;
    bool overlap_computation = true;
    bool enable_multi_threading = true;
    std::size_t num_processing_threads = 0;
    bool enable_gpu_streaming = true;
    bool enable_progressive_output = true;
    double target_efficiency = 0.8;
};

struct StreamInfo
{
    std::size_t totalPoints = 0;
    std::size_t point_dim = 0;
    std::size_t chunk_size = 0;
    std::size_t num_chunks = 0;
    double max_radius = 1.0;
    bool is_compressed = false;
    std::string data_format;
    std::string compression_type;
};

struct DataChunk
{
    std::vector<double> points;
    std::size_t chunk_index = 0;
    std::size_t point_dim = 0;
    std::size_t num_points = 0;
    double max_radius = 1.0;
    bool is_compressed = false;

    bool isValid() const
    {
        if (points.empty() || point_dim == 0 || num_points == 0 || !std::isfinite(max_radius) ||
            max_radius < 0.0 || point_dim > std::numeric_limits<std::size_t>::max() / num_points ||
            points.size() != point_dim * num_points)
        {
            return false;
        }
        return std::all_of(points.begin(), points.end(),
                           [](double value) { return std::isfinite(value); });
    }
};

struct StreamingStats
{
    double total_processing_time_ms = 0.0;
    std::size_t total_chunks_processed = 0;
    std::size_t total_points_processed = 0;
    double average_chunk_time_ms = 0.0;
    double memory_efficiency = 0.0;
    double processing_efficiency = 0.0;
    std::size_t peak_memory_usage_mb = 0;
    std::size_t memory_allocations = 0;
    std::size_t memory_deallocations = 0;
    std::string processing_details;
};

struct MemoryStats
{
    std::size_t current_memory_usage_mb = 0;
    std::size_t peak_memory_usage_mb = 0;
    std::size_t total_memory_allocated_mb = 0;
    std::size_t total_memory_freed_mb = 0;
    std::size_t memory_allocations = 0;
    std::size_t memory_deallocations = 0;
    double allocation_rate_mb_per_sec = 0.0;
    double deallocation_rate_mb_per_sec = 0.0;
    std::string memory_details;
};

struct PoolStats
{
    std::size_t total_pool_size_mb = 0;
    std::size_t allocated_memory_mb = 0;
    std::size_t free_memory_mb = 0;
    std::size_t total_allocations = 0;
    std::size_t total_deallocations = 0;
    double fragmentation_ratio = 0.0;
    std::string pool_details;
};

struct NetworkConfig
{
    std::string endpoint;
    std::uint16_t port = 0;
    std::string protocol = "tcp";
};

class DataStream
{
public:
    virtual ~DataStream() = default;
    virtual bool hasNext() = 0;
    virtual std::vector<double> getNextChunk() = 0;
    virtual void reset() = 0;
    virtual std::size_t totalPoints() = 0;
    virtual std::size_t pointDimension() = 0;
    virtual StreamInfo getStreamInfo() = 0;
};

class StreamingProcessor
{
public:
    static errors::ErrorResult<std::unique_ptr<StreamingProcessor>>
    create(const StreamingConfig &config);

    ~StreamingProcessor();

    errors::ErrorResult<std::vector<Pair>> processStreaming(const DataStream &data_stream,
                                                            const VRConfig &config);

    errors::ErrorResult<std::vector<Pair>>
    processStreamingProgressive(const DataStream &data_stream, const VRConfig &config,
                                std::function<void(const std::vector<Pair> &)> callback);

    const StreamingStats &getStreamingStats() const { return streaming_stats_; }
    const MemoryStats &getMemoryStats() const { return memory_stats_; }

private:
    explicit StreamingProcessor(const StreamingConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    StreamingConfig config_;
    StreamingStats streaming_stats_;
    MemoryStats memory_stats_;

    errors::ErrorResult<void> initializeMemoryManagement();
    errors::ErrorResult<DataChunk> getNextChunk(const DataStream &data_stream);
    errors::ErrorResult<std::vector<Pair>> processChunk(const DataChunk &chunk,
                                                        const VRConfig &config);
    errors::ErrorResult<void> manageMemoryUsage(std::size_t required_memory);
    errors::ErrorResult<void> optimizeChunkSize(const StreamInfo &stream_info);
};

class MemoryPool
{
public:
    static errors::ErrorResult<std::unique_ptr<MemoryPool>> create(std::size_t pool_size_mb);

    ~MemoryPool();

    errors::ErrorResult<void *> allocate(std::size_t size_bytes);
    errors::ErrorResult<void> deallocate(void *ptr, std::size_t size_bytes);

    const PoolStats &getPoolStats() const { return pool_stats_; }

private:
    explicit MemoryPool(std::size_t pool_size_mb);

    class Impl;
    std::unique_ptr<Impl> impl_;
    PoolStats pool_stats_;
};

class ChunkProcessor
{
public:
    static errors::ErrorResult<std::unique_ptr<ChunkProcessor>>
    create(const StreamingConfig &config);

    ~ChunkProcessor();

    errors::ErrorResult<std::vector<Pair>> processChunk(const DataChunk &chunk,
                                                        const VRConfig &config);

    errors::ErrorResult<std::vector<std::vector<Pair>>>
    processChunksParallel(const std::vector<DataChunk> &chunks, const VRConfig &config);

private:
    explicit ChunkProcessor(const StreamingConfig &config);

    class Impl;
    std::unique_ptr<Impl> impl_;
    StreamingConfig config_;
};

class StreamingProcessorFactory
{
public:
    static errors::ErrorResult<std::unique_ptr<StreamingProcessor>>
    createForFile(const std::string &file_path, const StreamingConfig &config);

    static errors::ErrorResult<std::unique_ptr<StreamingProcessor>>
    createForMemory(const std::vector<double> &data, const StreamingConfig &config);

    static errors::ErrorResult<std::unique_ptr<StreamingProcessor>>
    createForNetwork(const NetworkConfig &network_config, const StreamingConfig &config);
};

} // namespace nerve::persistence::adaptive_acceleration::streaming
