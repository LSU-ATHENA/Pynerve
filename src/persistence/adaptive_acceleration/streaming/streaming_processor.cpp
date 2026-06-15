
#include "memory/safe_memory_pool.hpp"
#include "nerve/persistence/adaptive_acceleration/streaming/streaming_processor.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <limits>
#include <unordered_map>
#include <utility>

namespace nerve::persistence::adaptive_acceleration::streaming
{
namespace
{

constexpr std::size_t kBytesPerMb = 1024ULL * 1024ULL;

bool checkedProduct(std::size_t lhs, std::size_t rhs, std::size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        out = 0;
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checkedAdd(std::size_t lhs, std::size_t rhs, std::size_t &out)
{
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    {
        out = 0;
        return false;
    }
    out = lhs + rhs;
    return true;
}

std::size_t bytesToMb(std::size_t bytes)
{
    if (bytes > std::numeric_limits<std::size_t>::max() - (kBytesPerMb - 1))
    {
        return std::numeric_limits<std::size_t>::max() / kBytesPerMb;
    }
    return (bytes + kBytesPerMb - 1) / kBytesPerMb;
}

} // namespace

class StreamingProcessor::Impl
{
public:
    explicit Impl(const StreamingConfig &config)
        : config(config)
    {}

    StreamingConfig config;
};

errors::ErrorResult<std::unique_ptr<StreamingProcessor>>
StreamingProcessor::create(const StreamingConfig &config)
{
    if (!std::isfinite(config.target_efficiency) || config.target_efficiency < 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<StreamingProcessor>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::unique_ptr<StreamingProcessor> processor(new StreamingProcessor(config));
    return errors::ErrorResult<std::unique_ptr<StreamingProcessor>>::success(std::move(processor));
}

errors::ErrorResult<std::vector<Pair>>
StreamingProcessor::processStreaming(const DataStream &data_stream, const VRConfig &config)
{
    DataStream &mutable_stream = const_cast<DataStream &>(data_stream);
    mutable_stream.reset();
    auto init = initializeMemoryManagement();
    if (init.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(init.errorCode());
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<Pair> all_pairs;
    std::size_t chunk_count = 0;
    while (mutable_stream.hasNext())
    {
        auto chunk_result = getNextChunk(mutable_stream);
        if (chunk_result.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(chunk_result.errorCode());
        }
        auto pairs_result = processChunk(chunk_result.value(), config);
        if (pairs_result.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(pairs_result.errorCode());
        }
        const auto &pairs = pairs_result.value();
        if (pairs.size() > all_pairs.max_size() - all_pairs.size())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        all_pairs.insert(all_pairs.end(), pairs.begin(), pairs.end());
        ++chunk_count;
        if (chunk_result.value().num_points >
            std::numeric_limits<std::size_t>::max() - streaming_stats_.total_points_processed)
        {
            return errors::ErrorResult<std::vector<Pair>>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        streaming_stats_.total_points_processed += chunk_result.value().num_points;
    }
    const auto end = std::chrono::steady_clock::now();

    streaming_stats_.total_processing_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    streaming_stats_.total_chunks_processed = chunk_count;
    streaming_stats_.average_chunk_time_ms =
        chunk_count == 0
            ? 0.0
            : streaming_stats_.total_processing_time_ms / static_cast<double>(chunk_count);
    streaming_stats_.processing_efficiency = chunk_count == 0 ? 0.0 : 1.0;
    streaming_stats_.processing_details = "deterministic chunked streaming";
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(all_pairs));
}

errors::ErrorResult<std::vector<Pair>> StreamingProcessor::processStreamingProgressive(
    const DataStream &data_stream, const VRConfig &config,
    std::function<void(const std::vector<Pair> &)> callback)
{
    DataStream &mutable_stream = const_cast<DataStream &>(data_stream);
    mutable_stream.reset();
    auto init = initializeMemoryManagement();
    if (init.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(init.errorCode());
    }

    std::vector<Pair> running_pairs;
    while (mutable_stream.hasNext())
    {
        auto chunk_result = getNextChunk(mutable_stream);
        if (chunk_result.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(chunk_result.errorCode());
        }
        auto pairs_result = processChunk(chunk_result.value(), config);
        if (pairs_result.isError())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(pairs_result.errorCode());
        }
        const auto &pairs = pairs_result.value();
        if (pairs.size() > running_pairs.max_size() - running_pairs.size())
        {
            return errors::ErrorResult<std::vector<Pair>>::error(
                errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        running_pairs.insert(running_pairs.end(), pairs.begin(), pairs.end());
        if (callback)
        {
            callback(running_pairs);
        }
    }
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(running_pairs));
}

StreamingProcessor::StreamingProcessor(const StreamingConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
    , streaming_stats_()
    , memory_stats_()
{}

StreamingProcessor::~StreamingProcessor() = default;

errors::ErrorResult<void> StreamingProcessor::initializeMemoryManagement()
{
    memory_stats_ = MemoryStats{};
    streaming_stats_ = StreamingStats{};
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<DataChunk> StreamingProcessor::getNextChunk(const DataStream &data_stream)
{
    DataStream &mutable_stream = const_cast<DataStream &>(data_stream);
    if (!mutable_stream.hasNext())
    {
        return errors::ErrorResult<DataChunk>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    DataChunk chunk;
    chunk.points = mutable_stream.getNextChunk();
    chunk.point_dim = mutable_stream.pointDimension();
    if (chunk.point_dim == 0 || chunk.points.empty() || chunk.points.size() % chunk.point_dim != 0)
    {
        return errors::ErrorResult<DataChunk>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    chunk.num_points = chunk.points.size() / chunk.point_dim;
    chunk.chunk_index = streaming_stats_.total_chunks_processed;
    chunk.max_radius = 1.0;
    chunk.is_compressed = false;
    return errors::ErrorResult<DataChunk>::success(std::move(chunk));
}

errors::ErrorResult<std::vector<Pair>> StreamingProcessor::processChunk(const DataChunk &chunk,
                                                                        const VRConfig &config)
{
    if (!chunk.isValid())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::size_t required_bytes = 0;
    if (!checkedProduct(chunk.points.size(), sizeof(double), required_bytes))
    {
        return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    auto memory_status = manageMemoryUsage(required_bytes);
    if (memory_status.isError())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(memory_status.errorCode());
    }

    core::BufferView<const double> pointsView(chunk.points.data(), chunk.points.size());
    std::vector<Pair> pairs = computeVrPersistenceFast(pointsView, chunk.point_dim, config);
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
}

errors::ErrorResult<void> StreamingProcessor::manageMemoryUsage(std::size_t required_memory)
{
    const std::size_t required_mb = bytesToMb(required_memory);
    const std::size_t limit_mb = std::max<std::size_t>(1, config_.memory_limit_mb);
    if (required_mb > limit_mb)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    memory_stats_.current_memory_usage_mb = required_mb;
    memory_stats_.peak_memory_usage_mb = std::max(memory_stats_.peak_memory_usage_mb, required_mb);
    if (!checkedAdd(memory_stats_.total_memory_allocated_mb, required_mb,
                    memory_stats_.total_memory_allocated_mb))
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    if (memory_stats_.memory_allocations < std::numeric_limits<std::size_t>::max())
    {
        memory_stats_.memory_allocations += 1;
    }
    streaming_stats_.peak_memory_usage_mb =
        std::max(streaming_stats_.peak_memory_usage_mb, required_mb);
    return errors::ErrorResult<void>::ok();
}

errors::ErrorResult<void> StreamingProcessor::optimizeChunkSize(const StreamInfo &stream_info)
{
    if (stream_info.point_dim == 0 || stream_info.totalPoints == 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    if (config_.chunk_size == 0)
    {
        std::size_t memory_bytes = 0;
        std::size_t point_bytes = 0;
        if (!checkedProduct(config_.memory_limit_mb, kBytesPerMb, memory_bytes) ||
            !checkedProduct(stream_info.point_dim, sizeof(double), point_bytes) || point_bytes == 0)
        {
            return errors::ErrorResult<void>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
        }
        config_.chunk_size = std::max<std::size_t>(1, memory_bytes / point_bytes);
    }
    return errors::ErrorResult<void>::ok();
}

class MemoryPool::Impl
{
public:
    explicit Impl(std::size_t pool_size_bytes)
        : poolSizeBytes(pool_size_bytes)
        , pool(pool_size_bytes)
    {}

    std::size_t poolSizeBytes = 0;
    std::size_t allocated_bytes = 0;
    std::unordered_map<void *, std::size_t> allocations;
    nerve::memory::RawArrayPool pool;
};

errors::ErrorResult<std::unique_ptr<MemoryPool>> MemoryPool::create(std::size_t pool_size_mb)
{
    if (pool_size_mb == 0)
    {
        return errors::ErrorResult<std::unique_ptr<MemoryPool>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::size_t pool_size_bytes = 0;
    if (!checkedProduct(pool_size_mb, kBytesPerMb, pool_size_bytes))
    {
        return errors::ErrorResult<std::unique_ptr<MemoryPool>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    (void)pool_size_bytes;
    std::unique_ptr<MemoryPool> pool(new MemoryPool(pool_size_mb));
    return errors::ErrorResult<std::unique_ptr<MemoryPool>>::success(std::move(pool));
}

errors::ErrorResult<void *> MemoryPool::allocate(std::size_t size_bytes)
{
    if (size_bytes == 0)
    {
        return errors::ErrorResult<void *>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::size_t new_allocated = 0;
    if (!checkedAdd(impl_->allocated_bytes, size_bytes, new_allocated) ||
        new_allocated > impl_->poolSizeBytes)
    {
        return errors::ErrorResult<void *>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    void *ptr = impl_->pool.allocate(size_bytes);
    if (ptr == nullptr)
    {
        return errors::ErrorResult<void *>::error(errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    impl_->allocations.insert_or_assign(ptr, size_bytes);
    impl_->allocated_bytes = new_allocated;
    pool_stats_.allocated_memory_mb = bytesToMb(impl_->allocated_bytes);
    pool_stats_.free_memory_mb = bytesToMb(impl_->poolSizeBytes - impl_->allocated_bytes);
    if (pool_stats_.total_allocations < std::numeric_limits<std::size_t>::max())
    {
        pool_stats_.total_allocations += 1;
    }
    return errors::ErrorResult<void *>::success(static_cast<void *>(ptr));
}

errors::ErrorResult<void> MemoryPool::deallocate(void *ptr, std::size_t size_bytes)
{
    if (ptr == nullptr)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    const auto it = impl_->allocations.find(ptr);
    if (it == impl_->allocations.end())
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    const std::size_t released = it->second;
    if (size_bytes != 0 && size_bytes != released)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    impl_->pool.deallocate(ptr, released);
    impl_->allocated_bytes -= released;
    impl_->allocations.erase(it);
    pool_stats_.allocated_memory_mb = bytesToMb(impl_->allocated_bytes);
    pool_stats_.free_memory_mb = bytesToMb(impl_->poolSizeBytes - impl_->allocated_bytes);
    if (pool_stats_.total_deallocations < std::numeric_limits<std::size_t>::max())
    {
        pool_stats_.total_deallocations += 1;
    }
    return errors::ErrorResult<void>::ok();
}

MemoryPool::MemoryPool(std::size_t pool_size_mb)
    : impl_(nullptr)
    , pool_stats_()
{
    std::size_t pool_size_bytes = 0;
    checkedProduct(pool_size_mb, kBytesPerMb, pool_size_bytes);
    impl_ = std::make_unique<Impl>(pool_size_bytes);
    pool_stats_.total_pool_size_mb = pool_size_mb;
    pool_stats_.free_memory_mb = pool_size_mb;
    pool_stats_.pool_details = "heap-backed pool";
}

MemoryPool::~MemoryPool() = default;

class ChunkProcessor::Impl
{
public:
    explicit Impl(const StreamingConfig &config)
        : config(config)
    {}

    StreamingConfig config;
};

errors::ErrorResult<std::unique_ptr<ChunkProcessor>>
ChunkProcessor::create(const StreamingConfig &config)
{
    if (!std::isfinite(config.target_efficiency) || config.target_efficiency < 0.0)
    {
        return errors::ErrorResult<std::unique_ptr<ChunkProcessor>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    std::unique_ptr<ChunkProcessor> processor(new ChunkProcessor(config));
    return errors::ErrorResult<std::unique_ptr<ChunkProcessor>>::success(std::move(processor));
}

errors::ErrorResult<std::vector<Pair>> ChunkProcessor::processChunk(const DataChunk &chunk,
                                                                    const VRConfig &config)
{
    if (!chunk.isValid())
    {
        return errors::ErrorResult<std::vector<Pair>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    core::BufferView<const double> pointsView(chunk.points.data(), chunk.points.size());
    std::vector<Pair> pairs = computeVrPersistenceFast(pointsView, chunk.point_dim, config);
    return errors::ErrorResult<std::vector<Pair>>::success(std::move(pairs));
}

errors::ErrorResult<std::vector<std::vector<Pair>>>
ChunkProcessor::processChunksParallel(const std::vector<DataChunk> &chunks, const VRConfig &config)
{
    std::vector<std::future<errors::ErrorResult<std::vector<Pair>>>> futures;
    if (chunks.size() > futures.max_size())
    {
        return errors::ErrorResult<std::vector<std::vector<Pair>>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    futures.reserve(chunks.size());
    for (const DataChunk &chunk : chunks)
    {
        futures.push_back(std::async(
            std::launch::async, [this, chunk, &config]() { return processChunk(chunk, config); }));
    }
    std::vector<std::vector<Pair>> output;
    if (chunks.size() > output.max_size())
    {
        return errors::ErrorResult<std::vector<std::vector<Pair>>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    output.reserve(chunks.size());
    for (auto &future : futures)
    {
        auto result = future.get();
        if (result.isError())
        {
            return errors::ErrorResult<std::vector<std::vector<Pair>>>::error(result.errorCode());
        }
        output.push_back(result.value());
    }
    return errors::ErrorResult<std::vector<std::vector<Pair>>>::success(std::move(output));
}

ChunkProcessor::ChunkProcessor(const StreamingConfig &config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
{}

ChunkProcessor::~ChunkProcessor() = default;

errors::ErrorResult<std::unique_ptr<StreamingProcessor>>
StreamingProcessorFactory::createForFile(const std::string &file_path,
                                         const StreamingConfig &config)
{
    if (file_path.empty() || !std::filesystem::exists(file_path))
    {
        return errors::ErrorResult<std::unique_ptr<StreamingProcessor>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    return StreamingProcessor::create(config);
}

errors::ErrorResult<std::unique_ptr<StreamingProcessor>>
StreamingProcessorFactory::createForMemory(const std::vector<double> &data,
                                           const StreamingConfig &config)
{
    if (data.empty() ||
        !std::all_of(data.begin(), data.end(), [](double value) { return std::isfinite(value); }))
    {
        return errors::ErrorResult<std::unique_ptr<StreamingProcessor>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    return StreamingProcessor::create(config);
}

errors::ErrorResult<std::unique_ptr<StreamingProcessor>>
StreamingProcessorFactory::createForNetwork(const NetworkConfig &network_config,
                                            const StreamingConfig &config)
{
    if (network_config.endpoint.empty() || network_config.port == 0)
    {
        return errors::ErrorResult<std::unique_ptr<StreamingProcessor>>::error(
            errors::ErrorCode::E54_PH4_INVALID_INPUT);
    }
    return StreamingProcessor::create(config);
}

} // namespace nerve::persistence::adaptive_acceleration::streaming
