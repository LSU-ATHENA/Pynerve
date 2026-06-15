#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/persistence/adaptive_acceleration/streaming/streaming_processor.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace
{

class TestDataStream : public nerve::persistence::adaptive_acceleration::streaming::DataStream
{
public:
    explicit TestDataStream(std::vector<double> data, std::size_t point_dim, std::size_t chunk_sz)
        : data_(std::move(data))
        , point_dim_(point_dim)
        , chunk_size_(chunk_sz)
        , pos_(0)
    {
        n_points_ = data_.size() / point_dim_;
    }

    bool hasNext() override { return pos_ < n_points_; }

    std::vector<double> getNextChunk() override
    {
        std::size_t remaining = n_points_ - pos_;
        std::size_t take = std::min(chunk_size_, remaining);
        std::size_t start = pos_ * point_dim_;
        std::size_t count = take * point_dim_;
        pos_ += take;
        return std::vector<double>(data_.begin() + static_cast<std::ptrdiff_t>(start),
                                   data_.begin() + static_cast<std::ptrdiff_t>(start + count));
    }

    void reset() override { pos_ = 0; }

    std::size_t totalPoints() override { return n_points_; }

    std::size_t pointDimension() override { return point_dim_; }

    nerve::persistence::adaptive_acceleration::streaming::StreamInfo getStreamInfo() override
    {
        nerve::persistence::adaptive_acceleration::streaming::StreamInfo info;
        info.totalPoints = n_points_;
        info.point_dim = point_dim_;
        info.chunk_size = chunk_size_;
        info.num_chunks = (n_points_ + chunk_size_ - 1) / chunk_size_;
        info.max_radius = 2.0;
        return info;
    }

private:
    std::vector<double> data_;
    std::size_t point_dim_;
    std::size_t chunk_size_;
    std::size_t n_points_;
    std::size_t pos_;
};

nerve::common::VRConfig testConfig()
{
    nerve::common::VRConfig cfg;
    cfg.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;
    cfg.max_dim = 1;
    cfg.max_radius = 2.0;
    return cfg;
}

nerve::core::BufferView<const double> viewOf(const std::vector<double> &v)
{
    return nerve::core::BufferView<const double>(v.data(), v.size());
}

} // namespace

#ifdef NERVE_ENABLE_STREAMING

TEST(SOTAStreamingTest, WindowedPersistenceMatchesBatch)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0, 0.2, 0.3};
    constexpr std::size_t dim = 2;

    nerve::persistence::adaptive_acceleration::streaming::StreamingConfig sconfig;
    sconfig.chunk_size = 3;
    sconfig.enable_compression = false;
    sconfig.enable_multi_threading = false;

    auto processor =
        nerve::persistence::adaptive_acceleration::streaming::StreamingProcessor::create(sconfig);
    ASSERT_TRUE(processor.isSuccess());

    auto stream = std::make_unique<TestDataStream>(points, dim, 3);
    const auto streaming_result = processor.value()->processStreaming(*stream, testConfig());
    EXPECT_TRUE(streaming_result.isSuccess());

    const auto batch_result =
        nerve::persistence::computeVrPersistenceFastResult(viewOf(points), dim, testConfig());
    ASSERT_TRUE(batch_result.isSuccess());

    const auto &streaming_pairs = streaming_result.value();
    const auto &batch_pairs = batch_result.value();
    EXPECT_EQ(streaming_pairs.size(), batch_pairs.size());
}

TEST(SOTAStreamingTest, StreamingProcessorReportsStats)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};
    constexpr std::size_t dim = 2;

    nerve::persistence::adaptive_acceleration::streaming::StreamingConfig sconfig;
    sconfig.chunk_size = 2;
    sconfig.enable_compression = false;

    auto processor =
        nerve::persistence::adaptive_acceleration::streaming::StreamingProcessor::create(sconfig);
    ASSERT_TRUE(processor.isSuccess());

    auto stream = std::make_unique<TestDataStream>(points, dim, 2);
    auto result = processor.value()->processStreaming(*stream, testConfig());
    EXPECT_TRUE(result.isSuccess());

    const auto &stats = processor.value()->getStreamingStats();
    EXPECT_GE(stats.total_chunks_processed, 1u);
    EXPECT_GE(stats.total_points_processed, 3u);

    const auto &mem_stats = processor.value()->getMemoryStats();
    EXPECT_GE(mem_stats.current_memory_usage_mb, 0u);
}

TEST(SOTAStreamingTest, StreamingWithEmptyChunks)
{
    const std::vector<double> points{};
    constexpr std::size_t dim = 2;

    nerve::persistence::adaptive_acceleration::streaming::StreamingConfig sconfig;
    sconfig.chunk_size = 4;
    sconfig.enable_compression = false;

    auto processor =
        nerve::persistence::adaptive_acceleration::streaming::StreamingProcessor::create(sconfig);
    ASSERT_TRUE(processor.isSuccess());

    auto stream = std::make_unique<TestDataStream>(points, dim, 4);
    auto result = processor.value()->processStreaming(*stream, testConfig());
    EXPECT_TRUE(result.isSuccess());
    EXPECT_TRUE(result.value().empty());
}

#else

TEST(SOTAStreamingTest, StreamingNotEnabled)
{
    GTEST_SKIP() << "NERVE_ENABLE_STREAMING is not defined";
}

#endif
