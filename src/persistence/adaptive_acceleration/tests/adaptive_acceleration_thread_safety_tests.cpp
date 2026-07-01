#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

namespace
{

nerve::common::VRConfig testConfig()
{
    nerve::common::VRConfig cfg;
    cfg.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;
    cfg.max_dim = 1;
    cfg.max_radius = 3.0;
    return cfg;
}

nerve::core::BufferView<const double> viewOf(const std::vector<double> &v)
{
    return nerve::core::BufferView<const double>(v.data(), v.size());
}

} // namespace

TEST(SAThreadSafetyTest, MultipleThreadsCallCompute)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;
    constexpr int num_threads = 4;

    std::vector<std::thread> threads;
    std::mutex results_mutex;
    std::vector<std::size_t> result_sizes;

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&points, dim, &results_mutex, &result_sizes]() {
            const auto result = nerve::persistence::computeVrPersistenceFastResult(
                viewOf(points), dim, testConfig());
            if (result.isSuccess())
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                result_sizes.push_back(result.value().size());
            }
        });
    }

    for (auto &th : threads)
    {
        th.join();
    }

    EXPECT_EQ(result_sizes.size(), num_threads);
    for (std::size_t i = 1; i < result_sizes.size(); ++i)
    {
        EXPECT_EQ(result_sizes[i], result_sizes[0])
            << "thread " << i << " produced different result size than thread 0";
    }
}

TEST(SAThreadSafetyTest, ConcurrentResultsConsistent)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 0.0, 1.0};
    constexpr std::size_t dim = 2;
    constexpr int num_threads = 4;

    std::vector<std::thread> threads;
    std::mutex mutex;
    std::vector<std::vector<nerve::persistence::Pair>> all_results;

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&points, dim, &mutex, &all_results]() {
            const auto result =
                nerve::persistence::computeVrPersistenceFast(viewOf(points), dim, testConfig());
            std::lock_guard<std::mutex> lock(mutex);
            all_results.push_back(result);
        });
    }

    for (auto &th : threads)
    {
        th.join();
    }

    ASSERT_EQ(all_results.size(), num_threads);
    for (std::size_t i = 1; i < all_results.size(); ++i)
    {
        ASSERT_EQ(all_results[i].size(), all_results[0].size());
        for (std::size_t j = 0; j < all_results[i].size(); ++j)
        {
            EXPECT_EQ(all_results[i][j].birth, all_results[0][j].birth);
            EXPECT_EQ(all_results[i][j].death, all_results[0][j].death);
            EXPECT_EQ(all_results[i][j].dimension, all_results[0][j].dimension);
        }
    }
}

TEST(SAThreadSafetyTest, ConcurrentEngineAccess)
{
    constexpr int num_threads = 4;
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0};
    constexpr std::size_t dim = 2;

    std::vector<std::thread> threads;
    std::mutex mutex;
    int success_count = 0;

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&points, dim, &mutex, &success_count]() {
            auto engine =
                nerve::persistence::adaptive_acceleration::AdaptiveAccelerationVrEngine::create(
                    testConfig());
            if (engine.isSuccess())
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++success_count;
            }
        });
    }

    for (auto &th : threads)
    {
        th.join();
    }

    EXPECT_EQ(success_count, num_threads);
}
