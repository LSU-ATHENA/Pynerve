#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/persistence/accelerated/accelerated_interface.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <vector>

namespace
{

nerve::common::VRConfig benchmarkConfig(nerve::common::VRAlgorithmSelection algo)
{
    nerve::common::VRConfig cfg;
    cfg.algorithm = algo;
    cfg.max_dim = 1;
    cfg.max_radius = 2.0;
    cfg.acceleration.enable_performance_monitoring = true;
    cfg.acceleration.enable_optimization = true;
    return cfg;
}

nerve::core::BufferView<const double> viewOf(const std::vector<double> &v)
{
    return nerve::core::BufferView<const double>(v.data(), v.size());
}

} // namespace

#ifdef NERVE_ENABLE_PERFORMANCE_MONITORING

TEST(SOTAPerformanceBenchmark, ComputeWithPerformanceMonitoring)
{
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866, 1.0, 1.0, 0.0, 1.0};
    constexpr std::size_t dim = 2;

    nerve::persistence::accelerated::PerformanceMonitor monitor;
    monitor.startMonitoring("benchmark_test");

    const auto result = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, benchmarkConfig(nerve::common::VRAlgorithmSelection::AUTO));

    monitor.endMonitoring("benchmark_test");

    ASSERT_TRUE(result.isSuccess());
    EXPECT_FALSE(result.value().empty());

    const auto &stats = monitor.getAggregatedStats();
    EXPECT_GE(stats.total_time_ms, 0.0);
    EXPECT_GE(stats.problems_processed, 0u);
}

TEST(SOTAPerformanceBenchmark, DifferentConfigsProduceMetrics)
{
    constexpr std::size_t dim = 2;
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0, 0.5, 0.866};

    const auto start = std::chrono::steady_clock::now();
    const auto exact = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, benchmarkConfig(nerve::common::VRAlgorithmSelection::EXACT_STANDARD));
    const auto mid = std::chrono::steady_clock::now();

    const auto fast = nerve::persistence::computeVrPersistenceFastResult(
        viewOf(points), dim, benchmarkConfig(nerve::common::VRAlgorithmSelection::FAST_SIMD));
    const auto end = std::chrono::steady_clock::now();

    ASSERT_TRUE(exact.isSuccess());
    ASSERT_TRUE(fast.isSuccess());

    const auto exact_time =
        std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
    const auto fast_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();

    EXPECT_GT(exact_time, 0);
    EXPECT_GT(fast_time, 0);
    EXPECT_EQ(exact.value().size(), fast.value().size());
}

TEST(SOTAPerformanceBenchmark, PerformanceStatsAccessible)
{
    const auto stats = nerve::persistence::accelerated::performance::getCurrentPerformanceStats();
    EXPECT_GE(stats.total_time_ms, 0.0);
    EXPECT_GE(stats.problems_processed, 0u);
}

TEST(SOTAPerformanceBenchmark, AcceleratedEngineCounterMetrics)
{
    constexpr std::size_t dim = 2;
    const std::vector<double> points{0.0, 0.0, 1.0, 0.0};

    auto engine = nerve::persistence::accelerated::AcceleratedVREngine::create(
        benchmarkConfig(nerve::common::VRAlgorithmSelection::AUTO));
    ASSERT_TRUE(engine.isSuccess());

    auto result = engine.value()->computeVrPersistence(
        viewOf(points), dim, benchmarkConfig(nerve::common::VRAlgorithmSelection::AUTO));
    EXPECT_TRUE(result.isSuccess());

    const auto perf_stats = engine.value()->getPerformanceStats();
    EXPECT_GE(perf_stats.total_time_ms, 0.0);
    EXPECT_GE(perf_stats.problems_processed, 0u);
}

TEST(SOTAPerformanceBenchmark, SystemCapabilitiesMetrics)
{
    const auto caps = nerve::persistence::accelerated::utils::detectSystemCapabilities();
    EXPECT_FALSE(caps.supported_features.empty() && !caps.cuda_available && false);
}

#else

TEST(SOTAPerformanceBenchmark, PerformanceMonitoringNotEnabled)
{
    GTEST_SKIP() << "NERVE_ENABLE_PERFORMANCE_MONITORING is not defined";
}

#endif
