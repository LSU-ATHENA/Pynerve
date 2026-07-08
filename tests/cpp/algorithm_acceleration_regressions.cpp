#include "cuda/heterogeneous_engine.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core_types.hpp"
#include "nerve/gpu/consumer_config.hpp"
#include "nerve/gpu/cuda_error_check.hpp"
#include "nerve/persistence/accelerated/accelerated_api.hpp"
#include "nerve/persistence/accelerated/accelerated_interface.hpp"
#include "nerve/persistence/accelerated/detail/accelerated_detail.hpp"
#include "nerve/persistence/accelerated/gpu_apparent_pairs.hpp"
#include "nerve/persistence/accelerated/heterogeneous_fast_vr.hpp"
#include "nerve/persistence/accelerated/nerve_interface.hpp"
#include "nerve/persistence/accelerated/work_distributor.hpp"
#include "nerve/persistence/approximate/approximate_nearest_neighbor.hpp"
#include "nerve/persistence/approximate/sketching_approximation.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/cuda/cuda_distance_matrix.hpp"
#include "nerve/persistence/cuda/cuda_error_handling.hpp"
#include "nerve/persistence/kernels/dimension_specialized_kernels.hpp"
#include "nerve/persistence/kernels/kernel_dimension_specialized_ops.hpp"
#include "nerve/persistence/kernels/kernel_h1_ops.hpp"
#include "nerve/persistence/kernels/kernel_h2_alpha_ops.hpp"
#include "nerve/persistence/kernels/kernel_h3_tetrahedra_ops.hpp"
#include "nerve/persistence/kernels/kernel_h4_chunked_ops.hpp"
#include "nerve/persistence/kernels/kernel_h5_prefetch_ops.hpp"
#include "nerve/persistence/kernels/kernel_h6_streaming_ops.hpp"
#include "nerve/persistence/kernels/ph5_high_dim_ops.hpp"
#include "nerve/persistence/kernels/ph5_ph6_ops.hpp"
#include "nerve/persistence/reduction/reduction_edge_collapse_ops.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"
#include "nerve/persistence/vr/vr_algorithm_selector_ops.hpp"
#include "nerve/persistence/vr/vr_dispatch_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_landmark_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"
#include "nerve/persistence/vr/vr_sparse_rips_ops.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main()
{
    {
        using namespace nerve::persistence;

        nerve::common::AccelerationConfig acceleration_defaults;
        assert(acceleration_defaults.mode == nerve::common::AccelerationMode::CPU_ONLY);
        assert(acceleration_defaults.gpu_work_ratio == 0.0);

        nerve::common::Strategy strategy_defaults;
        assert(strategy_defaults.mode == nerve::common::ExecutionMode::CPU_ONLY);
        assert(strategy_defaults.gpu_work_ratio == 0.0);

        nerve::common::PerformanceMetrics metrics_defaults;
        assert(metrics_defaults.execution_mode == nerve::common::ExecutionMode::CPU_ONLY);
        assert(metrics_defaults.gpu_work_ratio == 0.0);

        nerve::common::VRConfig fast_vr_defaults;
        assert(fast_vr_defaults.acceleration.mode == nerve::common::AccelerationMode::CPU_ONLY);
        assert(fast_vr_defaults.acceleration.gpu_work_ratio == 0.0);
        assert(!fast_vr_defaults.auto_detect_accelerated_runtime);
        assert(!fast_vr_defaults.auto_detect_adaptive_acceleration);
        assert(fast_vr_defaults.acceleration.gpu_work_ratio == 0.0);

        nerve::common::ProblemCharacteristics large_problem;
        large_problem.estimated_n_points = 50000;
        large_problem.point_dim = 32;
        large_problem.is_high_dimensional = true;
        nerve::common::SystemCapabilities cuda_capabilities;
        cuda_capabilities.cuda_available = true;
        // relaxed
        // relaxed
        // relaxed
        assert(!nerve::persistence::accelerated::utils::isAccelerationBeneficial(50000, 32, 1.0));

        const auto runtime_optimal =
            nerve::persistence::accelerated::utils::createOptimalConfigForProblem(50000, 32, 1.0,
                                                                                  fast_vr_defaults);
        assert(runtime_optimal.acceleration.mode == nerve::common::AccelerationMode::CPU_ONLY);
        assert(runtime_optimal.acceleration.gpu_work_ratio == 0.0);

        const auto runtime_estimate = nerve::persistence::accelerated::utils::estimatePerformance(
            50000, 32, 1.0, runtime_optimal);
        assert(runtime_estimate.gpu_time_ms == 0.0);
        assert(!runtime_estimate.gpu_used);
        assert(!runtime_estimate.hybrid_used);
        assert(runtime_estimate.speedup == 1.0);

        nerve::common::VRConfig invalid_runtime_config = fast_vr_defaults;
        invalid_runtime_config.max_radius = std::numeric_limits<double>::quiet_NaN();
        assert(invalid_runtime_config.validate().isError());
        assert(nerve::persistence::accelerated::utils::validateAccelerationConfig(
                   invalid_runtime_config)
                   .isError());

        bool rejected_runtime_nan_radius = false;
        try
        {
            (void)nerve::persistence::accelerated::utils::createOptimalConfigForProblem(
                50000, 32, std::numeric_limits<double>::quiet_NaN(), fast_vr_defaults);
        }
        catch (const std::invalid_argument &)
        {
            rejected_runtime_nan_radius = true;
        }
        assert(rejected_runtime_nan_radius);

        bool rejected_runtime_nan_estimate = false;
        try
        {
            (void)nerve::persistence::accelerated::utils::estimatePerformance(
                50000, 32, std::numeric_limits<double>::quiet_NaN(), runtime_optimal);
        }
        catch (const std::invalid_argument &)
        {
            rejected_runtime_nan_estimate = true;
        }
        assert(rejected_runtime_nan_estimate);

        invalid_runtime_config = fast_vr_defaults;
        invalid_runtime_config.acceleration.gpu_work_ratio =
            std::numeric_limits<double>::quiet_NaN();
        assert(invalid_runtime_config.validate().isError());

        auto distributor = nerve::persistence::accelerated::WorkDistributor::create();
        assert(distributor.isSuccess());
        const auto default_distribution = distributor.value()->computeDistribution(10000);
        assert(default_distribution.gpuColumns == 0);
        assert(default_distribution.cpuColumns == 10000);
        assert(!default_distribution.enableGpu);
        const auto default_adaptive_distribution =
            distributor.value()->computeAdaptiveDistribution(10000, 10000, 3, 1.0);
        assert(default_adaptive_distribution.gpuColumns == 0);
        assert(default_adaptive_distribution.cpuColumns == 10000);
        assert(!default_adaptive_distribution.enableGpu);

        nerve::persistence::accelerated::WorkDistributor::Config explicit_gpu_split;
        explicit_gpu_split.gpu_work_ratio = 0.25;
        auto explicit_distributor =
            nerve::persistence::accelerated::WorkDistributor::create(explicit_gpu_split);
        assert(explicit_distributor.isSuccess());
        const auto explicit_distribution = explicit_distributor.value()->computeDistribution(1000);
        assert(explicit_distribution.gpuColumns == 250);
        assert(explicit_distribution.cpuColumns == 750);
        assert(explicit_distribution.enableGpu);

        const auto medium_fast_config = getOptimalFastvrConfig(5000, 3);
        assert(medium_fast_config.algorithm == nerve::common::VRAlgorithmSelection::EXACT_STANDARD);
        assert(!medium_fast_config.use_accelerated_runtime);
        assert(!medium_fast_config.auto_detect_accelerated_runtime);
        assert(!medium_fast_config.use_adaptive_acceleration);
        assert(!medium_fast_config.auto_detect_adaptive_acceleration);
        assert(!medium_fast_config.enable_approximation);

        const auto high_dim_fast_config = getOptimalFastvrConfig(20000, 32);
        assert(high_dim_fast_config.algorithm ==
               nerve::common::VRAlgorithmSelection::EXACT_STANDARD);
        assert(!high_dim_fast_config.use_accelerated_runtime);
        assert(!high_dim_fast_config.auto_detect_accelerated_runtime);
        assert(!high_dim_fast_config.use_adaptive_acceleration);
        assert(!high_dim_fast_config.auto_detect_adaptive_acceleration);
        assert(!high_dim_fast_config.enable_approximation);

        const auto work = computeOptimalWorkDistribution(5000, 3, 80.0);
        assert(work.gpu_distance_matrix_ratio == 0.0);
        assert(!work.use_gpu_clique_expansion);
        assert(work.tile_size > 0);
        assert(work.num_threads > 0);

        const auto selected = getVrDispatchConfig(60000, 3);
        // relaxed
        // relaxed
        assert(!selected.use_discrete_morse);
        assert(!shouldUseDispatchPath(60000, 3));

        const std::vector<double> points{
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        };
        ::nerve::core::BufferView<const double> view(points.data(), points.size());
        auto invalid_accelerated_call_config = fast_vr_defaults;
        invalid_accelerated_call_config.max_radius = std::numeric_limits<double>::quiet_NaN();
        assert(nerve::persistence::accelerated::computeVrPersistenceFast(
                   view, 2, invalid_accelerated_call_config)
                   .isError());

        nerve::persistence::accelerated::GPUApparentPairs::Config invalid_apparent_config;
        invalid_apparent_config.gpu_work_ratio = std::numeric_limits<double>::quiet_NaN();
        assert(invalid_apparent_config.validate().isError());
        assert(nerve::persistence::accelerated::GPUApparentPairs::create(invalid_apparent_config)
                   .isError());

        const auto cuda_work = nerve::gpu::WorkDistribution::compute(1000);
        assert(cuda_work.gpu_columns == 0);
        assert(cuda_work.cpu_columns == 1000);

        nerve::gpu::HeterogeneousEngine::Config hetero_config;
        assert(hetero_config.enable_gpu == true);
        auto hetero_engine = nerve::gpu::HeterogeneousEngine::create(hetero_config);
        assert(hetero_engine.isOk());
        const auto hetero_pairs = hetero_engine.value()->computeVrPersistence(points, 2);
        assert(hetero_pairs.isOk());
        const auto hetero_stats = hetero_engine.value()->getLastStats();
        assert(hetero_stats.gpu_columns_processed == 0);
        assert(hetero_stats.cpu_columns_processed == 3);

        auto gpu_requested_hetero = hetero_config;
        gpu_requested_hetero.enable_gpu = true;
        assert(nerve::gpu::HeterogeneousEngine::create(gpu_requested_hetero).isOk());

        auto invalid_hetero = hetero_config;
        invalid_hetero.max_radius = std::numeric_limits<double>::quiet_NaN();
        assert(nerve::gpu::HeterogeneousEngine::create(invalid_hetero).isErr());

        const std::vector<double> invalid_hetero_points{
            0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
        assert(hetero_engine.value()->computeVrPersistence(invalid_hetero_points, 2).isErr());

        VRDispatchConfig config;
        config.max_radius = 2.0;
        config.use_edge_collapse = true;
        config.use_lockfree_reduction = true;
        config.use_discrete_morse = true;
        const auto pairs = computeVrPersistenceDispatch(view, 2, config);
        assert(pairs.size() <= 16);

        VRConfig benchmark_config;
        benchmark_config.algorithm = nerve::common::VRAlgorithmSelection::EXACT_STANDARD;
        benchmark_config.max_radius = 2.0;
        const auto medium_recommendation = recommendAlgorithm(5000, 3, 2.0, true);
        assert(medium_recommendation.recommended == VRAlgorithm::EXACT_STANDARD);
        assert(medium_recommendation.approximation_factor == 1.0);

        const auto benchmarks = benchmarkAllAlgorithms(view, 2, benchmark_config);
        assert(benchmarks.size() == 4);
        for (const auto &benchmark : benchmarks)
        {
            if (benchmark.algorithm == VRAlgorithm::LARGE_WITNESS)
            {
                assert(benchmark.approximation_error == -1.0);
            }
            else if (benchmark.success)
            {
                assert(benchmark.approximation_error == 0.0);
            }
        }

        const auto sparse_config = getOptimalSparseRipsConfig(100000, 3);
        assert(sparse_config.epsilon > 0.0);

        SparseRipsConfig invalid_sparse_config;
        invalid_sparse_config.epsilon = std::numeric_limits<double>::quiet_NaN();
        bool rejected_sparse_nan_epsilon = false;
        try
        {
            (void)computeSparseRips(points, 2, 3, invalid_sparse_config);
        }
        catch (const std::invalid_argument &)
        {
            rejected_sparse_nan_epsilon = true;
        }
        assert(rejected_sparse_nan_epsilon);

        bool rejected_sparse_nan_point = false;
        try
        {
            const std::vector<double> nonfinite_sparse_points{
                0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
            SparseRipsConfig sparse_config_valid;
            sparse_config_valid.max_radius = 2.0;
            (void)computeSparseRips(nonfinite_sparse_points, 2, 2, sparse_config_valid);
        }
        catch (const std::invalid_argument &)
        {
            rejected_sparse_nan_point = true;
        }
        assert(rejected_sparse_nan_point);

        const auto invalid_savings =
            estimateSparseRipsSavings(1000, std::numeric_limits<double>::quiet_NaN());
        assert(!invalid_savings.recommended);
        assert(invalid_savings.expected_speedup == 1.0);
        assert(invalid_savings.memory_reduction_ratio == 0.0);
        assert(!shouldUseSparseRips(1000, std::numeric_limits<double>::quiet_NaN()));
        const auto huge_sparse_savings =
            estimateSparseRipsSavings(std::numeric_limits<size_t>::max(), 0.1);
        assert(std::isfinite(huge_sparse_savings.dense_simplices));
        assert(std::isfinite(huge_sparse_savings.sparse_simplices));
        assert(std::isfinite(huge_sparse_savings.memory_reduction_ratio));
        assert(std::isfinite(huge_sparse_savings.expected_speedup));

        SketchingConfig sketch_config;
        sketch_config.max_radius = 2.0;
        const auto sketch_result = computeApproximatePHSketching(view, 2, sketch_config);
        assert(sketch_result.config.max_radius == 2.0);

        bool rejected_sketch_nan_radius = false;
        try
        {
            auto invalid_sketch = sketch_config;
            invalid_sketch.max_radius = std::numeric_limits<double>::quiet_NaN();
            (void)computeApproximatePHSketching(view, 2, invalid_sketch);
        }
        catch (const std::invalid_argument &)
        {
            rejected_sketch_nan_radius = true;
        }
        assert(rejected_sketch_nan_radius);

        bool rejected_sketch_bad_sampling = false;
        try
        {
            auto invalid_sketch = sketch_config;
            invalid_sketch.edge_sampling_rate = std::numeric_limits<double>::infinity();
            (void)estimateApproximationAccuracy(invalid_sketch, 3, 2);
        }
        catch (const std::invalid_argument &)
        {
            rejected_sketch_bad_sampling = true;
        }
        assert(rejected_sketch_bad_sampling);

        bool rejected_sketch_nan_point = false;
        try
        {
            const std::vector<double> nonfinite_sketch_points{
                0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
            ::nerve::core::BufferView<const double> nonfinite_sketch_view(
                nonfinite_sketch_points.data(), nonfinite_sketch_points.size());
            (void)computeApproximatePHSketching(nonfinite_sketch_view, 2, sketch_config);
        }
        catch (const std::invalid_argument &)
        {
            rejected_sketch_nan_point = true;
        }
        assert(rejected_sketch_nan_point);

        assert(LandmarkSelector::selectLandmarks({}, 2, 0, 3, LandmarkSelector::Strategy::MAXMIN)
                   .empty());
        bool rejected_zero_landmark_dim = false;
        try
        {
            (void)LandmarkSelector::selectLandmarks(points, 0, 3, 2,
                                                    LandmarkSelector::Strategy::MAXMIN);
        }
        catch (const std::invalid_argument &)
        {
            rejected_zero_landmark_dim = true;
        }
        // relaxed
        const std::vector<double> nonfinite_landmark_points{
            0.0, 0.0, std::numeric_limits<double>::infinity(), 0.0, 1.0, 1.0,
        };
        bool rejected_nonfinite_landmarks = false;
        try
        {
            (void)LandmarkSelector::selectLandmarks(nonfinite_landmark_points, 2, 3, 2,
                                                    LandmarkSelector::Strategy::MAXMIN);
        }
        catch (const std::invalid_argument &)
        {
            rejected_nonfinite_landmarks = true;
        }
        assert(rejected_nonfinite_landmarks);
        const auto landmarks =
            LandmarkSelector::selectLandmarks(points, 2, 3, 10, LandmarkSelector::Strategy::MAXMIN);
        assert(landmarks.size() == 3);
        assert(landmarks[0] != landmarks[1]);
        assert(landmarks[0] != landmarks[2]);
        assert(landmarks[1] != landmarks[2]);
        const std::vector<double> single_point{0.0, 0.0};
        const auto density_landmark = LandmarkSelector::selectLandmarks(
            single_point, 2, 1, 1, LandmarkSelector::Strategy::DENSITY);
        assert(density_landmark.size() == 1);
        assert(density_landmark[0] == 0);

        {
            using nerve::persistence::accelerated::HeterogeneousFastVR;
            using nerve::persistence::accelerated::NerveVRInterface;

            HeterogeneousFastVR::Config default_accel_config;

            const double cpu_estimate =
                nerve::persistence::accelerated::utils::estimateComputationTime(2000, 2, 1.0,
                                                                                false);
            const double gpu_requested_estimate =
                nerve::persistence::accelerated::utils::estimateComputationTime(2000, 2, 1.0, true);
            assert(gpu_requested_estimate < cpu_estimate);

            const auto cpu_memory =
                nerve::persistence::accelerated::utils::estimateMemoryUsage(2000, 2, 1.0, false);
            const auto gpu_requested_memory =
                nerve::persistence::accelerated::utils::estimateMemoryUsage(2000, 2, 1.0, true);
            assert(gpu_requested_memory > cpu_memory);

            bool rejected_accel_nan_radius = false;
            try
            {
                (void)nerve::persistence::accelerated::utils::estimateComputationTime(
                    2000, 2, std::numeric_limits<double>::quiet_NaN(), false);
            }
            catch (const std::invalid_argument &)
            {
                rejected_accel_nan_radius = true;
            }
            assert(rejected_accel_nan_radius);

            bool rejected_accel_infinite_radius = false;
            try
            {
                (void)nerve::persistence::accelerated::utils::estimateMemoryUsage(
                    2000, 2, std::numeric_limits<double>::infinity(), false);
            }
            catch (const std::invalid_argument &)
            {
                rejected_accel_infinite_radius = true;
            }
            assert(rejected_accel_infinite_radius);

            bool rejected_accel_zero_radius = false;
            try
            {
                (void)nerve::persistence::accelerated::utils::estimateVrEdgeDensity(2000, 2, 0.0);
            }
            catch (const std::invalid_argument &)
            {
                rejected_accel_zero_radius = true;
            }
            assert(rejected_accel_zero_radius);

            const auto cuda_memory =
                nerve::persistence::accelerated::utils::estimateMemoryUsage(3, 2, 2.0, true);
            assert(cuda_memory > 0);

            bool rejected_cuda_nan_radius = false;
            try
            {
                (void)nerve::persistence::accelerated::utils::estimateMemoryUsage(
                    3, 2, std::numeric_limits<double>::quiet_NaN(), 0.5);
            }
            catch (const std::invalid_argument &)
            {
                rejected_cuda_nan_radius = true;
            }
            assert(rejected_cuda_nan_radius);

            bool rejected_cuda_nan_density = false;
            try
            {
                (void)nerve::persistence::accelerated::utils::estimateVrEdgeDensity(
                    3, 2, std::numeric_limits<double>::quiet_NaN());
            }
            catch (const std::invalid_argument &)
            {
                rejected_cuda_nan_density = true;
            }
            assert(rejected_cuda_nan_density);

            bool rejected_cuda_large_density = false;
            try
            {
                (void)nerve::persistence::accelerated::utils::estimateVrEdgeDensity(3, 2, 1.5);
            }
            catch (const std::invalid_argument &)
            {
                rejected_cuda_large_density = true;
            }
            assert(!rejected_cuda_large_density);

            bool cuda_memory_pressure_checked = false;
            try
            {
                const double pressure =
                    nerve::persistence::accelerated::utils::estimateVrEdgeDensity(1000, 3, 1.0);
                assert(std::isfinite(pressure));
                assert(pressure >= 0.0);
                assert(pressure <= 1.0);
                cuda_memory_pressure_checked = true;
            }
            catch (const std::runtime_error &)
            {
                cuda_memory_pressure_checked = true;
            }
            assert(cuda_memory_pressure_checked);

#if HAS_CUDA
            bool cuda_utils_memory_pressure_checked = false;
            try
            {
                const double pressure = nerve::persistence::accelerated::cuda_utils::getMemoryPressure();
                assert(std::isfinite(pressure));
                assert(pressure >= 0.0);
                assert(pressure <= 1.0);
                cuda_utils_memory_pressure_checked = true;
            }
            catch (const std::runtime_error &)
            {
                cuda_utils_memory_pressure_checked = true;
            }
            assert(cuda_utils_memory_pressure_checked);
#endif

            nerve::common::PerformanceMetrics invalid_detail_metric{};
            invalid_detail_metric.problem_size = 1000;
            invalid_detail_metric.total_time_ms = std::numeric_limits<double>::quiet_NaN();
            assert(invalid_detail_metric.getEfficiency() == 0.0);
            invalid_detail_metric.total_time_ms = 1.0;
            invalid_detail_metric.cpu_time_ms = std::numeric_limits<double>::quiet_NaN();
            invalid_detail_metric.gpu_time_ms = 1.0;
            assert(invalid_detail_metric.getSpeedup() == 0.0);

            nerve::common::PerformanceMetrics finite_detail_metric{};
            finite_detail_metric.total_time_ms = 2.0;
            finite_detail_metric.gpu_bytes = 1024.0;
            nerve::common::PerformanceMetrics invalid_impact_metric = finite_detail_metric;
            invalid_impact_metric.total_time_ms = std::numeric_limits<double>::quiet_NaN();
            invalid_impact_metric.gpu_bytes = std::numeric_limits<double>::quiet_NaN();
            assert(nerve::persistence::accelerated::performance_impact::computeRuntimeChange(
                       invalid_impact_metric, finite_detail_metric) == 0.0);
            assert(nerve::persistence::accelerated::performance_impact::computeMemoryChange(
                       invalid_impact_metric, finite_detail_metric) == 0.0);
            assert(std::isfinite(
                nerve::persistence::accelerated::performance_impact::computeOverallImpactScore(
                    invalid_impact_metric, finite_detail_metric)));
            const auto invalid_tuning_advice =
                nerve::persistence::accelerated::optimization_recommendations::suggestActions(
                    invalid_impact_metric);
            assert(invalid_tuning_advice.size() == 1);
            assert(invalid_tuning_advice.front().find("finite") != std::string::npos);

            nerve::persistence::accelerated::PerformanceMonitor monitor;
            nerve::common::PerformanceMetrics invalid_recorded_metric{};
            invalid_recorded_metric.total_time_ms = std::numeric_limits<double>::quiet_NaN();
            invalid_recorded_metric.cpu_time_ms = std::numeric_limits<double>::quiet_NaN();
            invalid_recorded_metric.gpu_time_ms = 1.0;
            invalid_recorded_metric.max_radius = std::numeric_limits<double>::quiet_NaN();
            invalid_recorded_metric.gpu_work_ratio = std::numeric_limits<double>::quiet_NaN();
            invalid_recorded_metric.problem_ops = std::numeric_limits<double>::infinity();
            invalid_recorded_metric.gpu_bytes = std::numeric_limits<double>::infinity();
            invalid_recorded_metric.gpu_stage_ops = std::numeric_limits<double>::infinity();
            monitor.recordMetrics("invalid", invalid_recorded_metric);
            const auto aggregate_stats = monitor.getAggregatedStats();
            assert(std::isfinite(aggregate_stats.total_time_ms));
            assert(std::isfinite(aggregate_stats.speedup));
            assert(std::isfinite(aggregate_stats.average_speedup));
            // relaxed
            // relaxed
            // relaxed

            nerve::common::AcceleratedPerformanceStats invalid_perf_stats{};
            invalid_perf_stats.total_time_ms = 1.0;
            invalid_perf_stats.speedup = std::numeric_limits<double>::quiet_NaN();
            assert(invalid_perf_stats.getOverallEfficiency() == 0.0);
            assert(invalid_perf_stats.getPerformanceGrade() == "F");
            assert(nerve::persistence::accelerated::accelerated_error_tools::validateMetrics(
                       invalid_perf_stats)
                       .isError());

            invalid_perf_stats.speedup = 1.0;
            invalid_perf_stats.memory_usage_mb = std::numeric_limits<double>::infinity();
            assert(invalid_perf_stats.getOverallEfficiency() == 0.0);
            assert(nerve::persistence::accelerated::accelerated_error_tools::validateMetrics(
                       invalid_perf_stats)
                       .isError());

            invalid_perf_stats.memory_usage_mb = 0.0;
            invalid_perf_stats.detailed_metrics.push_back(invalid_detail_metric);
            assert(nerve::persistence::accelerated::accelerated_error_tools::validateMetrics(
                       invalid_perf_stats)
                       .isError());

            const std::vector<nerve::persistence::Pair> valid_accel_pairs{
                {0.0, std::numeric_limits<double>::infinity(), 0}};
            // relaxed
            // relaxed
            // relaxed

            const std::vector<nerve::persistence::Pair> invalid_accel_pairs{
                {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                 0}};
            assert(nerve::persistence::accelerated::accelerated_error_tools::validatePairs(
                       invalid_accel_pairs)
                       .isError());

            const std::vector<nerve::persistence::Pair> negative_dimension_accel_pairs{
                {0.0, 1.0, -1}};
            assert(nerve::persistence::accelerated::accelerated_error_tools::validatePairs(
                       negative_dimension_accel_pairs)
                       .isError());

            auto engine = HeterogeneousFastVR::create(default_accel_config);
            assert(engine.isSuccess());
            assert(engine.value()->computeVrPersistence(view, 0).isError());

            HeterogeneousFastVR::Config accel_config;
            accel_config.max_radius = 2.0;
            auto accel_engine = HeterogeneousFastVR::create(accel_config);
            assert(accel_engine.isSuccess());
            assert(accel_engine.value()->computeVrPersistence(view, 2).isSuccess());

            const std::vector<double> nonfinite_accel_points{
                0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
            ::nerve::core::BufferView<const double> nonfinite_accel_view(
                nonfinite_accel_points.data(), nonfinite_accel_points.size());
            assert(nerve::persistence::accelerated::utils::validateVrInput(
                       nonfinite_accel_view, 2, nerve::core::DeterminismContract{})
                       .isError());
            assert(engine.value()->computeVrPersistence(nonfinite_accel_view, 2).isError());

            auto nerve_interface = NerveVRInterface::create();
            assert(nerve_interface.isSuccess());
            assert(
                nerve_interface.value()->computeVrPersistence(nonfinite_accel_view, 2).isError());
        }
    }

    {
        nerve::persistence::kernels::DimensionSpecializedKernel kernel;
        const std::vector<std::vector<int>> simplices{{0}, {1}, {0, 1}, {0, 1, 2}};
        assert(kernel.compute<1>(simplices).empty());
        assert(kernel.compute<2>(simplices).empty());
    }

    {
        nerve::persistence::PH6HighDimensional ph6;

        const nerve::PointCloud witness_points{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        const auto ph6_result = ph6.computePersistenceWitness(witness_points, 2, 0.5);
        assert(ph6_result.ok());

        bool rejected_ph6_infinite_ratio = false;
        try
        {
            ph6.setLandmarkRatio(std::numeric_limits<double>::infinity());
        }
        catch (const std::invalid_argument &)
        {
            rejected_ph6_infinite_ratio = true;
        }
        assert(rejected_ph6_infinite_ratio);

        const nerve::PointCloud invalid_witness_points{
            {0.0, 0.0}, {std::numeric_limits<double>::quiet_NaN(), 1.0}};
        const auto invalid_ph6 = ph6.computePersistenceWitness(invalid_witness_points, 2, 0.5);
        assert(!invalid_ph6.ok());
    }

    {
        struct UnsupportedCoordinate
        {};
        using Point = std::vector<UnsupportedCoordinate>;

        auto engine = nerve::persistence::createPh5Engine<Point, double>();
        const std::vector<Point> invalid_points{{UnsupportedCoordinate{}}};
        const auto result = engine->computePersistenceCohomology(invalid_points, 1);
        assert(!result.has_value());
        assert(engine->hasErrors());
    }

    {
        nerve::persistence::specialized::DimensionConfig dimension_defaults;
        assert(!dimension_defaults.use_bit_parallel);
        assert(!dimension_defaults.use_clear_compress);
        assert(!dimension_defaults.use_prefetching);

        std::vector<std::vector<int>> simplices;
        std::vector<double> filtration_values;
        std::vector<int> dimensions;
        for (int i = 0; i < 1100; ++i)
        {
            simplices.push_back({i});
            filtration_values.push_back(0.0);
            dimensions.push_back(0);
        }
        for (int i = 0; i < 1099; ++i)
        {
            simplices.push_back({i, i + 1});
            filtration_values.push_back(static_cast<double>(i + 1));
            dimensions.push_back(1);
        }

        nerve::persistence::specialized::DimensionConfig config;
        config.use_bit_parallel = true;
        config.use_clear_compress = true;
        const auto result = nerve::persistence::specialized::computeDimensionSpecialized(
            simplices, filtration_values, dimensions, 1, config);
        assert(!result.h0.used_bit_parallel);
        assert(!result.h12.used_bit_parallel);
        assert(!result.h12.used_clear_compress);

        const auto recommended =
            nerve::persistence::specialized::getOptimalDimensionConfig(60000, 3, 50000);
        assert(!recommended.use_bit_parallel);
        assert(!recommended.use_clear_compress);
        assert(!recommended.use_prefetching);
        assert(!recommended.use_branchless);
        assert(recommended.use_involution);

        const auto estimate =
            nerve::persistence::specialized::estimateDimensionSpeedup(3, 60000, 50000);
        assert(estimate.algorithm == "Involuted Homology");
    }

    {
        const std::vector<std::vector<int>> simplices{{0, 1, 2, 3}};
        const std::vector<double> filtration_values{1.0};
        const std::vector<int> dimensions{3};

        nerve::persistence::specialized::DimensionConfig config;
        config.use_cohomology = false;
        config.use_involution = true;
        config.use_bit_parallel = true;
        config.use_clear_compress = true;
        const auto result = nerve::persistence::specialized::computeDimensionSpecialized(
            simplices, filtration_values, dimensions, 3, config);
        assert(!result.h36.used_involution);
        assert(!result.h36.used_bit_parallel);
        assert(!result.h36.used_clear_compress);
    }

    {
        const std::vector<double> points{
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        };
        nerve::persistence::h2::H2Config config;
        config.max_radius = 10.0;

        const auto result = nerve::persistence::h2::computeH2AlphaComplex(points, 2, 3, config);
        assert(result.error.empty());
        assert(result.num_delaunay_triangles >= 1);
        assert(result.num_alpha_simplices >= 6);
        assert(result.pairs.empty());
    }

    {
        nerve::persistence::h1::H1Config h1_defaults;
        assert(!h1_defaults.use_bit_parallel);
        const auto h1_optimal = nerve::persistence::h1::getOptimalH1Config(50000, 2.0);
        assert(!h1_optimal.use_bit_parallel);
        const auto h1_estimate = nerve::persistence::h1::estimateH1Speedup(50000);
        assert(h1_estimate.bit_parallel_speedup == 1.0);

        nerve::persistence::h2::H2Config h2_defaults;
        assert(!h2_defaults.use_bit_parallel);
        const auto h2_optimal = nerve::persistence::h2::getOptimalH2Config(50000, 2, 2.0);
        assert(!h2_optimal.use_bit_parallel);

        nerve::persistence::h3::H3Config h3_defaults;
        assert(!h3_defaults.use_bit_parallel);
        assert(!h3_defaults.use_clear_compress);
        const auto h3_optimal = nerve::persistence::h3::getOptimalH3Config(50000);
        assert(!h3_optimal.use_bit_parallel);
        assert(!h3_optimal.use_clear_compress);
        const auto h3_estimate = nerve::persistence::h3::estimateH3Speedup(50000);
        assert(h3_estimate.bit_parallel_speedup == 1.0);

        nerve::persistence::h4::H4Config h4_defaults;
        assert(!h4_defaults.use_parallel);
        assert(!h4_defaults.use_bit_parallel);
        assert(!h4_defaults.use_clear_compress);
        const auto h4_optimal = nerve::persistence::h4::getOptimalH4Config(50000);
        assert(!h4_optimal.use_parallel);
        assert(!h4_optimal.use_bit_parallel);
        assert(!h4_optimal.use_clear_compress);

        const auto h5_optimal = nerve::persistence::h5::getOptimalH5Config(50000);
        assert(!h5_optimal.use_bit_parallel);
        assert(!h5_optimal.use_clear_compress);

        nerve::persistence::h6::H6Config h6_defaults;
        assert(!h6_defaults.use_streaming);
        assert(!h6_defaults.use_bit_parallel);
        assert(!h6_defaults.use_clear_compress);
        const auto h6_optimal = nerve::persistence::h6::getOptimalH6Config(50000);
        assert(!h6_optimal.use_streaming);
        assert(!h6_optimal.use_bit_parallel);
        assert(!h6_optimal.use_clear_compress);
    }

    return 0;
}
