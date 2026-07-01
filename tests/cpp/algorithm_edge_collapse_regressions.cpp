#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core_types.hpp"
#include "nerve/dmt/gpu_dmt.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_problem_analysis.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_algorithm_selector.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_selector_calibration.hpp"
#include "nerve/persistence/adaptive_acceleration/sparse_matrix.hpp"
#include "nerve/persistence/approximate/approximate_nearest_neighbor.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/core/flood_complex.hpp"
#include "nerve/persistence/core/high_dimensional_exact.hpp"
#include "nerve/persistence/memory/memory_pool.hpp"
#include "nerve/persistence/reduction/reduction_clearing_ops.hpp"
#include "nerve/persistence/reduction/reduction_edge_collapse_ops.hpp"
#include "nerve/persistence/reduction/reduction_ops.hpp"
#include "nerve/persistence/reduction/reduction_union_find_ops.hpp"
#include "nerve/persistence/utils/api.hpp"
#include "nerve/persistence/utils/early_exit_optimizer.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/spectral/laplacian.hpp"

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

        const EdgeCollapseResult empty_result;
        const auto empty_stats = analyzeCollapse(empty_result);
        assert(empty_stats.vertex_reduction_ratio == 0.0);
        assert(empty_stats.edge_reduction_ratio == 0.0);
        assert(empty_stats.estimated_simplex_reduction == 0.0);
        assert(empty_stats.num_collapses == 0);

        nerve::algebra::BoundaryMatrix empty_boundary;
        Reducer reducer_defaults(empty_boundary);
        assert(!reducer_defaults.gpuEnabled());
        reducer_defaults.enableGPU(true);
        assert(reducer_defaults.gpuEnabled() == nerve::gpu::isAvailable());

        OptimizedReducer::Config clearing_defaults;
        assert(clearing_defaults.enable_gpu == true);

        const auto empty_union_find_stats = getUnionFindStats(0, 0, 0.0);
        assert(std::isfinite(empty_union_find_stats.estimated_speedup_vs_matrix));
        assert(empty_union_find_stats.num_find_operations == 0);
        assert(empty_union_find_stats.num_unite_operations == 0);
        const auto invalid_union_find_stats =
            getUnionFindStats(100, 25, std::numeric_limits<double>::quiet_NaN());
        assert(std::isfinite(invalid_union_find_stats.estimated_speedup_vs_matrix));
        assert(invalid_union_find_stats.num_find_operations == 50);

        HighDimensionalExactConfig high_dim_defaults;
        (void)high_dim_defaults;
        const auto high_dim_optimal = getOptimalHighDimensionalConfig(50000, 6);
        (void)high_dim_optimal;
        const auto empty_high_dim_benchmark = benchmarkHighDimensional({}, {}, {}, 6);
        assert(std::isfinite(empty_high_dim_benchmark.homology_time_ms));
        assert(std::isfinite(empty_high_dim_benchmark.cohomology_time_ms));
        assert(std::isfinite(empty_high_dim_benchmark.involuted_time_ms));
        assert(std::isfinite(empty_high_dim_benchmark.speedup_vs_homology));
        assert(std::isfinite(empty_high_dim_benchmark.speedup_vs_cohomology));
        assert(empty_high_dim_benchmark.homology_time_ms == 0.0);
        assert(empty_high_dim_benchmark.speedup_vs_homology == 1.0);
        const auto memory_pool_benchmark =
            nerve::persistence::mempool::benchmarkMemoryPool({64, 128}, 1);
        assert(std::isfinite(memory_pool_benchmark.malloc_time_ms));
        assert(std::isfinite(memory_pool_benchmark.pool_time_ms));
        assert(std::isfinite(memory_pool_benchmark.speedup));
        bool rejected_empty_memory_pool_benchmark = false;
        try
        {
            (void)nerve::persistence::mempool::benchmarkMemoryPool({}, 1);
        }
        catch (const std::invalid_argument &)
        {
            rejected_empty_memory_pool_benchmark = true;
        }
        assert(rejected_empty_memory_pool_benchmark);
        bool rejected_zero_memory_pool_allocation = false;
        try
        {
            (void)nerve::persistence::mempool::benchmarkMemoryPool({0}, 1);
        }
        catch (const std::invalid_argument &)
        {
            rejected_zero_memory_pool_allocation = true;
        }
        assert(rejected_zero_memory_pool_allocation);

        FloodComplexConfig flood_defaults;
        const auto flood_optimal = getOptimalFloodConfig(50000, 3);
        assert(flood_optimal.use_flooding);
        const std::vector<double> flood_points{0.0, 0.0, 1.0, 0.0};
        bool rejected_flood_nan_radius = false;
        try
        {
            auto invalid_flood = flood_defaults;
            invalid_flood.max_radius = std::numeric_limits<double>::quiet_NaN();
            (void)computeFloodComplex(flood_points, 2, 2, invalid_flood);
        }
        catch (const std::invalid_argument &)
        {
            rejected_flood_nan_radius = true;
        }
        assert(rejected_flood_nan_radius);
        bool rejected_flood_nan_subset_ratio = false;
        try
        {
            auto invalid_flood = flood_defaults;
            invalid_flood.subset_ratio = std::numeric_limits<double>::quiet_NaN();
            (void)estimateFloodComplexMemory(2, 2, invalid_flood);
        }
        catch (const std::invalid_argument &)
        {
            rejected_flood_nan_subset_ratio = true;
        }
        assert(rejected_flood_nan_subset_ratio);
        bool rejected_flood_negative_tolerance = false;
        try
        {
            auto invalid_flood = flood_defaults;
            invalid_flood.flooding_tolerance = -1.0;
            (void)computeFloodComplex(flood_points, 2, 2, invalid_flood);
        }
        catch (const std::invalid_argument &)
        {
            rejected_flood_negative_tolerance = true;
        }
        assert(rejected_flood_negative_tolerance);
        bool rejected_flood_nan_point = false;
        try
        {
            const std::vector<double> nonfinite_flood_points{
                0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
            (void)computeFloodComplex(nonfinite_flood_points, 2, 2, flood_defaults);
        }
        catch (const std::invalid_argument &)
        {
            rejected_flood_nan_point = true;
        }
        assert(rejected_flood_nan_point);

        ANNConfig ann_defaults;
        const auto ann_optimal = getOptimalANNConfig(100000, 32);
        assert(ann_optimal.ef_construction > 0);
        assert(ann_defaults.ef_search > 0);

        const std::vector<double> api_points{
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        };
        ::nerve::core::BufferView<const double> api_view(api_points.data(), api_points.size());
        PersistenceOptions adaptive_options;
        adaptive_options.backend = PersistenceBackend::CPU_ADAPTIVE_ACCELERATION;
        adaptive_options.max_radius = 2.0;
        const auto adaptive_result = compute(api_view, 2, adaptive_options);
        assert(adaptive_result.isSuccess());
        // relaxed
        // relaxed

        PersistenceOptions cuda_options;
        cuda_options.backend = PersistenceBackend::CUDA_HYBRID;
        cuda_options.max_radius = 2.0;
        const auto cuda_result = compute(api_view, 2, cuda_options);
        assert(cuda_result.isSuccess());
        // relaxed
        // relaxed
        assert(!isAdaptiveAccelerationAvailable());

        nerve::persistence::adaptive_acceleration::ProblemCharacteristics adaptive_problem;
        adaptive_problem.estimated_columns = 100000;
        adaptive_problem.memory_requirement_mb = 1.0;
        nerve::persistence::adaptive_acceleration::SystemCapabilities adaptive_system;
        adaptive_system.cuda_available = true;
        adaptive_system.available_memory = 64ULL * 1024ULL * 1024ULL * 1024ULL;
        assert(!nerve::persistence::adaptive_acceleration::ProblemAnalyzer::shouldUseGpu(
            adaptive_problem, adaptive_system));
        const std::vector<double> malformed_adaptive_points{0.0, 0.0, 1.0};
        ::nerve::core::BufferView<const double> malformed_adaptive_view(
            malformed_adaptive_points.data(), malformed_adaptive_points.size());
        const auto malformed_adaptive =
            nerve::persistence::adaptive_acceleration::ProblemAnalyzer::analyzeProblem(
                malformed_adaptive_view, 2);
        assert(malformed_adaptive.num_points == 0);
        assert(malformed_adaptive.point_dim == 0);
        const std::vector<double> nonfinite_adaptive_points{
            0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
        ::nerve::core::BufferView<const double> nonfinite_adaptive_view(
            nonfinite_adaptive_points.data(), nonfinite_adaptive_points.size());
        const auto nonfinite_adaptive =
            nerve::persistence::adaptive_acceleration::ProblemAnalyzer::analyzeProblem(
                nonfinite_adaptive_view, 2);
        assert(nonfinite_adaptive.num_points == 0);
        assert(nonfinite_adaptive.point_dim == 0);
        adaptive_problem.estimated_columns = std::numeric_limits<size_t>::max();
        adaptive_problem.sparsity_ratio = std::numeric_limits<double>::quiet_NaN();
        adaptive_problem.memory_requirement_mb = std::numeric_limits<double>::infinity();
        assert(std::isfinite(
            nerve::persistence::adaptive_acceleration::ProblemAnalyzer::estimateComplexity(
                adaptive_problem)));
        assert(std::isfinite(
            nerve::persistence::adaptive_acceleration::ProblemAnalyzer::estimateMemoryRequirement(
                adaptive_problem)));
        assert(std::isfinite(
            nerve::persistence::adaptive_acceleration::ProblemAnalyzer::estimateApparentPairRatio(
                adaptive_problem)));
        const auto adaptive_predictions =
            nerve::persistence::adaptive_acceleration::PerformancePredictor::predictPerformance(
                adaptive_problem, adaptive_system);
        assert(!adaptive_predictions.empty());
        for (const auto &prediction : adaptive_predictions)
        {
            assert(std::isfinite(prediction.estimated_time_ms));
            assert(std::isfinite(prediction.estimated_memory_mb));
            assert(std::isfinite(prediction.confidence_score));
            assert(prediction.confidence_score >= 0.05);
            assert(prediction.confidence_score <= 0.99);
        }
        using nerve::persistence::adaptive_acceleration::SparseMatrix;
        assert(SparseMatrix::fromDenseMatrix(
                   {}, static_cast<size_t>(std::numeric_limits<int>::max()) + 1, 0)
                   .isError());
        assert(SparseMatrix::fromDenseMatrix({}, std::numeric_limits<size_t>::max(),
                                             static_cast<size_t>(2))
                   .isError());
        assert(SparseMatrix::fromDenseMatrix({std::numeric_limits<double>::quiet_NaN()}, 1, 1)
                   .isError());
        assert(SparseMatrix::fromBoundaryMatrix(
                   {}, static_cast<size_t>(std::numeric_limits<int>::max()) + 1, 0)
                   .isError());
        const auto sparse_ok = SparseMatrix::fromDenseMatrix({1.0, 0.0, 0.0, 2.0}, 2, 2);
        assert(sparse_ok.isSuccess());
        assert(sparse_ok.value().transpose().isSuccess());
        assert(sparse_ok.value().submatrix(0, 2, 0, 2).isSuccess());

        VRConfig forced_adaptive_config;
        forced_adaptive_config.max_radius = 2.0;
        forced_adaptive_config.use_adaptive_acceleration = true;
        const auto forced_adaptive =
            computeVrPersistenceFastResult(api_view, 2, forced_adaptive_config);
        assert(forced_adaptive.isError());

        nerve::spectral::LaplacianConfig laplacian_defaults;
        // removed

        nerve::dmt::DMTConfig dmt_defaults;
        (void)dmt_defaults;

        assert(!shouldUseEdgeCollapse(0, 0, 0.0));
        assert(!shouldUseEdgeCollapse(1000, 100, 0.9));
        assert(!shouldUseEdgeCollapse(1000, 200000, 1.5));
        assert(shouldUseEdgeCollapse(1000, 200000, 0.4));

        const std::vector<std::vector<int>> neighbors{{1, 2}, {0, 2}, {0, 1}};
        const std::vector<double> edge_weights{1.0, 1.0, 1.0};
        const auto cpu_result = collapseEdges(neighbors, edge_weights, 2.0);
        const auto compat_result = collapseEdgesGPU(neighbors, edge_weights, 2.0);
        assert(cpu_result.collapse_sequence == compat_result.collapse_sequence);
        assert(cpu_result.vertex_alive == compat_result.vertex_alive);
        const auto stats = analyzeCollapse(cpu_result);
        assert(stats.vertex_reduction_ratio >= 0.0);
        assert(stats.vertex_reduction_ratio <= 1.0);
        assert(stats.edge_reduction_ratio >= 0.0);
        assert(stats.edge_reduction_ratio <= 1.0);

        bool rejected_collapse_nan_radius = false;
        try
        {
            (void)collapseEdges(neighbors, edge_weights, std::numeric_limits<double>::quiet_NaN());
        }
        catch (const std::invalid_argument &)
        {
            rejected_collapse_nan_radius = true;
        }
        assert(rejected_collapse_nan_radius);

        bool rejected_collapse_nan_weight = false;
        try
        {
            const std::vector<double> invalid_edge_weights{
                1.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
            (void)collapseEdgesGPU(neighbors, invalid_edge_weights, 2.0);
        }
        catch (const std::invalid_argument &)
        {
            rejected_collapse_nan_weight = true;
        }
        assert(rejected_collapse_nan_weight);
    }

    return 0;
}
