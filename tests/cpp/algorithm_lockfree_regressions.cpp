#include "nerve/core_types.hpp"
#include "nerve/persistence/reduction/reduction_lock_free_structures.hpp"
#include "nerve/persistence/reduction/reduction_lockfree_ops.hpp"
#include "nerve/persistence/streaming/tile_streaming_ph.hpp"
#include "nerve/streaming/lock_free_streaming.hpp"

#include <algorithm>
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
        using namespace nerve::persistence::lockfree;

        LockFreePivotTable table(1024);
        for (int i = 0; i < 300; ++i)
        {
            assert(table.tryInsert(i * 1024, i));
        }
        assert(table.size() == 300);
        for (int i = 0; i < 300; ++i)
        {
            assert(table.find(i * 1024) == i);
        }
        assert(!table.tryInsert(0, 999));

        bool rejected_negative_pivot = false;
        try
        {
            (void)table.tryInsert(-1, 0);
        }
        catch (const std::invalid_argument &)
        {
            rejected_negative_pivot = true;
        }
        assert(rejected_negative_pivot);

        LockFreeWorkQueue queue(2);
        int executed = 0;
        queue.push([&executed]() { executed += 1; });
        queue.push([&executed]() { executed += 2; });
        bool rejected_full_queue = false;
        try
        {
            queue.push([]() {});
        }
        catch (const std::runtime_error &)
        {
            rejected_full_queue = true;
        }
        assert(rejected_full_queue);
        auto owned_task = queue.pop();
        assert(owned_task.has_value());
        (*owned_task)();
        auto stolen_task = queue.steal();
        assert(stolen_task.has_value());
        (*stolen_task)();
        assert(executed == 3);
        assert(queue.pop() == std::nullopt);
        assert(queue.isEmpty());

        LockFreePivotAnnounce announce(1);
        announce.announce(0, 7);
        assert(announce.isBeingWorkedOn(7));
        announce.clear(0);
        assert(!announce.isBeingWorkedOn(7));
        bool rejected_bad_announce = false;
        try
        {
            announce.announce(1, 7);
        }
        catch (const std::out_of_range &)
        {
            rejected_bad_announce = true;
        }
        assert(rejected_bad_announce);

        LockFreeReductionCoordinator coordinator(2);
        assert(coordinator.claimColumn() == 0);
        assert(coordinator.claimColumn() == 1);
        assert(coordinator.claimColumn() == -1);
        coordinator.markReduced(0);
        coordinator.markReduced(0);
        assert(coordinator.numReduced() == 1);
        bool rejected_bad_reduced_column = false;
        try
        {
            coordinator.markReduced(2);
        }
        catch (const std::out_of_range &)
        {
            rejected_bad_reduced_column = true;
        }
        assert(rejected_bad_reduced_column);
        bool rejected_oversize_coordinator = false;
        try
        {
            LockFreeReductionCoordinator too_large(
                static_cast<size_t>(std::numeric_limits<int>::max()) + 1);
            (void)too_large;
        }
        catch (const std::overflow_error &)
        {
            rejected_oversize_coordinator = true;
        }
        assert(rejected_oversize_coordinator);

        bool rejected_empty_benchmark = false;
        try
        {
            (void)benchmarkLockFree(1, 1, 0);
        }
        catch (const std::invalid_argument &)
        {
            rejected_empty_benchmark = true;
        }
        assert(rejected_empty_benchmark);
        const auto config = getOptimalLockFreeConfig(-4);
        assert(config.max_steal_attempts >= 0);
    }

    {
        constexpr int kColumns = 3000;
        std::vector<std::vector<int>> boundary(static_cast<std::size_t>(kColumns));
        for (int i = 1; i < kColumns; ++i)
        {
            boundary[static_cast<std::size_t>(i)].push_back(i - 1);
        }
        std::vector<double> filtration(static_cast<std::size_t>(kColumns), 0.0);

        const auto pairs = nerve::persistence::reduceMatrixLockfree(
            boundary, filtration, std::vector<nerve::Dimension>(kColumns, 1), 4);
        assert(pairs.size() <= static_cast<std::size_t>(kColumns));
    }

    {
        using namespace nerve::streaming::lockfree;

        WaitFreeRingBuffer<int> ring(1);
        assert(ring.capacity() == 1);
        assert(ring.tryPush(11));
        assert(!ring.tryPush(22));
        const auto first = ring.tryPop();
        assert(first.has_value());
        assert(*first == 11);
        assert(!ring.tryPop().has_value());

        LockFreeMPMCQueue<int> queue(2);
        assert(queue.tryEnqueue(3));
        assert(queue.tryEnqueue(4));
        assert(!queue.tryEnqueue(5));
        const auto dequeued = queue.tryDequeue();
        assert(dequeued.has_value());
        assert(*dequeued == 3);
        const auto second_dequeued = queue.tryDequeue();
        assert(second_dequeued.has_value());
        assert(*second_dequeued == 4);
        assert(!queue.tryDequeue().has_value());

        LockFreeStreamingWindow<int> window(0);
        window.addPoint(1);
        window.addPoint(2);
        const auto snapshot = window.getSnapshot();
        assert(snapshot.size() == 1);
        assert(snapshot[0] == 2);

        bool rejected_huge_spsc = false;
        try
        {
            WaitFreeRingBuffer<int> huge(std::numeric_limits<std::size_t>::max());
            (void)huge;
        }
        catch (const std::length_error &)
        {
            rejected_huge_spsc = true;
        }
        assert(rejected_huge_spsc);

        bool rejected_huge_mpmc = false;
        try
        {
            LockFreeMPMCQueue<int> huge(std::numeric_limits<std::size_t>::max());
            (void)huge;
        }
        catch (const std::length_error &)
        {
            rejected_huge_mpmc = true;
        }
        assert(rejected_huge_mpmc);
    }

    {
        nerve::persistence::StreamingConfig config;
        config.tile_size = 1;
        config.max_dim = 1;
        config.max_radius = 2.0;
        const std::vector<double> points{
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        };
        const auto result = nerve::persistence::computeTiledPH(points, 2, config);
        assert(result.total_points == 3);
        assert(result.point_dim == 2);
        assert(result.num_tiles == 3);

        bool rejected_nan_overlap_ratio = false;
        try
        {
            auto invalid_config = config;
            invalid_config.overlap_ratio = std::numeric_limits<double>::quiet_NaN();
            (void)nerve::persistence::computeTiledPH(points, 2, invalid_config);
        }
        catch (const std::invalid_argument &)
        {
            rejected_nan_overlap_ratio = true;
        }
        assert(rejected_nan_overlap_ratio);

        bool rejected_large_overlap_ratio = false;
        try
        {
            auto invalid_config = config;
            invalid_config.overlap_ratio = 0.75;
            (void)nerve::persistence::computeTiledPH(points, 2, invalid_config);
        }
        catch (const std::invalid_argument &)
        {
            rejected_large_overlap_ratio = true;
        }
        assert(rejected_large_overlap_ratio);

        bool rejected_nan_merge_tolerance = false;
        try
        {
            auto invalid_config = config;
            invalid_config.merge_tolerance = std::numeric_limits<double>::quiet_NaN();
            (void)nerve::persistence::computeTiledPH(points, 2, invalid_config);
        }
        catch (const std::invalid_argument &)
        {
            rejected_nan_merge_tolerance = true;
        }
        assert(rejected_nan_merge_tolerance);

        bool rejected_streaming_nan_radius = false;
        try
        {
            auto invalid_config = config;
            invalid_config.max_radius = std::numeric_limits<double>::quiet_NaN();
            (void)nerve::persistence::computeTiledPH(points, 2, invalid_config);
        }
        catch (const std::invalid_argument &)
        {
            rejected_streaming_nan_radius = true;
        }
        assert(rejected_streaming_nan_radius);

        bool rejected_huge_point_dim = false;
        try
        {
            (void)nerve::persistence::getOptimalStreamingConfig(
                1, std::numeric_limits<std::size_t>::max() / sizeof(double) + 1, 1);
        }
        catch (const std::overflow_error &)
        {
            rejected_huge_point_dim = true;
        }
        assert(rejected_huge_point_dim);
    }

#if defined(NERVE_HAS_CUDA)
    {
        nerve::gpu::Config config;
        config.enable_gpu = false;
        auto &manager = nerve::gpu::ComputeManager::getInstance();
        manager.initialize(config);

        double tile_point = 0.0;
        double tile_distance = 0.0;
        nerve::gpu::advanced::TileKernelConfig tile_config;
        assert(nerve::gpu::advanced::launchTileDistanceMatrix(&tile_point, &tile_distance, 1, 0,
                                                              tile_config,
                                                              nullptr) == cudaErrorInvalidValue);
        nerve::gpu::kernels::KernelDispatcher kernel_dispatcher;
        assert(kernel_dispatcher.computeDistanceMatrix(&tile_point, &tile_distance, 1, 0, 0.0) ==
               cudaErrorInvalidValue);
        assert(kernel_dispatcher.computeDistanceMatrix(&tile_point, &tile_distance, 1, 1,
                                                       std::numeric_limits<double>::quiet_NaN()) ==
               cudaErrorInvalidValue);
        assert(kernel_dispatcher.computeDistanceMatrix(&tile_point, &tile_distance, 1, 1,
                                                       std::numeric_limits<double>::infinity()) ==
               cudaErrorInvalidValue);
#if !HAS_CUDA
        assert(nerve::gpu::advanced::launchTileDistanceMatrix(&tile_point, &tile_distance, 1, 1,
                                                              tile_config,
                                                              nullptr) == cudaErrorInvalidValue);
#endif

        const std::vector<std::vector<double>> points{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        const std::vector<std::vector<double>> invalid_gpu_manager_points{
            {0.0, 0.0}, {std::numeric_limits<double>::quiet_NaN(), 1.0}};
        std::vector<std::vector<double>> distances;
        assert(manager.computeDistanceMatrix(invalid_gpu_manager_points, distances).isError());
        assert(distances.empty());
        const std::vector<std::vector<double>> overflowing_gpu_manager_points{
            {0.0}, {std::numeric_limits<double>::max()}};
        assert(manager.computeDistanceMatrix(overflowing_gpu_manager_points, distances).isError());
        assert(distances.empty());

        manager.clearPerformanceHistory();
        manager.clearStats();
        nerve::gpu::PerformanceProfile invalid_profile;
        invalid_profile.operation_name = "invalid-profile";
        invalid_profile.cpu_time_ms = std::numeric_limits<double>::infinity();
        invalid_profile.gpu_time_ms = std::numeric_limits<double>::quiet_NaN();
        invalid_profile.speedup = std::numeric_limits<double>::infinity();
        manager.recordPerformance(invalid_profile);
        const auto history_after_invalid_profile = manager.getPerformanceHistory();
        assert(history_after_invalid_profile.size() == 1);
        assert(history_after_invalid_profile[0].cpu_time_ms == 0.0);
        assert(history_after_invalid_profile[0].gpu_time_ms == 0.0);
        assert(history_after_invalid_profile[0].speedup == 1.0);
        const auto stats_after_invalid_profile = manager.getStats();
        assert(stats_after_invalid_profile.total_operations == 1);
        assert(std::isfinite(stats_after_invalid_profile.average_speedup));
        assert(std::isfinite(stats_after_invalid_profile.total_time_saved_ms));

        std::vector<std::vector<double>> laplacian;
        assert(manager.constructLaplacian(points, {{0, 1}, {1, 2}}, laplacian).isSuccess());
        assert(laplacian.size() == 3);
        assert(laplacian[1][1] == 2.0);
        assert(laplacian[0][1] == -1.0);

        nerve::algebra::BoundaryMatrix empty_clearing_boundary;
        std::vector<int> empty_simplex_dimensions;
        std::vector<double> empty_filtration_values;
        nerve::gpu::ComputeManager::ClearingResult clearing_result;
        const auto invalid_clearing = manager.applyClearing(
            empty_clearing_boundary, empty_simplex_dimensions, empty_filtration_values, 0,
            std::numeric_limits<double>::infinity(), clearing_result);
        assert(invalid_clearing.isError());
        assert(invalid_clearing.errorCode() == nerve::errors::ErrorCode::E51_PH_INPUT);

        nerve::persistence::accelerated::EdgeExtractionConfig edge_config;
        edge_config.max_edges = 4;
        auto edge_extractor =
            nerve::persistence::accelerated::CUDAEgdeExtractor::create(edge_config);
        assert(edge_extractor.isSuccess());
        std::vector<double> invalid_distances{0.0, std::numeric_limits<double>::infinity(),
                                              std::numeric_limits<double>::infinity(), 0.0};
        std::vector<nerve::persistence::accelerated::Edge> extracted_edges(4);
        nerve::core::BufferView<const double> invalid_distance_view(invalid_distances.data(),
                                                                    invalid_distances.size());
        nerve::core::BufferView<nerve::persistence::accelerated::Edge> edge_view(
            extracted_edges.data(), extracted_edges.size());
        const auto invalid_edge_extraction = edge_extractor.value()->extractEdges(
            invalid_distance_view, std::move(edge_view), 2, 2.0);
        assert(invalid_edge_extraction.isError());
        assert(invalid_edge_extraction.errorCode() == nerve::errors::ErrorCode::E20_NUM_NAN);

        std::vector<nerve::gpu::ComputeManager::VRSimplex> vr_simplices;
        assert(manager.buildVRComplex(points, 1.5, 1, vr_simplices).isSuccess());
        assert(vr_simplices.size() >= points.size());

        std::vector<nerve::gpu::ComputeManager::CechSimplex> cech_simplices;
        const double large_cech_coord = 1.0e100;
        const std::vector<std::vector<double>> large_cech_points{
            {0.0, 0.0}, {large_cech_coord, 0.0}, {0.0, large_cech_coord}};
        assert(manager.buildCechComplex(large_cech_points, large_cech_coord, 2, cech_simplices)
                   .isSuccess());
        assert(std::ranges::all_of(cech_simplices, [](const auto &simplex) {
            return std::isfinite(simplex.filtration_value) && std::isfinite(simplex.alpha_value);
        }));
        assert(std::ranges::any_of(cech_simplices,
                                   [](const auto &simplex) { return simplex.dimension == 2; }));

        std::vector<std::pair<int, int>> assignment;
        const std::vector<std::vector<double>> costs{{4.0, 1.0}, {2.0, 3.0}};
        const auto assignment_cost = manager.solveAssignment(costs, assignment);
        assert(assignment_cost.isSuccess());
        assert(assignment_cost.value() == 3.0);
        assert(assignment.size() == 2);

        const auto bottleneck_cost = manager.solveBottleneck(costs, assignment);
        assert(bottleneck_cost.isSuccess());
        assert(bottleneck_cost.value() == 2.0);

        nerve::persistence::Diagram diagram_cost_a;
        diagram_cost_a.addPair({0.0, 1.0, 0});
        nerve::persistence::Diagram diagram_cost_b;
        diagram_cost_b.addPair({0.5, std::numeric_limits<double>::infinity(), 0});
        std::vector<std::vector<double>> diagram_cost_matrix;
        assert(manager.computeDiagramCostMatrix(diagram_cost_a, diagram_cost_b, diagram_cost_matrix)
                   .isSuccess());
        assert(!diagram_cost_matrix.empty());
        for (const auto &row : diagram_cost_matrix)
        {
            assert(std::ranges::all_of(row, [](double value) { return std::isfinite(value); }));
        }

        nerve::persistence::Diagram overflow_diagram_cost;
        overflow_diagram_cost.addPair(
            {-std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0});
        assert(manager
                   .computeDiagramCostMatrix(overflow_diagram_cost, diagram_cost_a,
                                             diagram_cost_matrix)
                   .isError());
        assert(diagram_cost_matrix.empty());

        nerve::persistence::Diagram negative_dimension_diagram_cost;
        negative_dimension_diagram_cost.addPair({0.0, 1.0, -1});
        assert(manager
                   .computeDiagramCostMatrix(negative_dimension_diagram_cost, diagram_cost_a,
                                             diagram_cost_matrix)
                   .isError());
        assert(diagram_cost_matrix.empty());

        nerve::streaming::Window window;
        window.points = points;
        window.max_size = 3;
        window.max_dimension = 1;
        window.max_radius = 1.5;

        nerve::streaming::Slide slide;
        slide.old_point = {1.0, 0.0};
        slide.new_point = {0.0, 1.0};
        slide.max_radius = 1.1;
        window.pending_slide = slide;

        nerve::persistence::Diagram streaming_diagram;
        assert(manager.processWindowSlide(window, streaming_diagram).isSuccess());
        assert(!window.pending_slide.has_value());
        assert(window.points.size() == 3);
        assert(!streaming_diagram.isEmpty());
        assert(!window.last_affected_indices.empty());

        std::vector<int> affected;
        assert(manager.detectAffectedRegion(window, slide, affected).isSuccess());
        assert(!affected.empty());
        nerve::streaming::Window overflowing_streaming_window;
        overflowing_streaming_window.points = {{0.0}};
        overflowing_streaming_window.max_radius = 1.0;
        nerve::streaming::Slide overflowing_streaming_slide;
        overflowing_streaming_slide.new_point = {std::numeric_limits<double>::max()};
        assert(manager
                   .detectAffectedRegion(overflowing_streaming_window, overflowing_streaming_slide,
                                         affected)
                   .isError());
        assert(affected.empty());
        overflowing_streaming_slide.new_point = {0.0};
        overflowing_streaming_slide.max_radius = std::numeric_limits<double>::max();
        assert(manager
                   .detectAffectedRegion(overflowing_streaming_window, overflowing_streaming_slide,
                                         affected)
                   .isError());
        assert(affected.empty());

        manager.shutdown();

        assert(nerve::gpu::multi::recommendedNumGpus(0, 2) == 0);
        assert(nerve::gpu::multi::recommendedNumGpus(2, 0) == 0);
    }
#endif

    {
        return 0;
    }
}
