#include "detail/vr_medium_hybrid_helpers.inl"
#include "nerve/algebra/boundary.hpp"
#include "nerve/algebra/complex.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"
#include "nerve/persistence/vr/vr_distance_tiled_ops.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"
#include "nerve/persistence/vr/vr_fast_simd_ops.hpp"
#include "nerve/persistence/vr/vr_medium_hybrid_ops.hpp"
#include "nerve/runtime/hardware_probe.hpp"
#include "vr_medium_hybrid_expander.hpp"

#ifdef NERVE_HAS_CUDA
#include "nerve/gpu/distance_fasted.cuh"
#include "nerve/gpu/distance_tedjoin.cuh"
#include "nerve/gpu/persistence_reducer.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <thread>
#include <tuple>
#include <vector>

namespace nerve::persistence
{

void buildDistanceMatrixOptimized(const std::vector<double> &points, size_t point_dim,
                                  size_t num_points,
                                  std::vector<std::vector<double>> &distance_matrix,
                                  const ExecutionPlan &plan, const HybridWorkDistribution &work)
{
    (void)work;
    std::vector<double> flat_matrix;
    size_t matrix_size = 0;
    if (!checkedSquareCount(num_points, matrix_size) ||
        matrix_size > std::vector<double>().max_size() ||
        num_points > std::vector<double>().max_size() ||
        num_points > std::vector<std::vector<double>>().max_size())
    {
        distance_matrix.clear();
        return;
    }

#ifdef NERVE_HAS_CUDA
    if (work.gpu_distance_matrix_ratio > 0.0)
    {
        flat_matrix.assign(matrix_size, 0.0);
        const int n = static_cast<int>(num_points);
        const int dim = static_cast<int>(point_dim);
        cudaError_t cuda_status = ::nerve::gpu::tedjoin::launchFp64TensorDistance(
            points.data(), n, dim, flat_matrix.data(), n);
        if (cuda_status == cudaSuccess)
        {
            if (flat_matrix.size() == matrix_size)
            {
                distance_matrix.assign(num_points, std::vector<double>(num_points, 0.0));
                for (size_t i = 0; i < num_points; ++i)
                {
                    for (size_t j = 0; j < num_points; ++j)
                    {
                        distance_matrix[i][j] = flat_matrix[i * num_points + j];
                    }
                }
                return;
            }
        }
    }
#endif

    if (plan.prefer_numa_tiling)
    {
        flat_matrix =
            computeDistanceMatrixNumaAware(points, point_dim, num_points, plan.numa_nodes);
    }
    else
    {
        flat_matrix = computeDistanceMatrixTiled(points, point_dim, num_points, 0);
    }
    if (flat_matrix.size() != matrix_size)
    {
        distance_matrix.clear();
        return;
    }

    distance_matrix.assign(num_points, std::vector<double>(num_points, 0.0));
    for (size_t i = 0; i < num_points; ++i)
    {
        for (size_t j = 0; j < num_points; ++j)
        {
            distance_matrix[i][j] = flat_matrix[i * num_points + j];
        }
    }
}

void buildRadiusGraph(const std::vector<std::vector<double>> &distance_matrix, double max_radius,
                      std::vector<std::vector<int>> &neighbors)
{
    size_t num_points = distance_matrix.size();
    neighbors.assign(num_points, {});

    for (size_t i = 0; i < num_points; ++i)
    {
        for (size_t j = i + 1; j < num_points; ++j)
        {
            if (!std::isfinite(distance_matrix[i][j]) || distance_matrix[i][j] > max_radius)
            {
                continue;
            }
            neighbors[i].push_back(static_cast<int>(j));
            neighbors[j].push_back(static_cast<int>(i));
        }
    }

    for (size_t i = 0; i < num_points; ++i)
    {
        std::ranges::sort(neighbors[i]);
    }
}

std::vector<Pair> computeVrPersistenceMediumHybrid(core::BufferView<const double> points,
                                                   Size point_dim, const VRConfig &config)
{
    if (!hasValidMediumHybridInput(points, point_dim, config))
    {
        return {};
    }

    const Size num_points = points.size() / point_dim;
    if (num_points < FAST_PATH_THRESHOLD)
    {
        return computeVrPersistenceFastSimd(points, point_dim, config);
    }
    if (num_points > EXACT_PATH_THRESHOLD)
    {
        return computeVrPersistenceFast(points, point_dim, config);
    }

    std::vector<double> point_data(points.begin(), points.end());
    ExecutionPlan plan = computeExecutionPlan(num_points, point_dim, config);

    HybridWorkDistribution work =
        computeOptimalWorkDistribution(num_points, point_dim, /*available_gpu_memory_gb=*/0.0);

    std::vector<std::vector<double>> distance_matrix;
    buildDistanceMatrixOptimized(point_data, point_dim, num_points, distance_matrix, plan, work);
    if (distance_matrix.empty())
    {
        return {};
    }

    std::vector<std::vector<int>> neighbors;
    buildRadiusGraph(distance_matrix, config.max_radius, neighbors);

    algebra::SimplicialComplex complex;
    SimplexSet seen;
    seen.reserve(num_points * 16);

    ParallelCliqueExpander expander(neighbors, distance_matrix, config.max_dim, config.max_radius);
    expander.expand(num_points, complex, seen);

#ifdef NERVE_HAS_CUDA
    if (work.use_gpu_clique_expansion)
    {
        algebra::BoundaryMatrix boundary(complex);
        gpu::kernels::GpuPersistenceReducer gpu_reducer;
        std::vector<Index> gpu_pivots;
        std::vector<std::pair<Size, Size>> gpu_pairs;
        auto gpu_result = gpu_reducer.computeCohomology(boundary, gpu_pivots, gpu_pairs);
        if (gpu_result.isSuccess())
        {
            std::vector<Pair> gpairs;
            gpairs.reserve(gpu_pairs.size());
            for (const auto &[birth_idx, death_idx] : gpu_pairs)
            {
                if (static_cast<Size>(birth_idx) >= boundary.cols() ||
                    static_cast<Size>(death_idx) >= boundary.cols())
                {
                    continue;
                }
                Dimension dim = static_cast<Dimension>(
                    boundary.getColSimplexDimension(static_cast<Size>(birth_idx)));
                if (dim <= static_cast<Dimension>(config.max_dim))
                {
                    double birth_val = boundary.getFiltrationValue(static_cast<Size>(birth_idx));
                    double death_val =
                        (static_cast<Size>(death_idx) < boundary.cols())
                            ? boundary.getFiltrationValue(static_cast<Size>(death_idx))
                            : std::numeric_limits<double>::infinity();
                    gpairs.push_back(Pair{birth_val, death_val, dim});
                }
            }
            std::ranges::sort(gpairs, {}, [](const Pair &p) {
                return std::tuple(p.dimension, p.birth, p.death);
            });
            return gpairs;
        }
    }
#endif

    auto exact = computeExactPersistenceZ2(complex, config.max_dim);
    const auto &diagram = exact.pairs;

    std::vector<Pair> pairs;
    pairs.reserve(diagram.size());
    for (const auto &pair : diagram)
    {
        if (pair.dimension <= static_cast<Dimension>(config.max_dim))
        {
            pairs.push_back(pair);
        }
    }

    std::ranges::sort(pairs, {},
                      [](const Pair &p) { return std::tuple(p.dimension, p.birth, p.death); });

    return pairs;
}

HybridWorkDistribution computeOptimalWorkDistribution(size_t n_points, size_t point_dim,
                                                      double available_gpu_memory_gb)
{
    (void)n_points;
    (void)point_dim;

    HybridWorkDistribution work{};
    work.gpu_distance_matrix_ratio = 0.0;
    work.use_gpu_clique_expansion = false;
    work.tile_size = getOptimalTileSize();
    work.num_threads = std::max<int>(1, static_cast<int>(std::thread::hardware_concurrency()));

    if (available_gpu_memory_gb <= 0.0)
    {
        return work;
    }

    try
    {
        const auto snapshot = runtime::collectHardwareSnapshot();
        if (!runtime::has_cuda_gpu(snapshot))
        {
            return work;
        }

        const auto &gpu = snapshot.gpus.value.front();
        const double gpu_mem_gb =
            static_cast<double>(gpu.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
        const auto &sys_mem = snapshot.total_memory_bytes;
        const double cpu_mem_gb =
            sys_mem.ok() ? static_cast<double>(sys_mem.value) / (1024.0 * 1024.0 * 1024.0) : 0.0;

        if (cpu_mem_gb > 0.0)
        {
            work.gpu_distance_matrix_ratio = gpu_mem_gb / (gpu_mem_gb + cpu_mem_gb);
        }
        else
        {
            work.gpu_distance_matrix_ratio = 0.5;
        }

        work.use_gpu_clique_expansion = (gpu_mem_gb > 2.0);
    }
    catch (...)
    {
        work.gpu_distance_matrix_ratio = 0.0;
        work.use_gpu_clique_expansion = false;
    }

    return work;
}

} // namespace nerve::persistence
