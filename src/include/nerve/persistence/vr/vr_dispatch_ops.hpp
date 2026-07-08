
#pragma once

#include "nerve/core.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <vector>

namespace nerve::persistence
{

/**
 * @brief Result from edge collapse preprocessing.
 *
 * Contains the reduced vertex set and neighbor information
 * after edge collapse stage.
 */
struct VRDispatchEdgeCollapseResult
{
    std::vector<bool> vertex_alive;
    std::vector<std::vector<int>> reduced_neighbors;
    size_t n_vertices_alive = 0;

    VRDispatchEdgeCollapseResult() = default;
    explicit VRDispatchEdgeCollapseResult(size_t n_vertices)
        : vertex_alive(n_vertices, true)
        , reduced_neighbors(n_vertices)
        , n_vertices_alive(n_vertices)
    {}
};

/**
 * @brief Configuration for dispatch VR computation.
 *
 * Optional acceleration flags are explicit opt-ins.
 */
struct VRDispatchConfig
{
    // Standard VR parameters
    Size max_dim = 2;
    double max_radius = 1.0;

    // Algorithm selection.
    bool use_edge_collapse = false;      // Edge Collapser preprocessing
    bool use_union_find_d0 = true;       // Union-Find for 0D persistence
    bool use_union_find_top = true;      // Union-Find for top-dimensional
    bool use_lockfree_reduction = false; // Lockfree parallel reduction
    bool use_discrete_morse = false;     // Discrete Morse preprocessing

    // Runtime tuning.
    int num_threads = 0;                  // 0 = auto-detect
    size_t min_points_for_collapse = 100; // Don't collapse small graphs
    double collapse_min_reduction = 0.2;  // Min 20% reduction to be worthwhile
};

/**
 * @brief Benchmark results comparing to standard implementation
 */
struct VRDispatchBenchmark
{
    size_t num_points;
    size_t point_dim;

    double standard_time_ms;
    double dispatch_time_ms;
    double speedup;

    size_t standard_num_pairs;
    size_t dispatch_num_pairs;

    bool correct;
};

/**
 * @brief Dispatch VR computation.
 *
 * This API orchestrates several optional preprocessing and reduction stages.
 * Actual throughput depends on dataset geometry, dimension, and available
 * CPU resources; benchmark in the target environment.
 */
std::vector<Pair> computeVrPersistenceDispatch(core::BufferView<const double>points,
                                               Size point_dim, const VRDispatchConfig &config);

/**
 * @brief Benchmark dispatch against standard implementation
 */
VRDispatchBenchmark benchmarkVrDispatch(core::BufferView<const double>points,
                                        Size point_dim, const VRDispatchConfig &config);

/**
 * @brief Select configuration for dataset size.
 */
VRDispatchConfig getVrDispatchConfig(size_t num_points, size_t point_dim);

/**
 * @brief Check if dispatch stages should be used.
 *
 * Returns false until concrete dispatch stages are wired into this API.
 */
inline bool shouldUseDispatchPath(size_t num_points, size_t point_dim)
{
    (void)num_points;
    (void)point_dim;
    return false;
}

} // namespace nerve::persistence
