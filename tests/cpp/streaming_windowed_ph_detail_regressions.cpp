
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/streaming/incremental.hpp"
#include "nerve/streaming/windowed_ph.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace
{

using nerve::algebra::Simplex;
using nerve::persistence::Diagram;
using nerve::persistence::Pair;
using nerve::streaming::AcceleratedWindowedPH;
using nerve::streaming::NUMAMemoryPool;
using nerve::streaming::PartialRecomputeHeuristic;
using nerve::streaming::StreamingPHPerformanceHarness;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(42);
}

bool check_accelerated_windowed_ph_default()
{
    AcceleratedWindowedPH aph;
    auto metrics = aph.getPerformanceMetrics();
    if (metrics.total_updates_processed > 0)
    {
        std::cerr << "new PH should have 0 updates\n";
        return false;
    }
    return true;
}

bool check_accelerated_windowed_ph_add_simplex()
{
    AcceleratedWindowedPH aph;
    aph.addSimplex(Simplex({0}));
    aph.addSimplex(Simplex({1}));
    aph.addSimplex(Simplex({0, 1}));
    auto diagram = aph.compute();
    if (diagram.getPairs().empty() && diagram.count() == 0)
    {
        std::cerr << "accelerated PH should produce diagram\n";
        return false;
    }
    return true;
}

bool check_accelerated_windowed_ph_add_batch()
{
    AcceleratedWindowedPH aph;
    std::vector<Simplex> batch = {Simplex({0}),    Simplex({1}),    Simplex({2}),
                                  Simplex({0, 1}), Simplex({0, 2}), Simplex({1, 2})};
    aph.addSimplicesBatch(batch);
    auto diagram = aph.compute();
    if (diagram.getPairs().empty() && diagram.count() == 0)
    {
        std::cerr << "batch add should produce diagram\n";
        return false;
    }
    return true;
}

bool check_window_slide_operation()
{
    AcceleratedWindowedPH aph;
    aph.addSimplex(Simplex({0}));
    aph.addSimplex(Simplex({1}));
    aph.addSimplex(Simplex({0, 1}));
    aph.slideWindow(2);
    aph.addSimplex(Simplex({2}));
    aph.addSimplex(Simplex({0, 2}));
    auto diagram = aph.compute();
    for (const auto &p : diagram.getPairs())
    {
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "negative persistence after slide\n";
            return false;
        }
    }
    return true;
}

bool check_overlapping_windows()
{
    AcceleratedWindowedPH aph;
    aph.setWindowSize(4);
    aph.addSimplex(Simplex({0}));
    aph.addSimplex(Simplex({1}));
    aph.addSimplex(Simplex({0, 1}));
    auto diag1 = aph.compute();
    aph.slideWindow(2);
    aph.addSimplex(Simplex({2}));
    aph.addSimplex(Simplex({1, 2}));
    auto diag2 = aph.compute();
    if (diag1.getPairs().empty() && diag2.getPairs().empty())
    {
        std::cerr << "overlapping windows should both produce diagrams\n";
        return false;
    }
    return true;
}

bool check_partial_recompute_heuristic()
{
    PartialRecomputeHeuristic heuristic;
    Diagram valid;
    valid.addPair({0.0, 1.0, 0});
    std::vector<Simplex> added = {Simplex({2})};
    auto strategy = heuristic.determineStrategy(added, {}, valid);
    if (heuristic.getConfig().strategy == PartialRecomputeHeuristic::RecomputeStrategy::ADAPTIVE)
    {
        (void)strategy;
    }
    return true;
}

bool check_numa_memory_pool_construction()
{
    NUMAMemoryPool pool;
    if (pool.getPoolSize() == 0)
    {
        std::cerr << "memory pool should have non-zero size\n";
        return false;
    }
    return true;
}

bool check_accelerated_windowed_ph_optimization_config()
{
    AcceleratedWindowedPH::OptimizationConfig cfg;
    cfg.enable_numa_optimization = true;
    cfg.enable_parallel_processing = true;
    cfg.num_threads = 2;
    AcceleratedWindowedPH aph(cfg);
    auto retrieved = aph.getOptimizationConfig();
    if (retrieved.num_threads != cfg.num_threads)
    {
        std::cerr << "optimization config not preserved\n";
        return false;
    }
    if (!retrieved.obey_no_alloc_invariant)
    {
        std::cerr << "no-alloc invariant should be enabled\n";
        return false;
    }
    return true;
}

bool check_performance_harness_construction()
{
    StreamingPHPerformanceHarness harness;
    (void)harness;
    return true;
}

} // namespace

int main()
{
    if (!check_accelerated_windowed_ph_default())
    {
        std::cerr << "FAIL: accelerated PH default\n";
        return 1;
    }
    if (!check_accelerated_windowed_ph_add_simplex())
    {
        std::cerr << "FAIL: accelerated PH add\n";
        return 1;
    }
    if (!check_accelerated_windowed_ph_add_batch())
    {
        std::cerr << "FAIL: accelerated PH batch\n";
        return 1;
    }
    if (!check_window_slide_operation())
    {
        std::cerr << "FAIL: window slide\n";
        return 1;
    }
    if (!check_overlapping_windows())
    {
        std::cerr << "FAIL: overlapping windows\n";
        return 1;
    }
    if (!check_partial_recompute_heuristic())
    {
        std::cerr << "FAIL: partial recompute\n";
        return 1;
    }
    if (!check_numa_memory_pool_construction())
    {
        std::cerr << "FAIL: NUMA pool\n";
        return 1;
    }
    if (!check_accelerated_windowed_ph_optimization_config())
    {
        std::cerr << "FAIL: optimization config\n";
        return 1;
    }
    if (!check_performance_harness_construction())
    {
        std::cerr << "FAIL: performance harness\n";
        return 1;
    }
    return 0;
}
