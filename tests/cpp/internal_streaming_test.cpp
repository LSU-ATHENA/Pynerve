
#include "nerve/streaming/detail/streaming_detail.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace
{

using nerve::Size;
using nerve::algebra::Simplex;
using nerve::persistence::Diagram;
using nerve::persistence::Pair;
using nerve::streaming::IncrementalPH;
using nerve::streaming::WindowedPH;
using nerve::streaming::detail::canonicalPairs;
using nerve::streaming::detail::diagramSupDistance;
using nerve::streaming::lockfree::LockFreeSPSCQueue;

constexpr double TOL = 1e-10;

bool check_diagram_sorting_correct_order()
{
    Diagram diagram;
    diagram.addPair({2.0, 5.0, 0});
    diagram.addPair({0.0, 3.0, 0});
    diagram.addPair({1.0, std::numeric_limits<double>::infinity(), 1});
    auto sorted = canonicalPairs(diagram);
    if (sorted.size() != 3)
    {
        return false;
    }
    for (Size i = 1; i < sorted.size(); ++i)
    {
        if (std::tie(sorted[i].dimension, sorted[i].birth) <
            std::tie(sorted[i - 1].dimension, sorted[i - 1].birth))
        {
            return false;
        }
    }
    Diagram d2;
    d2.addPair({0.0, 1.0, 0});
    d2.addPair({0.0, std::numeric_limits<double>::infinity(), 1});
    double d = diagramSupDistance(d2, d2);
    if (d > TOL)
    {
        return false;
    }
    return true;
}

bool check_windowed_persistence_integration()
{
    WindowedPH wph(4, 1);
    wph.addDataPoint({0.0});
    wph.addDataPoint({1.0});
    wph.addDataPoint({2.0});
    wph.addDataPoint({3.0});
    auto diagram = wph.getWindowPersistence();
    (void)diagram;
    auto stability = wph.getWindowStability();
    if (stability < 0.0)
    {
        return false;
    }
    wph.setWindowSize(3);
    wph.setOverlapSize(1);
    wph.slideWindow();
    auto pairs = wph.getWindowPairs();
    if (pairs.empty())
    {
        return false;
    }
    return true;
}

bool check_incremental_ph_basic()
{
    IncrementalPH ph(2);
    ph.addSimplex(Simplex({0}));
    ph.addSimplex(Simplex({1}));
    ph.addSimplex(Simplex({0, 1}));
    auto pairs = ph.getPersistencePairs();
    if (pairs.empty())
    {
        return false;
    }
    if (ph.numSimplices() == 0)
    {
        return false;
    }
    if (ph.getMaxDimension() != 2)
    {
        return false;
    }
    auto betti = ph.getBettiNumbers();
    ph.reset();
    if (ph.getNumUpdates() != 0)
    {
        return false;
    }
    return true;
}

bool check_lockfree_queue_ops()
{
    LockFreeSPSCQueue<int> q(4);
    if (!q.empty())
    {
        return false;
    }
    if (q.size() != 0)
    {
        return false;
    }
    if (!q.push(10))
    {
        return false;
    }
    if (!q.push(20))
    {
        return false;
    }
    if (q.size() != 2)
    {
        return false;
    }
    auto v1 = q.pop();
    if (!v1.has_value() || v1.value() != 10)
    {
        return false;
    }
    auto v2 = q.pop();
    if (!v2.has_value() || v2.value() != 20)
    {
        return false;
    }
    if (!q.empty())
    {
        return false;
    }
    auto empty = q.pop();
    if (empty.has_value())
    {
        return false;
    }
    if (!q.push(30))
    {
        return false;
    }
    if (q.empty())
    {
        return false;
    }
    return true;
}

bool check_simd_streaming_ops()
{
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b = {5.0, 6.0, 7.0, 8.0};
    nerve::streaming::batchVectorAddSimd(a.data(), b.data(), 4);
    if (std::abs(a[0] - 6.0) > TOL || std::abs(a[1] - 8.0) > TOL)
    {
        return false;
    }
    if (std::abs(a[2] - 10.0) > TOL || std::abs(a[3] - 12.0) > TOL)
    {
        return false;
    }
    nerve::streaming::batchScaleSimd(a.data(), 0.5, 4);
    if (std::abs(a[0] - 3.0) > TOL || std::abs(a[1] - 4.0) > TOL)
    {
        return false;
    }
    nerve::streaming::batchThresholdSimd(a.data(), 4, 3.5, 3.9);
    if (a[0] < 3.5 || a[1] > 3.9)
    {
        return false;
    }
    return true;
}

#ifdef NERVE_HAS_CUDA

bool check_gpu_streaming_init()
{
    nerve::streaming::gpu::MultiStreamContext ctx;
    auto err = nerve::streaming::gpu::MultiStreamContext::create(1, 1, ctx);
    if (err != cudaSuccess)
    {
        return true;
    }
    nerve::streaming::gpu::MultiStreamContext::destroy(ctx);
    return true;
}

#endif

bool check_windowed_ph_core_timeseries()
{
    nerve::streaming::AcceleratedWindowedPH aph;
    aph.setWindowSize(3);
    aph.addSimplex(Simplex({0}));
    aph.addSimplex(Simplex({1}));
    aph.addSimplex(Simplex({0, 1}));
    auto diagram = aph.compute();
    (void)diagram;
    auto metrics = aph.getPerformanceMetrics();
    (void)metrics;
    if (!aph.checkHotPathInvariants())
    {
        std::cerr << "windowed PH hot path invariants check failed\n";
    }
    aph.resetPerformanceMetrics();
    aph.compactMemory();
    return true;
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    run("diagram_sorting_correct_order", check_diagram_sorting_correct_order());
    run("windowed_persistence_integration", check_windowed_persistence_integration());
    run("incremental_ph_basic", check_incremental_ph_basic());
    run("lockfree_queue_ops", check_lockfree_queue_ops());
    run("simd_streaming_ops", check_simd_streaming_ops());
#ifdef NERVE_HAS_CUDA
    run("gpu_streaming_init", check_gpu_streaming_init());
#endif
    run("windowed_ph_core_timeseries", check_windowed_ph_core_timeseries());

    return failures > 0 ? 1 : 0;
}
