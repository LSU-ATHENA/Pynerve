
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/streaming/diagram_sorting.hpp"
#include "nerve/streaming/incremental.hpp"
#include "nerve/streaming/streaming_tda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Size;
using nerve::persistence::Diagram;
using nerve::persistence::Pair;
using nerve::streaming::IncrementalPH;
using nerve::streaming::WindowedPH;
using nerve::streaming::detail::canonicalPairs;
using nerve::streaming::detail::diagramSupDistance;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(42);
}

bool check_diagram_sorting_preserves_pairs()
{
    Diagram diagram;
    diagram.addPair({2.0, 5.0, 0});
    diagram.addPair({0.0, 3.0, 0});
    diagram.addPair({1.0, std::numeric_limits<double>::infinity(), 1});
    auto sorted = canonicalPairs(diagram);
    if (sorted.size() != 3)
    {
        std::cerr << "canonical pairs should preserve count\n";
        return false;
    }
    for (Size i = 1; i < sorted.size(); ++i)
    {
        if (std::tie(sorted[i].dimension, sorted[i].birth) <
            std::tie(sorted[i - 1].dimension, sorted[i - 1].birth))
        {
            std::cerr << "sorting not monotonic\n";
            return false;
        }
    }
    return true;
}

bool check_diagram_sup_distance()
{
    Diagram a, b;
    a.addPair({0.0, 1.0, 0});
    a.addPair({0.0, std::numeric_limits<double>::infinity(), 1});
    b.addPair({0.0, 1.0, 0});
    b.addPair({0.0, std::numeric_limits<double>::infinity(), 1});
    double d = diagramSupDistance(a, b);
    if (d > TOL)
    {
        std::cerr << "identical diagrams should have distance 0, got " << d << "\n";
        return false;
    }
    return true;
}

bool check_incremental_ph_basic()
{
    IncrementalPH ph(2);
    ph.addSimplex(nerve::algebra::Simplex({0}));
    ph.addSimplex(nerve::algebra::Simplex({1}));
    ph.addSimplex(nerve::algebra::Simplex({0, 1}));
    auto pairs = ph.getPersistencePairs();
    if (pairs.empty())
    {
        std::cerr << "incremental PH should produce pairs\n";
        return false;
    }
    return true;
}

bool check_windowed_persistence_on_timeseries()
{
    WindowedPH wph(3, 1);
    wph.addDataPoint({0.0});
    wph.addDataPoint({1.0});
    wph.addDataPoint({2.0});
    auto diagram = wph.getWindowPersistence();
    if (!diagram.getPairs().empty())
    {
        for (const auto &p : diagram.getPairs())
        {
            if (!p.isInfinite() && p.birth > p.death + TOL)
            {
                std::cerr << "birth > death in windowed persistence\n";
                return false;
            }
        }
    }
    return true;
}

bool check_window_slide()
{
    WindowedPH wph(2, 1);
    wph.addDataPoint({0.0});
    wph.addDataPoint({1.0});
    wph.slideWindow();
    wph.addDataPoint({2.0});
    auto pairs = wph.getWindowPairs();
    for (const auto &p : pairs)
    {
        if (!p.isInfinite() && p.lifetime() < 0.0)
        {
            std::cerr << "negative persistence in window\n";
            return false;
        }
    }
    return true;
}

bool check_streaming_tda_config()
{
    nerve::streaming::StreamingConfig config;
    config.window_size = 100;
    config.stride = 50;

    nerve::streaming::ApproximateStreamingPH ph(config);
    nerve::streaming::StreamDataPoint pt;
    pt.coordinates = {0.0f, 1.0f};
    pt.timestamp_ns = 1000;
    pt.point_id = 0;
    pt.weight = 1.0f;
    if (!pt.isValid())
    {
        std::cerr << "stream data point should be valid\n";
        return false;
    }
    ph.addDataPoint(pt);
    auto window = ph.computeCurrentWindow();
    if (!window.isValid() && !window.points.empty())
    {
        std::cerr << "window should be valid when non-empty\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_diagram_sorting_preserves_pairs())
    {
        std::cerr << "FAIL: diagram sorting\n";
        return 1;
    }
    if (!check_diagram_sup_distance())
    {
        std::cerr << "FAIL: diagram sup distance\n";
        return 1;
    }
    if (!check_incremental_ph_basic())
    {
        std::cerr << "FAIL: incremental ph basic\n";
        return 1;
    }
    if (!check_windowed_persistence_on_timeseries())
    {
        std::cerr << "FAIL: windowed persistence\n";
        return 1;
    }
    if (!check_window_slide())
    {
        std::cerr << "FAIL: window slide\n";
        return 1;
    }
    if (!check_streaming_tda_config())
    {
        std::cerr << "FAIL: streaming tda config\n";
        return 1;
    }
    return 0;
}
