#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/utils/diagram_statistics.hpp"
#include "nerve/persistence/vr/vr_fast_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::Dimension;
using nerve::Field;
using nerve::Size;
using nerve::common::VRConfig;
using nerve::core::BufferView;
using nerve::persistence::Pair;
using nerve::persistence::VRAlgorithmSelection;

constexpr double kTol = 1e-10;

BufferView<const double> view_of(const std::vector<double> &v)
{
    return {v.data(), v.size()};
}

std::vector<Pair> canonical(std::vector<Pair> pairs)
{
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return std::tuple(a.dimension, a.birth, a.death) <
               std::tuple(b.dimension, b.birth, b.death);
    });
    return pairs;
}

bool check_pipeline_simple_cloud()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);
    if (pairs.empty())
    {
        std::cerr << "pipeline: pairs should not be empty\n";
        return false;
    }

    nerve::persistence::Diagram diagram(pairs);

    if (diagram.count() != pairs.size())
    {
        std::cerr << "pipeline: diagram count " << diagram.count() << " != pairs " << pairs.size()
                  << "\n";
        return false;
    }

    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
        {
            if (p.birth > p.death + kTol)
            {
                std::cerr << "pipeline: birth>death " << p.birth << " > " << p.death << "\n";
                return false;
            }
            if (p.lifetime() < -kTol)
            {
                std::cerr << "pipeline: negative persistence " << p.lifetime() << "\n";
                return false;
            }
        }
        if (p.dimension < 0)
        {
            std::cerr << "pipeline: negative dimension\n";
            return false;
        }
    }

    const auto betti = diagram.computeBetti();
    Size total_infinite = 0;
    for (const auto &b : betti)
        total_infinite += b;
    Size count_infinite_from_pairs = 0;
    for (const auto &p : pairs)
        if (p.isInfinite())
            ++count_infinite_from_pairs;
    if (total_infinite != count_infinite_from_pairs)
    {
        std::cerr << "pipeline: Betti count " << total_infinite << " != infinite pairs "
                  << count_infinite_from_pairs << "\n";
        return false;
    }

    if (betti.empty() || betti[0] < 1)
    {
        std::cerr << "pipeline: Betti_0 must be >= 1\n";
        return false;
    }

    double max_pers = diagram.getMaxPersistence();
    if (max_pers < 0.0)
    {
        std::cerr << "pipeline: max persistence negative\n";
        return false;
    }

    double avg_pers = diagram.getAveragePersistence();
    if (avg_pers < 0.0)
    {
        std::cerr << "pipeline: avg persistence negative\n";
        return false;
    }

    return true;
}

bool check_pipeline_using_betti_numbers_from_pairs()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);
    const auto betti = nerve::persistence::bettiNumbersFromPairs(pairs);

    Size count_infinite = 0;
    for (const auto &p : pairs)
        if (p.isInfinite())
            ++count_infinite;
    Size betti_sum = 0;
    for (const auto &b : betti)
        betti_sum += b;
    if (betti_sum != count_infinite)
    {
        std::cerr << "bettiNumbersFromPairs sum " << betti_sum << " != infinite count "
                  << count_infinite << "\n";
        return false;
    }

    if (betti.empty() || betti[0] < 1)
    {
        std::cerr << "bettiNumbersFromPairs: Betti_0 < 1\n";
        return false;
    }

    return true;
}

bool check_pipeline_statistics_edge_to_edge()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> pts(static_cast<std::size_t>(10) * 2);
    for (auto &v : pts)
        v = dist(rng);

    VRConfig cfg;
    cfg.max_radius = 1.5;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);
    if (pairs.empty())
    {
        std::cerr << "pipeline stats: pairs should not be empty\n";
        return false;
    }

    nerve::persistence::Diagram diagram(pairs);
    const auto betti_from_diagram = diagram.computeBetti();
    const auto betti_from_fn = nerve::persistence::bettiNumbersFromPairs(pairs);

    if (betti_from_diagram.size() != betti_from_fn.size())
    {
        std::cerr << "pipeline stats: Betti size mismatch " << betti_from_diagram.size() << " vs "
                  << betti_from_fn.size() << "\n";
        return false;
    }
    for (std::size_t i = 0; i < betti_from_diagram.size(); ++i)
    {
        if (betti_from_diagram[i] != betti_from_fn[i])
        {
            std::cerr << "pipeline stats: Betti[" << i << "] " << betti_from_diagram[i] << " vs "
                      << betti_from_fn[i] << "\n";
            return false;
        }
    }

    double max_diagram = diagram.getMaxPersistence();
    double max_from_pairs = 0.0;
    for (const auto &p : pairs)
    {
        if (!p.isInfinite())
            max_from_pairs = std::max(max_from_pairs, p.lifetime());
    }
    if (std::abs(max_diagram - max_from_pairs) > kTol)
    {
        std::cerr << "pipeline stats: max persistence mismatch " << max_diagram << " vs "
                  << max_from_pairs << "\n";
        return false;
    }

    std::vector<double> weights;
    for (const auto &p : pairs)
        if (!p.isInfinite())
            weights.push_back(p.lifetime());
    if (!weights.empty())
    {
        double entropy = nerve::persistence::shannonEntropyNormalized(weights);
        if (entropy < 0.0 || entropy > 1.0 + kTol)
        {
            std::cerr << "pipeline stats: entropy out of range " << entropy << "\n";
            return false;
        }
    }

    return true;
}

bool check_pipeline_diagram_get_pairs_by_dimension()
{
    const std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5};
    VRConfig cfg;
    cfg.max_radius = 2.0;
    cfg.max_dim = 2;
    cfg.algorithm = VRAlgorithmSelection::EXACT_STANDARD;

    const auto pairs = nerve::persistence::computeVrPersistenceFast(view_of(pts), 2, cfg);
    nerve::persistence::Diagram diagram(pairs);

    for (Dimension d = 0; d <= 2; ++d)
    {
        auto dim_pairs = diagram.getPairsByDimension(d);
        for (const auto &p : dim_pairs)
        {
            if (p.dimension != d)
            {
                std::cerr << "pipeline: getPairsByDimension(" << d
                          << ") returned dim=" << p.dimension << "\n";
                return false;
            }
        }
    }

    Size total_by_dim = 0;
    for (Dimension d = 0; d <= 2; ++d)
        total_by_dim += diagram.getPairsByDimension(d).size();
    if (total_by_dim != pairs.size())
    {
        std::cerr << "pipeline: pairs by dim sum " << total_by_dim << " != total " << pairs.size()
                  << "\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_pipeline_simple_cloud())
    {
        std::cerr << "FAIL: pipeline simple cloud\n";
        return 1;
    }
    if (!check_pipeline_using_betti_numbers_from_pairs())
    {
        std::cerr << "FAIL: pipeline betti from pairs\n";
        return 1;
    }
    if (!check_pipeline_statistics_edge_to_edge())
    {
        std::cerr << "FAIL: pipeline stats edge-to-edge\n";
        return 1;
    }
    if (!check_pipeline_diagram_get_pairs_by_dimension())
    {
        std::cerr << "FAIL: pipeline pairs by dimension\n";
        return 1;
    }
    return 0;
}
