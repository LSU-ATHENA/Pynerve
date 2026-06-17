#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/dmt/gpu_dmt.hpp"

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
using nerve::core::BufferView;

constexpr double kTol = 1e-10;

bool check_simplex_pair_detection()
{
    std::vector<int> edge = {0, 1};
    std::vector<int> vertex = {0};
    std::vector<int> other = {2};

    bool edge_vertex = nerve::dmt::parallel::SimplexPairOps::canFormGradientPair(edge, vertex);
    bool vertex_edge = nerve::dmt::parallel::SimplexPairOps::canFormGradientPair(vertex, edge);
    bool edge_other = nerve::dmt::parallel::SimplexPairOps::canFormGradientPair(edge, other);

    if (!edge_vertex)
    {
        std::cerr << "edge should pair with vertex\n";
        return false;
    }
    if (!vertex_edge)
    {
        std::cerr << "vertex should pair with edge\n";
        return false;
    }
    if (edge_other)
    {
        std::cerr << "disjoint simplices should not pair\n";
        return false;
    }

    return true;
}

bool check_morse_pair_finder_basic()
{
    nerve::dmt::parallel::ParallelMorsePairFinder::Config cfg;
    cfg.num_threads = 1;
    cfg.batch_size = 64;
    cfg.use_simd = false;

    nerve::dmt::parallel::ParallelMorsePairFinder finder(cfg);

    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}};
    std::vector<float> filtration = {0.0f, 0.0f, 0.0f, 1.0f, 1.5f, 1.2f};

    auto pairs = finder.findMorsePairs(simplices, filtration);

    for (const auto &p : pairs)
    {
        if (p.first < 0 || p.second < 0)
        {
            std::cerr << "invalid pair indices\n";
            return false;
        }
        if (static_cast<size_t>(p.first) >= simplices.size() ||
            static_cast<size_t>(p.second) >= simplices.size())
        {
            std::cerr << "pair index out of range\n";
            return false;
        }
    }

    return true;
}

bool check_dmt_engine_construction()
{
    nerve::dmt::DMTConfig cfg;
    cfg.max_dimension = 2;
    cfg.use_parallel = false;

    nerve::dmt::DMTEngine engine(cfg);

    std::vector<std::vector<int>> simplices = {{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}, {0, 1, 2}};
    std::vector<float> filtration = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.5f, 3.0f};

    auto result = engine.computeMorseComplex(simplices, filtration);

    auto critical = engine.findCriticalPoints();

    for (auto c : critical)
    {
        if (c < 0 || static_cast<size_t>(c) >= simplices.size())
        {
            std::cerr << "critical index out of range\n";
            return false;
        }
    }

    return true;
}

bool check_traversal_basic()
{
    nerve::dmt::parallel::CacheOptimizedMorseTraversal traversal;

    std::vector<std::pair<int, int>> pairs = {{0, 1}, {2, 3}};
    std::vector<std::vector<int>> boundaries = {{1}, {0}, {3}, {2}};

    std::vector<int> visited;
    traversal.traverse(pairs, boundaries, visited);

    if (visited.size() != 4)
    {
        std::cerr << "traversal should visit all 4 cells\n";
        return false;
    }

    return true;
}


return true;
}

} // namespace

int main()
{
    if (!check_simplex_pair_detection())
    {
        std::cerr << "FAIL: simplex pair detection\n";
        return 1;
    }
    if (!check_morse_pair_finder_basic())
    {
        std::cerr << "FAIL: morse pair finder basic\n";
        return 1;
    }
    if (!check_dmt_engine_construction())
    {
        std::cerr << "FAIL: dmt engine construction\n";
        return 1;
    }
    if (!check_traversal_basic())
    {
        std::cerr << "FAIL: traversal basic\n";
        return 1;
    }
    return 0;
}
