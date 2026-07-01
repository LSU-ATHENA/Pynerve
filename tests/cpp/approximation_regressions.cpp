#include "nerve/approximation/distance_approximation.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"

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

using nerve::approximation::DiagramPoint;
using nerve::core::BufferView;

constexpr double kTol = 1e-6f;

DiagramPoint make_point(float birth, float death, uint8_t dim = 0)
{
    DiagramPoint p;
    p.birth = birth;
    p.death = death;
    p.persistence = death - birth;
    p.dimension = dim;
    return p;
}

bool check_diagram_point_validity()
{
    DiagramPoint valid = make_point(0.0f, 1.0f);
    if (!valid.isValid())
    {
        std::cerr << "valid point should be valid\n";
        return false;
    }

    DiagramPoint invalid;
    invalid.birth = 1.0f;
    invalid.death = 0.0f;
    invalid.persistence = -1.0f;
    invalid.dimension = 0;
    if (invalid.isValid())
    {
        std::cerr << "invalid point should not be valid\n";
        return false;
    }

    return true;
}

bool check_sliced_wasserstein_basic()
{
    nerve::approximation::ApproximationConfig cfg;
    cfg.num_projections = 50;
    cfg.random_seed = 42;

    nerve::approximation::SlicedWasserstein sw(cfg);

    std::vector<DiagramPoint> d1 = {make_point(0.0f, 1.0f), make_point(0.0f, 2.0f)};
    std::vector<DiagramPoint> d2 = {make_point(0.0f, 1.5f), make_point(0.5f, 2.5f)};

    float dist = sw.computeDistance(d1, d2);
    if (dist < 0.0f)
    {
        std::cerr << "sliced wasserstein distance negative: " << dist << "\n";
        return false;
    }

    float same = sw.computeDistance(d1, d1);
    if (same > kTol)
    {
        std::cerr << "sliced wasserstein(D,D) should be near 0, got " << same << "\n";
        return false;
    }

    return true;
}

bool check_sliced_wasserstein_symmetry()
{
    nerve::approximation::ApproximationConfig cfg;
    cfg.num_projections = 100;
    cfg.random_seed = 42;

    nerve::approximation::SlicedWasserstein sw(cfg);

    std::vector<DiagramPoint> d1 = {make_point(0.0f, 1.0f)};
    std::vector<DiagramPoint> d2 = {make_point(0.0f, 3.0f)};

    float d12 = sw.computeDistance(d1, d2);
    float d21 = sw.computeDistance(d2, d1);

    if (std::abs(d12 - d21) > kTol)
    {
        std::cerr << "sliced wasserstein not symmetric: " << d12 << " vs " << d21 << "\n";
        return false;
    }

    return true;
}

bool check_approximate_bottleneck_basic()
{
    nerve::approximation::ApproximateBottleneck::BottleneckConfig cfg;
    cfg.approximation_factor = 2.0f;
    cfg.num_landmark_points = 10;
    cfg.enable_random_sampling = false;

    nerve::approximation::ApproximateBottleneck ab(cfg);

    std::vector<DiagramPoint> d1 = {make_point(0.0f, 1.0f)};
    std::vector<DiagramPoint> d2 = {make_point(0.0f, 2.0f)};

    float dist = ab.computeDistance(d1, d2);
    if (dist < 0.0f)
    {
        std::cerr << "approx bottleneck negative: " << dist << "\n";
        return false;
    }

    return true;
}

bool check_distance_matrix()
{
    nerve::approximation::ApproximationConfig cfg;
    cfg.num_projections = 30;
    cfg.random_seed = 123;

    nerve::approximation::SlicedWasserstein sw(cfg);

    std::vector<std::vector<DiagramPoint>> diagrams = {
        {make_point(0.0f, 1.0f)}, {make_point(0.0f, 2.0f)}, {make_point(0.0f, 3.0f)}};

    auto matrix = sw.computeDistanceMatrix(diagrams);
    if (matrix.size() != diagrams.size())
    {
        std::cerr << "distance matrix rows mismatch\n";
        return false;
    }
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        if (matrix[i].size() != diagrams.size())
        {
            std::cerr << "distance matrix cols mismatch at row " << i << "\n";
            return false;
        }
        if (std::abs(matrix[i][i]) > kTol)
        {
            std::cerr << "distance matrix diagonal should be 0\n";
            return false;
        }
    }

    return true;
}

bool check_coarse_grained_matcher()
{
    nerve::approximation::CoarseGrainedMatcher::CoarseConfig cfg;
    cfg.grid_resolution = 10;
    cfg.birth_range_min = 0.0f;
    cfg.birth_range_max = 1.0f;
    cfg.death_range_min = 0.0f;
    cfg.death_range_max = 2.0f;

    nerve::approximation::CoarseGrainedMatcher matcher(cfg);

    std::vector<DiagramPoint> d1 = {make_point(0.2f, 1.0f)};
    std::vector<DiagramPoint> d2 = {make_point(0.3f, 1.5f)};

    auto grid1 = matcher.discretizeDiagram(d1);
    auto grid2 = matcher.discretizeDiagram(d2);

    if (grid1.empty() || grid2.empty())
    {
        std::cerr << "discretized grids should not be empty\n";
        return false;
    }

    float gdist = matcher.computeGridDistance(grid1, grid2);
    if (gdist < 0.0f)
    {
        std::cerr << "grid distance negative\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    if (!check_diagram_point_validity())
    {
        std::cerr << "FAIL: diagram point validity\n";
        return 1;
    }
    if (!check_sliced_wasserstein_basic())
    {
        std::cerr << "FAIL: sliced wasserstein basic\n";
        return 1;
    }
    if (!check_sliced_wasserstein_symmetry())
    {
        std::cerr << "FAIL: sliced wasserstein symmetry\n";
        return 1;
    }
    if (!check_approximate_bottleneck_basic())
    {
        std::cerr << "FAIL: approximate bottleneck basic\n";
        return 1;
    }
    if (!check_distance_matrix())
    {
        std::cerr << "FAIL: distance matrix\n";
        return 1;
    }
    if (!check_coarse_grained_matcher())
    {
        std::cerr << "FAIL: coarse grained matcher\n";
        return 1;
    }
    return 0;
}
