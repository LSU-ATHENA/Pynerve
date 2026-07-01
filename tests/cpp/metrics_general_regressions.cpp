#include "nerve/core_types.hpp"
#include "nerve/metrics/distances.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::metrics::Diagram;
using nerve::persistence::Pair;

constexpr double kTol = 1e-10;

Diagram make_diagram(const std::vector<Pair> &pairs)
{
    Diagram d;
    for (const auto &p : pairs)
        d.addPair(p);
    return d;
}

bool check_factory_creates_bottleneck()
{
    auto bd = nerve::metrics::DistanceMetricFactory::createBottleneck();
    if (!bd)
    {
        std::cerr << "factory returned null for bottleneck\n";
        return false;
    }
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});
    double dist = bd->compute(d1, d2);
    if (dist <= 0.0)
    {
        std::cerr << "factory bottleneck compute failed\n";
        return false;
    }
    return true;
}

bool check_factory_creates_wasserstein()
{
    auto wd = nerve::metrics::DistanceMetricFactory::createWasserstein(2.0);
    if (!wd)
    {
        std::cerr << "factory returned null for wasserstein\n";
        return false;
    }
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});
    double dist = wd->compute(d1, d2);
    if (dist < 0.0)
    {
        std::cerr << "factory wasserstein compute failed\n";
        return false;
    }
    return true;
}

bool check_factory_creates_frechet()
{
    auto fd = nerve::metrics::DistanceMetricFactory::createFrechet();
    if (!fd)
    {
        std::cerr << "factory returned null for frechet\n";
        return false;
    }
    return true;
}

bool check_factory_compute_distance()
{
    Diagram d1 = make_diagram({{0.0, 1.0, 0}});
    Diagram d2 = make_diagram({{0.0, 2.0, 0}});
    double dist = nerve::metrics::DistanceMetricFactory::computeDistance(
        nerve::metrics::DistanceMetricFactory::MetricType::BOTTLENECK, d1, d2);
    if (dist <= 0.0)
    {
        std::cerr << "factory computeDistance failed\n";
        return false;
    }
    return true;
}

bool check_hausdorff_point_sets()
{
    std::vector<std::vector<double>> p1 = {{0.0, 0.0}, {1.0, 0.0}};
    std::vector<std::vector<double>> p2 = {{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}};
    double dist = nerve::metrics::hausdorffDistance(p1, p2);
    if (dist < 0.0 || !std::isfinite(dist))
    {
        std::cerr << "hausdorff distance invalid: " << dist << "\n";
        return false;
    }
    return true;
}

bool check_hausdorff_self_zero()
{
    std::vector<std::vector<double>> pts = {{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}};
    double dist = nerve::metrics::hausdorffDistance(pts, pts);
    if (std::abs(dist) > kTol)
    {
        std::cerr << "hausdorff(P,P) should be 0, got " << dist << "\n";
        return false;
    }
    return true;
}

bool check_chamfer_distance()
{
    std::vector<std::vector<double>> p1 = {{0.0, 0.0}, {1.0, 0.0}};
    std::vector<std::vector<double>> p2 = {{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}};
    double dist = nerve::metrics::chamferDistance(p1, p2);
    if (dist < 0.0 || !std::isfinite(dist))
    {
        std::cerr << "chamfer distance invalid: " << dist << "\n";
        return false;
    }
    return true;
}

bool check_earth_movers_distance()
{
    std::vector<std::vector<double>> p1 = {{0.0, 0.0}, {1.0, 0.0}};
    std::vector<std::vector<double>> p2 = {{0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}};
    double dist = nerve::metrics::earthMoversDistance(p1, p2);
    if (dist < 0.0 || !std::isfinite(dist))
    {
        std::cerr << "earth mover distance invalid: " << dist << "\n";
        return false;
    }
    return true;
}

bool check_distance_matrix_compute()
{
    std::vector<Diagram> diagrams;
    diagrams.push_back(make_diagram({{0.0, 1.0, 0}}));
    diagrams.push_back(make_diagram({{0.0, 2.0, 0}}));
    diagrams.push_back(make_diagram({{0.5, 1.5, 0}}));
    auto mat = nerve::metrics::DistanceMatrix::computeDiagramDistanceMatrix(diagrams);
    if (mat.size() != 3)
    {
        std::cerr << "matrix should be 3x3\n";
        return false;
    }
    for (const auto &row : mat)
    {
        if (row.size() != 3)
        {
            std::cerr << "matrix row wrong size\n";
            return false;
        }
    }
    return true;
}

bool check_distance_matrix_symmetric()
{
    std::vector<Diagram> diagrams;
    diagrams.push_back(make_diagram({{0.0, 1.0, 0}}));
    diagrams.push_back(make_diagram({{0.0, 2.0, 0}}));
    auto mat = nerve::metrics::DistanceMatrix::computeDiagramDistanceMatrix(diagrams);
    if (std::abs(mat[0][1] - mat[1][0]) > kTol)
    {
        std::cerr << "distance matrix not symmetric\n";
        return false;
    }
    return true;
}

bool check_distance_stats_mean()
{
    std::vector<std::vector<double>> mat = {{0.0, 1.0}, {1.0, 0.0}};
    double mean = nerve::metrics::DistanceStatistics::computeMean(mat);
    if (std::abs(mean - 1.0) > kTol)
    {
        std::cerr << "mean should be 1.0, got " << mean << "\n";
        return false;
    }
    return true;
}

bool check_distance_stats_stddev()
{
    std::vector<std::vector<double>> mat = {{0.0, 1.0}, {1.0, 0.0}};
    double sd = nerve::metrics::DistanceStatistics::computeStdDeviation(mat);
    if (sd < 0.0 || !std::isfinite(sd))
    {
        std::cerr << "stddev invalid: " << sd << "\n";
        return false;
    }
    return true;
}

bool check_distance_stats_row_means()
{
    std::vector<std::vector<double>> mat = {{0.0, 2.0}, {2.0, 0.0}};
    auto means = nerve::metrics::DistanceStatistics::computeRowMeans(mat);
    if (means.size() != 2)
    {
        std::cerr << "row means wrong size\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_factory_creates_bottleneck())
    {
        std::cerr << "FAIL: factory bottleneck\n";
        return 1;
    }
    if (!check_factory_creates_wasserstein())
    {
        std::cerr << "FAIL: factory wasserstein\n";
        return 1;
    }
    if (!check_factory_creates_frechet())
    {
        std::cerr << "FAIL: factory frechet\n";
        return 1;
    }
    if (!check_factory_compute_distance())
    {
        std::cerr << "FAIL: factory compute\n";
        return 1;
    }
    if (!check_hausdorff_point_sets())
    {
        std::cerr << "FAIL: hausdorff\n";
        return 1;
    }
    if (!check_hausdorff_self_zero())
    {
        std::cerr << "FAIL: hausdorff self\n";
        return 1;
    }
    if (!check_chamfer_distance())
    {
        std::cerr << "FAIL: chamfer\n";
        return 1;
    }
    if (!check_earth_movers_distance())
    {
        std::cerr << "FAIL: earth mover\n";
        return 1;
    }
    if (!check_distance_matrix_compute())
    {
        std::cerr << "FAIL: dist matrix\n";
        return 1;
    }
    if (!check_distance_matrix_symmetric())
    {
        std::cerr << "FAIL: matrix symmetric\n";
        return 1;
    }
    if (!check_distance_stats_mean())
    {
        std::cerr << "FAIL: stats mean\n";
        return 1;
    }
    if (!check_distance_stats_stddev())
    {
        std::cerr << "FAIL: stats stddev\n";
        return 1;
    }
    if (!check_distance_stats_row_means())
    {
        std::cerr << "FAIL: stats row means\n";
        return 1;
    }
    return 0;
}
