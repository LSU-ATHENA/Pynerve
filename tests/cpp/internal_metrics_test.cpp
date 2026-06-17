
#include "nerve/algorithms/distance.hpp"
#include "nerve/metrics/detail/metrics_detail.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace
{

using Size;

constexpr double TOL = 1e-9;

bool check_bottleneck_self_distance_zero()
{
    std::vector<std::pair<float, float>> d1 = {{0.0f, 1.0f}, {2.0f, 3.0f}, {1.0f, 5.0f}};
    double dist = nerve::metrics::bottleneck::adaptiveBottleneckDistance(d1, d1);
    return std::abs(dist) < TOL;
}

bool check_bottleneck_symmetric()
{
    std::vector<std::pair<float, float>> d1 = {{0.0f, 1.0f}, {2.0f, 4.0f}};
    std::vector<std::pair<float, float>> d2 = {{0.5f, 1.5f}, {2.5f, 3.5f}};
    double d12 = nerve::metrics::bottleneck::adaptiveBottleneckDistance(d1, d2);
    double d21 = nerve::metrics::bottleneck::adaptiveBottleneckDistance(d2, d1);
    return std::abs(d12 - d21) < TOL;
}

bool check_bottleneck_empty_diagrams()
{
    std::vector<std::pair<float, float>> empty;
    double dist = nerve::metrics::bottleneck::adaptiveBottleneckDistance(empty, empty);
    return std::abs(dist) < TOL;
}

bool check_bottleneck_validate_diagram()
{
    std::vector<std::pair<float, float>> valid = {{0.0f, 1.0f}, {2.0f, 3.0f}};
    try
    {
        nerve::metrics::bottleneck::validateDiagram(valid, "test");
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool check_bottleneck_diagonal_distance()
{
    std::pair<float, float> point = {1.0f, 5.0f};
    double dd = nerve::metrics::bottleneck::diagonalDistance(point);
    return std::abs(dd - 2.0) < TOL;
}

bool check_frechet_self_distance_zero()
{
    nerve::persistence::Diagram d;
    d.addPair(0.0, 1.0, 0);
    d.addPair(2.0, 5.0, 0);
    double dist = nerve::metrics::frechetDistance(d, d);
    return std::abs(dist) < TOL;
}

bool check_frechet_identity_on_diagrams()
{
    nerve::persistence::Diagram d1;
    d1.addPair(0.0, 1.0, 0);
    d1.addPair(1.0, 3.0, 0);
    nerve::persistence::Diagram d2;
    d2.addPair(0.0, 1.0, 0);
    d2.addPair(1.0, 3.0, 0);
    double dist = nerve::metrics::frechetDistance(d1, d2);
    return std::abs(dist) < TOL;
}

bool check_lazy_distance_euclidean()
{
    std::vector<double> pts = {0.0, 0.0, 3.0, 4.0};
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 2, 2, "euclidean", 10);
    double d = mat.getDistance(0, 1);
    return std::abs(d - 5.0) < TOL;
}

bool check_lazy_distance_self_zero()
{
    std::vector<double> pts = {1.0, 2.0, 4.0, 6.0};
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 2, 2, "euclidean", 10);
    double d = mat.getDistance(0, 0);
    return std::abs(d) < TOL;
}

bool check_lazy_distance_manhattan()
{
    std::vector<double> pts = {0.0, 0.0, 3.0, 4.0};
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 2, 2, "manhattan", 10);
    double d = mat.getDistance(0, 1);
    return std::abs(d - 7.0) < TOL;
}

bool check_lazy_no_cache_equals_cached()
{
    std::vector<double> pts = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 3, 2, "euclidean", 100);
    double cached = mat.getDistance(0, 2);
    double nocache = mat.getDistanceNoCache(0, 2);
    return std::abs(cached - nocache) < TOL;
}

bool check_sparse_distance_structure()
{
    std::vector<double> pts = {0.0, 0.0, 0.5, 0.0, 5.0, 0.0};
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::SparseDistanceMatrix sparse(span, 3, 2, 1.0, "euclidean");

    if (sparse.getDistance(0, 1) > 1.0 + TOL)
    {
        return false;
    }
    if (!sparse.isEdge(0, 1))
    {
        return false;
    }
    if (sparse.isEdge(0, 2))
    {
        return false;
    }
    if (!std::isinf(sparse.getDistance(0, 2)))
    {
        return false;
    }
    if (std::abs(sparse.getDistance(0, 0)) > TOL)
    {
        return false;
    }
    return true;
}

bool check_sparse_distance_sparsity()
{
    std::vector<double> pts = {0.0, 0.0, 10.0, 0.0, 20.0, 0.0};
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::SparseDistanceMatrix sparse(span, 3, 2, 1.0, "euclidean");
    double sparsity = sparse.getSparsity();
    return sparsity > 0.5;
}

bool check_sinkhorn_converges_1d()
{
    std::vector<std::pair<float, float>> d1 = {{0.0f, 1.0f}, {2.0f, 3.0f}};
    std::vector<std::pair<float, float>> d2 = {{0.0f, 1.0f}, {2.0f, 3.0f}};
    nerve::metrics::sinkhorn::SinkhornConfig config;
    config.epsilon = 0.1;
    config.max_iterations = 500;
    config.tolerance = 1e-8;
    double dist = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d1, d2, config);
    return std::isfinite(dist) && dist >= 0.0;
}

bool check_sinkhorn_self_distance_low()
{
    std::vector<std::pair<float, float>> d = {{0.0f, 2.0f}, {1.0f, 3.0f}};
    nerve::metrics::sinkhorn::SinkhornConfig config;
    config.epsilon = 0.05;
    config.max_iterations = 1000;
    config.tolerance = 1e-10;
    double dist = nerve::metrics::sinkhorn::sinkhornDiagramDistance(d, d, config);
    return std::isfinite(dist) && dist < 0.5;
}

bool check_sinkhorn_empty_diagrams()
{
    std::vector<std::pair<float, float>> empty;
    double dist = nerve::metrics::sinkhorn::sinkhornDiagramDistance(empty, empty);
    return std::abs(dist) < TOL;
}

bool check_sinkhorn_config_validation()
{
    nerve::metrics::sinkhorn::SinkhornConfig bad;
    bad.epsilon = -1.0;
    try
    {
        std::vector<std::pair<float, float>> d = {{0.0f, 1.0f}};
        nerve::metrics::sinkhorn::sinkhornDiagramDistance(d, d, bad);
        return false;
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
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

    run("bottleneck_self_distance_zero", check_bottleneck_self_distance_zero());
    run("bottleneck_symmetric", check_bottleneck_symmetric());
    run("bottleneck_empty_diagrams", check_bottleneck_empty_diagrams());
    run("bottleneck_validate_diagram", check_bottleneck_validate_diagram());
    run("bottleneck_diagonal_distance", check_bottleneck_diagonal_distance());
    run("frechet_self_distance_zero", check_frechet_self_distance_zero());
    run("frechet_identity_on_diagrams", check_frechet_identity_on_diagrams());
    run("lazy_distance_euclidean", check_lazy_distance_euclidean());
    run("lazy_distance_self_zero", check_lazy_distance_self_zero());
    run("lazy_distance_manhattan", check_lazy_distance_manhattan());
    run("lazy_no_cache_equals_cached", check_lazy_no_cache_equals_cached());
    run("sparse_distance_structure", check_sparse_distance_structure());
    run("sparse_distance_sparsity", check_sparse_distance_sparsity());
    run("sinkhorn_converges_1d", check_sinkhorn_converges_1d());
    run("sinkhorn_self_distance_low", check_sinkhorn_self_distance_low());
    run("sinkhorn_empty_diagrams", check_sinkhorn_empty_diagrams());
    run("sinkhorn_config_validation", check_sinkhorn_config_validation());

    return failures > 0 ? 1 : 0;
}
