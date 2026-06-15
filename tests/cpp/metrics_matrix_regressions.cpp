#include "nerve/core_types.hpp"
#include "nerve/metrics/gpu_distances.hpp"
#include "nerve/metrics/lazy_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{

using nerve::Size;

constexpr double kTol = 1e-10;

std::vector<double> make_point_cloud(size_t n, size_t dim, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> pts(n * dim);
    for (size_t i = 0; i < n * dim; ++i)
        pts[i] = dist(rng);
    return pts;
}

bool check_lazy_matrix_construction()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 5, 3);
    if (mat.getCacheSize() != 0)
    {
        std::cerr << "fresh cache should be empty\n";
        return false;
    }
    return true;
}

bool check_lazy_matrix_distance()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 5, 3);
    double d = mat.getDistance(0, 1);
    if (d < 0.0 || !std::isfinite(d))
    {
        std::cerr << "lazy distance invalid: " << d << "\n";
        return false;
    }
    return true;
}

bool check_lazy_matrix_symmetric()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 5, 3);
    double d01 = mat.getDistance(0, 1);
    double d10 = mat.getDistance(1, 0);
    if (std::abs(d01 - d10) > kTol)
    {
        std::cerr << "lazy distance not symmetric: " << d01 << " vs " << d10 << "\n";
        return false;
    }
    return true;
}

bool check_lazy_matrix_caching()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 5, 3, "euclidean", 100);
    mat.getDistance(0, 1);
    mat.getDistance(0, 1);
    auto rate = mat.getCacheHitRate();
    if (rate < 0.0 || rate > 1.0)
    {
        std::cerr << "cache hit rate out of range: " << rate << "\n";
        return false;
    }
    return true;
}

bool check_lazy_matrix_within_radius()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 5, 3);
    bool within = mat.isWithinRadius(0, 1, 10.0);
    if (!within)
    {
        std::cerr << "points should be within large radius\n";
        return false;
    }
    return true;
}

bool check_lazy_matrix_knn()
{
    auto pts = make_point_cloud(10, 2, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 10, 2);
    auto nn = mat.getKNearestNeighbors(0, 3);
    if (nn.empty())
    {
        std::cerr << "kNN should return at least 1 neighbor\n";
        return false;
    }
    if (nn.size() > 4)
    {
        std::cerr << "kNN returned too many results\n";
        return false;
    }
    for (const auto &n : nn)
    {
        if (n.second < 0.0)
        {
            std::cerr << "negative distance in kNN\n";
            return false;
        }
    }
    return true;
}

bool check_sparse_matrix_construction()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::SparseDistanceMatrix mat(span, 5, 3, 0.5);
    auto sparsity = mat.getSparsity();
    if (sparsity < 0.0 || sparsity > 1.0)
    {
        std::cerr << "sparsity out of range: " << sparsity << "\n";
        return false;
    }
    return true;
}

bool check_sparse_matrix_is_edge()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::SparseDistanceMatrix mat(span, 5, 3, 10.0);
    bool edge = mat.isEdge(0, 1);
    if (!edge)
    {
        std::cerr << "edge should exist with large threshold\n";
        return false;
    }
    return true;
}

bool check_sparse_matrix_distance()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::SparseDistanceMatrix mat(span, 5, 3, 10.0);
    double d = mat.getDistance(0, 1);
    if (d < 0.0 || !std::isfinite(d))
    {
        std::cerr << "sparse distance invalid\n";
        return false;
    }
    return true;
}

bool check_lazy_equals_pairwise()
{
    auto pts = make_point_cloud(5, 2, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix lazy(span, 5, 2);
    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = i + 1; j < 5; ++j)
        {
            double lazy_d = lazy.getDistanceNoCache(i, j);
            double direct = 0.0;
            for (size_t k = 0; k < 2; ++k)
            {
                double diff = pts[i * 2 + k] - pts[j * 2 + k];
                direct += diff * diff;
            }
            direct = std::sqrt(direct);
            if (std::abs(lazy_d - direct) > kTol)
            {
                std::cerr << "lazy(" << i << "," << j << ")=" << lazy_d << " != direct=" << direct
                          << "\n";
                return false;
            }
        }
    }
    return true;
}

#ifdef NERVE_HAS_AVX512
bool check_avx512_available()
{
    bool avail = nerve::metrics::avx512::isAVX512Available();
    (void)avail;
    return true;
}

bool check_avx512_compute_matrix()
{
    float birth1[] = {0.0f, 0.2f, 0.5f};
    float death1[] = {1.0f, 0.8f, 1.5f};
    float birth2[] = {0.1f, 0.3f};
    float death2[] = {1.2f, 0.9f};
    float out[6] = {0};
    nerve::metrics::avx512::avx512DiagramDistanceMatrix(birth1, death1, 3, birth2, death2, 2, out);
    for (int i = 0; i < 6; ++i)
    {
        if (!std::isfinite(out[i]))
        {
            std::cerr << "AVX512 matrix produced non-finite value\n";
            return false;
        }
    }
    return true;
}
#endif

bool check_lazy_clear_cache()
{
    auto pts = make_point_cloud(5, 3, 42);
    std::span<const double> span(pts.data(), pts.size());
    nerve::metrics::lazy::LazyDistanceMatrix mat(span, 5, 3);
    mat.getDistance(0, 1);
    mat.getDistance(0, 2);
    mat.clearCache();
    if (mat.getCacheSize() != 0)
    {
        std::cerr << "cache should be empty after clear\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_lazy_matrix_construction())
    {
        std::cerr << "FAIL: lazy construction\n";
        return 1;
    }
    if (!check_lazy_matrix_distance())
    {
        std::cerr << "FAIL: lazy distance\n";
        return 1;
    }
    if (!check_lazy_matrix_symmetric())
    {
        std::cerr << "FAIL: lazy symmetric\n";
        return 1;
    }
    if (!check_lazy_matrix_caching())
    {
        std::cerr << "FAIL: lazy caching\n";
        return 1;
    }
    if (!check_lazy_matrix_within_radius())
    {
        std::cerr << "FAIL: lazy within radius\n";
        return 1;
    }
    if (!check_lazy_matrix_knn())
    {
        std::cerr << "FAIL: lazy knn\n";
        return 1;
    }
    if (!check_sparse_matrix_construction())
    {
        std::cerr << "FAIL: sparse construction\n";
        return 1;
    }
    if (!check_sparse_matrix_is_edge())
    {
        std::cerr << "FAIL: sparse is edge\n";
        return 1;
    }
    if (!check_sparse_matrix_distance())
    {
        std::cerr << "FAIL: sparse distance\n";
        return 1;
    }
    if (!check_lazy_equals_pairwise())
    {
        std::cerr << "FAIL: lazy equals pairwise\n";
        return 1;
    }
    if (!check_lazy_clear_cache())
    {
        std::cerr << "FAIL: lazy clear cache\n";
        return 1;
    }
#ifdef NERVE_HAS_AVX512
    if (!check_avx512_available())
    {
        std::cerr << "FAIL: avx512 available\n";
        return 1;
    }
    if (!check_avx512_compute_matrix())
    {
        std::cerr << "FAIL: avx512 compute\n";
        return 1;
    }
#endif
    return 0;
}
