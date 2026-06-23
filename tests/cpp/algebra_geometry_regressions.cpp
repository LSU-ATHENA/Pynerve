
#include "nerve/algebra/simd_distance.hpp"
#include "nerve/algebra/simd_distance_avx.hpp"
#include "nerve/core_types.hpp"

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
using nerve::algebra::EnhancedSIMDCalculator;
using nerve::algebra::SIMDDistanceCalculator;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(1729);
}

bool check_euclidean_distance_known()
{
    SIMDDistanceCalculator calc;
    double a[] = {0.0, 0.0};
    double b[] = {3.0, 4.0};
    double d = calc.euclideanDistance(a, b, 2);
    if (std::abs(d - 5.0) > TOL)
    {
        std::cerr << "expected 5.0, got " << d << "\n";
        return false;
    }
    return true;
}

bool check_manhattan_distance_known()
{
    SIMDDistanceCalculator calc;
    double a[] = {0.0, 0.0};
    double b[] = {3.0, 4.0};
    double d = calc.manhattanDistance(a, b, 2);
    if (std::abs(d - 7.0) > TOL)
    {
        std::cerr << "expected 7.0, got " << d << "\n";
        return false;
    }
    return true;
}

bool check_cosine_distance_known()
{
    SIMDDistanceCalculator calc;
    double a[] = {1.0, 0.0};
    double b[] = {0.0, 1.0};
    double d = calc.cosineDistance(a, b, 2);
    if (std::abs(d - 1.0) > TOL)
    {
        std::cerr << "expected 1.0, got " << d << "\n";
        return false;
    }
    return true;
}

bool check_simd_vs_scalar_equivalence()
{
    SIMDDistanceCalculator calc;
    std::vector<double> pts(100);
    auto rng = make_rng();
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (auto &p : pts)
        p = dist(rng);
    double a[] = {dist(rng), dist(rng)};
    double d_simd = calc.euclideanDistance(a, pts.data(), 2);
    double expected =
        std::sqrt((a[0] - pts[0]) * (a[0] - pts[0]) + (a[1] - pts[1]) * (a[1] - pts[1]));
    if (std::abs(d_simd - expected) > TOL)
    {
        std::cerr << "SIMD distance mismatch: " << d_simd << " vs " << expected << "\n";
        return false;
    }
    return true;
}

bool check_batch_euclidean_distances()
{
    SIMDDistanceCalculator calc;
    double pts[] = {0.0, 0.0, 3.0, 4.0, 0.0, 1.0};
    auto distances = calc.batchEuclideanDistances(pts, 3, 2);
    if (distances.size() != 3)
    {
        std::cerr << "expected 3 distances, got " << distances.size() << "\n";
        return false;
    }
    if (std::abs(distances[0] - 5.0) > TOL)
    {
        std::cerr << "first batch distance should be 5.0, got " << distances[0] << "\n";
        return false;
    }
    return true;
}

bool check_enhanced_compute_matrix()
{
    EnhancedSIMDCalculator calc;
    double pts[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    auto result = calc.computeDistanceMatrix(pts, 3, 2);
    if (result.isError())
    {
        std::cerr << "computeDistanceMatrix error: " << result.compactSummary() << "\n";
        return false;
    }
    auto mat = result.value();
    if (mat.size() != 9)
    {
        std::cerr << "expected 9 entries, got " << mat.size() << "\n";
        return false;
    }
    return true;
}

bool check_enhanced_compressed_matrix()
{
    EnhancedSIMDCalculator calc;
    double pts[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    auto result = calc.computeCompressedMatrix(pts, 4, 2);
    if (result.isError())
    {
        std::cerr << "computeCompressedMatrix error: " << result.compactSummary() << "\n";
        return false;
    }
    auto mat = result.value();
    if (mat.empty())
    {
        std::cerr << "compressed matrix cannot be empty\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_euclidean_distance_known())
    {
        std::cerr << "FAIL: euclidean distance known\n";
        return 1;
    }
    if (!check_manhattan_distance_known())
    {
        std::cerr << "FAIL: manhattan distance known\n";
        return 1;
    }
    if (!check_cosine_distance_known())
    {
        std::cerr << "FAIL: cosine distance known\n";
        return 1;
    }
    if (!check_simd_vs_scalar_equivalence())
    {
        std::cerr << "FAIL: simd vs scalar\n";
        return 1;
    }
    if (!check_batch_euclidean_distances())
    {
        std::cerr << "FAIL: batch euclidean\n";
        return 1;
    }
    if (!check_enhanced_compute_matrix())
    {
        std::cerr << "FAIL: enhanced compute matrix\n";
        return 1;
    }
    if (!check_enhanced_compressed_matrix())
    {
        std::cerr << "FAIL: enhanced compressed\n";
        return 1;
    }
    return 0;
}
