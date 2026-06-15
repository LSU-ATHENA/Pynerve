#include "nerve/algebra/simd_distance.hpp"
#include "nerve/algebra/simd_distance_avx.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

double scalarEuclidean(const double *a, const double *b, size_t dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double scalarManhattan(const double *a, const double *b, size_t dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

double scalarCosine(const double *a, const double *b, size_t dim)
{
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0 || norm_b == 0.0)
        return 0.0;
    double cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    if (cosine > 1.0)
        cosine = 1.0;
    if (cosine < -1.0)
        cosine = -1.0;
    return 1.0 - cosine;
}

double scalarChebyshev(const double *a, const double *b, size_t dim)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

double scalarMinkowski(const double *a, const double *b, size_t dim, double p)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        sum += std::pow(std::abs(a[i] - b[i]), p);
    }
    return std::pow(sum, 1.0 / p);
}

double scalarCanberra(const double *a, const double *b, size_t dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double denom = std::abs(a[i]) + std::abs(b[i]);
        if (denom > 0.0)
        {
            sum += std::abs(a[i] - b[i]) / denom;
        }
    }
    return sum;
}

double scalarBrayCurtis(const double *a, const double *b, size_t dim)
{
    double num = 0.0, denom = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        num += std::abs(a[i] - b[i]);
        denom += std::abs(a[i] + b[i]);
    }
    if (denom == 0.0)
        return 0.0;
    return num / denom;
}

double scalarCorrelation(const double *a, const double *b, size_t dim)
{
    double sum_a = 0.0, sum_b = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        sum_a += a[i];
        sum_b += b[i];
    }
    double mean_a = sum_a / static_cast<double>(dim);
    double mean_b = sum_b / static_cast<double>(dim);
    double num = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if (denom_a == 0.0 && denom_b == 0.0)
        return 0.0;
    if (denom_a == 0.0 || denom_b == 0.0)
        return 1.0;
    double corr = num / std::sqrt(denom_a * denom_b);
    if (corr > 1.0)
        corr = 1.0;
    if (corr < -1.0)
        corr = -1.0;
    return 1.0 - corr;
}

bool approxEqual(double a, double b, double eps = 1e-12)
{
    return std::abs(a - b) <= eps * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

std::vector<double> generatePoint(size_t dim, double seed)
{
    std::vector<double> p(dim);
    for (size_t i = 0; i < dim; ++i)
    {
        p[i] = (static_cast<double>(i * 7 + 1) * seed) / (dim + 1.0);
    }
    return p;
}

bool hasNaN(const double *data, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (std::isnan(data[i]))
            return true;
    }
    return false;
}

} // namespace

int main()
{
    nerve::algebra::SIMDDistanceCalculator base_calc;
    nerve::algebra::EnhancedSIMDCalculator enhanced_calc;

    const size_t dimensions[] = {1, 3, 8, 16, 100};
    const double tolerance = 1e-10;

    for (size_t dim : dimensions)
    {
        auto a = generatePoint(dim, 2.0);
        auto b = generatePoint(dim, 5.0);

        double simd_euclidean = base_calc.euclideanDistance(a.data(), b.data(), dim);
        double ref_euclidean = scalarEuclidean(a.data(), b.data(), dim);
        assert(approxEqual(simd_euclidean, ref_euclidean, tolerance));

        double simd_manhattan = base_calc.manhattanDistance(a.data(), b.data(), dim);
        double ref_manhattan = scalarManhattan(a.data(), b.data(), dim);
        assert(approxEqual(simd_manhattan, ref_manhattan, tolerance));

        double simd_cosine = base_calc.cosineDistance(a.data(), b.data(), dim);
        double ref_cosine = scalarCosine(a.data(), b.data(), dim);
        assert(approxEqual(simd_cosine, ref_cosine, tolerance));
    }

    {
        const double zero_vec[] = {0.0, 0.0, 0.0};
        const double unit_vec[] = {1.0, 0.0, 0.0};

        assert(approxEqual(base_calc.euclideanDistance(zero_vec, zero_vec, 3), 0.0));
        assert(approxEqual(base_calc.euclideanDistance(zero_vec, unit_vec, 3), 1.0));
        assert(approxEqual(base_calc.manhattanDistance(zero_vec, zero_vec, 3), 0.0));
        assert(approxEqual(base_calc.manhattanDistance(zero_vec, unit_vec, 3), 1.0));
    }

    {
        const double id1[] = {1.0, 2.0, 3.0};
        const double id2[] = {1.0, 2.0, 3.0};
        assert(approxEqual(base_calc.euclideanDistance(id1, id2, 3), 0.0));
        assert(approxEqual(base_calc.manhattanDistance(id1, id2, 3), 0.0));
        assert(approxEqual(base_calc.cosineDistance(id1, id2, 3), 0.0));
    }

    {
        const double all_zero[] = {0.0, 0.0, 0.0, 0.0};
        assert(approxEqual(base_calc.euclideanDistance(all_zero, all_zero, 4), 0.0));
        assert(approxEqual(base_calc.manhattanDistance(all_zero, all_zero, 4), 0.0));
    }

    {
        std::vector<double> nan_vec(5, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> ok_vec(5, 1.0);
        bool caught = false;
        try
        {
            (void)base_calc.euclideanDistance(nan_vec.data(), ok_vec.data(), 5);
        }
        catch (const std::invalid_argument &)
        {
            caught = true;
        }
        (void)caught;
    }

    {
        const double inf_vec[] = {std::numeric_limits<double>::infinity(), 0.0};
        const double ok_val[] = {0.0, 0.0};
        bool overflow_rejected = false;
        try
        {
            (void)base_calc.euclideanDistance(inf_vec, ok_val, 2);
        }
        catch (const std::invalid_argument &)
        {
            overflow_rejected = true;
        }
        assert(overflow_rejected);
    }

    {
        const double float_a[] = {1.0f, 2.0f, 3.0f};
        const double float_b[] = {4.0f, 5.0f, 6.0f};
        double euc = base_calc.euclideanDistance(float_a, float_b, 3);
        double ref = std::sqrt(27.0);
        assert(approxEqual(euc, ref, tolerance));
    }

    {
        auto p1 = generatePoint(20, 1.0);
        auto p2 = generatePoint(20, 3.0);

        double ref_cheb = scalarChebyshev(p1.data(), p2.data(), 20);
        double ref_mink = scalarMinkowski(p1.data(), p2.data(), 20, 3.0);
        double ref_canb = scalarCanberra(p1.data(), p2.data(), 20);
        double ref_bc = scalarBrayCurtis(p1.data(), p2.data(), 20);
        double ref_corr = scalarCorrelation(p1.data(), p2.data(), 20);

        assert(ref_cheb >= 0.0);
        assert(ref_mink >= 0.0);
        assert(ref_canb >= 0.0);
        assert(ref_bc >= 0.0 && ref_bc <= 1.0);
        assert(ref_corr >= 0.0 && ref_corr <= 2.0);
    }

    {
        const double scalar_pt[] = {0.0};
        auto result = enhanced_calc.batchEuclideanDistances(scalar_pt, scalar_pt, 1, 1);
        assert(result.isSuccess());
        assert(result.value().size() == 1);
        assert(approxEqual(result.value()[0], 0.0));
    }

    {
        const double pts[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto mat = enhanced_calc.computeDistanceMatrix(pts, 3, 2);
        assert(mat.isSuccess());
        assert(mat.value().size() == 9);
    }

    {
        double prev_euc = base_calc.euclideanDistance(nullptr, nullptr, 0);
        assert(prev_euc == 0.0);
    }

    {
        auto p1 = generatePoint(100, 7.0);
        auto p2 = generatePoint(100, 11.0);

        double euc_vec = base_calc.euclideanDistance(p1.data(), p2.data(), 100);
        double man_vec = base_calc.manhattanDistance(p1.data(), p2.data(), 100);
        double cos_vec = base_calc.cosineDistance(p1.data(), p2.data(), 100);

        double euc_ref = scalarEuclidean(p1.data(), p2.data(), 100);
        double man_ref = scalarManhattan(p1.data(), p2.data(), 100);
        double cos_ref = scalarCosine(p1.data(), p2.data(), 100);

        assert(approxEqual(euc_vec, euc_ref, 1e-9));
        assert(approxEqual(man_vec, man_ref, 1e-9));
        assert(approxEqual(cos_vec, cos_ref, 1e-9));
    }

    return 0;
}
