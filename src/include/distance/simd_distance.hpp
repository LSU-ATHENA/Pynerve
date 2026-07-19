/// @file simd_distance.hpp
/// Distance computation utilities using the global SIMD dispatch table.
/// The dispatch table (nerve::simd::SIMD) is initialized once via simd_init()
/// and selects the best available ISA (AVX-512, AVX2, SSE4.1, NEON, scalar).
#pragma once
#include "nerve/core_types.hpp"
#include "nerve/simd/simd_base.hpp"
#include "nerve/simd/simd_distance.hpp"
#include "nerve/simd/simd_reduce.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace nerve::distance
{

// Stores only upper triangle: n*(n-1)/2 doubles
class DistanceMatrix
{
public:
    explicit DistanceMatrix(nerve::Size n)
        : n_(n)
        , data_(checkedUpperTriangleSize(n), 0.0)
    {}

    double operator()(nerve::Size i, nerve::Size j) const noexcept
    {
        if (i == j)
            return 0.0;
        if (i > j)
            std::swap(i, j);
        return data_[index(i, j)];
    }

    double &operator()(nerve::Size i, nerve::Size j) noexcept
    {
        assert(i != j);
        if (i > j)
            std::swap(i, j);
        return data_[index(i, j)];
    }

    nerve::Size size() const noexcept { return n_; }

private:
    nerve::Size n_;
    std::vector<double> data_;

    static nerve::Size checkedUpperTriangleSize(nerve::Size n)
    {
        if (n < 2)
            return 0;
        if (n > std::numeric_limits<nerve::Size>::max() / (n - 1))
            throw std::length_error("distance matrix upper-triangle size overflows");
        const nerve::Size count = (n * (n - 1)) / 2;
        if (count > std::vector<double>().max_size())
            throw std::length_error("distance matrix upper-triangle size exceeds vector capacity");
        return count;
    }

    nerve::Size index(nerve::Size i, nerve::Size j) const noexcept
    {
        return i * n_ - (i * (i + 1)) / 2 + j - i - 1;
    }
};

/// Runtime CPU capability queries -- delegates to the global dispatch table.
struct CpuCapabilities
{
    static bool hasAvx512() noexcept
    {
        return nerve::simd::detect_simd_arch() == nerve::simd::SimdArch::AVX512;
    }
    static bool hasAvx2() noexcept
    {
        return nerve::simd::detect_simd_arch() == nerve::simd::SimdArch::AVX2;
    }
    static bool hasSse4() noexcept
    {
        return nerve::simd::detect_simd_arch() == nerve::simd::SimdArch::SSE41;
    }
    static bool hasFma() noexcept
    {
        // FMA is implied by AVX2 and AVX-512 architectures
        auto arch = nerve::simd::detect_simd_arch();
        return arch == nerve::simd::SimdArch::AVX2 || arch == nerve::simd::SimdArch::AVX512;
    }
};

/// Pure scalar Euclidean distance -- no SIMD, deterministic.
/// Used as the correctness reference and for benchmarking.
inline double euclideanScalar(const double *a, const double *b, nerve::Size dim) noexcept
{
    double sum = 0.0;
    for (nerve::Size k = 0; k < dim; ++k)
    {
        double d = a[k] - b[k];
        sum += d * d;
    }
    return std::sqrt(sum);
}

/// Euclidean distance using the SIMD dispatch table (best available ISA).
inline double euclidean(const double *a, const double *b, nerve::Size dim) noexcept
{
    return std::sqrt(nerve::simd::SIMD.sqdiff_sum(a, b, static_cast<std::size_t>(dim)));
}

/// DistanceComputer -- picks the best ISA via the dispatch table.
///
/// Unlike the old implementation that selected a function pointer at construction,
/// this now delegates to nerve::simd::simd_euclidean() which goes through the
/// global dispatch table. The indirection cost is negligible (single function
/// pointer call) and eliminates all #ifdef and CPUID duplication.
class DistanceComputer
{
public:
    using DistFn = double (*)(const double *, const double *, nerve::Size);

    DistanceComputer() { nerve::simd::simd_init(); }

    // Hot path: delegates to the dispatch table
    double compute(const double *a, const double *b, nerve::Size dim) const noexcept
    {
        return nerve::simd::simd_euclidean(a, b, static_cast<std::size_t>(dim));
    }

    DistanceMatrix buildMatrix(const double *points, nerve::Size n_points, nerve::Size dim,
                               nerve::Size n_threads = 1) const
    {
        if (points == nullptr || n_points == 0 || dim == 0)
            return DistanceMatrix(n_points);
        if (n_points > std::numeric_limits<nerve::Size>::max() / dim)
            throw std::length_error("distance matrix point coordinate count overflows");

        DistanceMatrix mat(n_points);
        const nerve::Size coordinate_count = n_points * dim;
        for (nerve::Size k = 0; k < coordinate_count; ++k)
            if (!std::isfinite(points[k]))
                throw std::invalid_argument("distance matrix coordinates must be finite");

        if (n_threads == 1)
        {
            for (nerve::Size i = 0; i < n_points; ++i)
            {
                const double *pi = points + i * dim;
                for (nerve::Size j = i + 1; j < n_points; ++j)
                {
                    double d = compute(pi, points + j * dim, dim);
                    if (!std::isfinite(d))
                        throw std::overflow_error(
                            "distance matrix computation produced a non-finite value");
                    mat(i, j) = d;
                }
            }
        }
        else
        {
            const nerve::Size worker_count =
                std::max<nerve::Size>(1, std::min<nerve::Size>(n_threads, n_points));
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            std::exception_ptr first_exception = nullptr;
            std::mutex exception_mutex;

            for (nerve::Size w = 0; w < worker_count; ++w)
            {
                workers.emplace_back([&, w]() {
                    try
                    {
                        for (nerve::Size i = w; i < n_points; i += worker_count)
                        {
                            const double *pi = points + i * dim;
                            for (nerve::Size j = i + 1; j < n_points; ++j)
                            {
                                double d = compute(pi, points + j * dim, dim);
                                if (!std::isfinite(d))
                                    throw std::overflow_error(
                                        "distance matrix computation produced a non-finite value");
                                mat(i, j) = d;
                            }
                        }
                    }
                    catch (...)
                    {
                        std::lock_guard<std::mutex> lock(exception_mutex);
                        if (!first_exception)
                            first_exception = std::current_exception();
                    }
                });
            }

            for (auto &worker : workers)
                if (worker.joinable())
                    worker.join();

            if (first_exception)
                std::rethrow_exception(first_exception);
        }

        return mat;
    }

    void computeBatch(const double *points, nerve::Size n_points, nerve::Size dim,
                      const std::vector<std::pair<nerve::Size, nerve::Size>> &pairs,
                      std::vector<double> &results) const
    {
        if (points == nullptr || dim == 0)
            throw std::invalid_argument("distance batch points must be non-null with positive dim");
        for (const auto &[i, j] : pairs)
            if (i >= n_points || j >= n_points)
                throw std::out_of_range("distance batch pair index exceeds point count");

        std::vector<double> computed;
        computed.reserve(pairs.size());
        for (nerve::Size k = 0; k < pairs.size(); ++k)
        {
            auto [i, j] = pairs[k];
            const double *pi = points + i * dim;
            const double *pj = points + j * dim;
            for (nerve::Size d = 0; d < dim; ++d)
                if (!std::isfinite(pi[d]) || !std::isfinite(pj[d]))
                    throw std::invalid_argument("distance batch coordinates must be finite");

            const double distance = compute(pi, pj, dim);
            if (!std::isfinite(distance))
                throw std::overflow_error("distance batch computation produced a non-finite value");
            computed.push_back(distance);
        }
        results = std::move(computed);
    }
};

inline std::unique_ptr<DistanceComputer> createDistanceComputer()
{
    return std::make_unique<DistanceComputer>();
}

struct DistanceBenchmark
{
    nerve::Size n_points;
    nerve::Size dim;
    double scalar_time_ms;
    double simd_time_ms;
    double speedup;

    static double finiteSpeedup(double baseline_ms, double accelerated_ms)
    {
        if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
            accelerated_ms <= 0.0)
            return 1.0;
        const double measured_speedup = baseline_ms / accelerated_ms;
        return std::isfinite(measured_speedup) && measured_speedup >= 0.0 ? measured_speedup : 1.0;
    }

    static DistanceBenchmark run(const double *points, nerve::Size n_points, nerve::Size dim,
                                 nerve::Size n_iterations = 10)
    {
        if (n_iterations == 0)
            throw std::invalid_argument("distance benchmark iterations must be positive");
        if (points == nullptr && n_points > 1)
            throw std::invalid_argument("distance benchmark points pointer is null");

        nerve::simd::simd_init();
        DistanceComputer simd_comp;

        // Benchmark scalar version -- pure double loop, no SIMD
        auto start = std::chrono::high_resolution_clock::now();
        for (nerve::Size iter = 0; iter < n_iterations; ++iter)
        {
            for (nerve::Size i = 0; i < n_points; ++i)
            {
                for (nerve::Size j = i + 1; j < n_points; ++j)
                {
                    const double *pi = points + i * dim;
                    const double *pj = points + j * dim;
                    double s = 0.0;
                    for (nerve::Size k = 0; k < dim; ++k)
                    {
                        double d = pi[k] - pj[k];
                        s += d * d;
                    }
                    // consume result to prevent optimization
                    volatile double sink = s;
                    (void)sink;
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        // Benchmark SIMD version
        start = std::chrono::high_resolution_clock::now();
        for (nerve::Size iter = 0; iter < n_iterations; ++iter)
        {
            for (nerve::Size i = 0; i < n_points; ++i)
            {
                for (nerve::Size j = i + 1; j < n_points; ++j)
                {
                    simd_comp.compute(points + i * dim, points + j * dim, dim);
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        double simd_time = std::chrono::duration<double, std::milli>(end - start).count();
        const double iteration_count = static_cast<double>(n_iterations);

        return {.n_points = n_points,
                .dim = dim,
                .scalar_time_ms = scalar_time / iteration_count,
                .simd_time_ms = simd_time / iteration_count,
                .speedup = finiteSpeedup(scalar_time, simd_time)};
    }
};

} // namespace nerve::distance
