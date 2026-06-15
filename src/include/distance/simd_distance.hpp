
/// @file simd_distance.hpp
/// The runtime detects the best available ISA once (AVX-512, AVX2, SSE4.1, scalar)
/// and reuses that dispatch choice for all subsequent distance calls.
#pragma once
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
#if defined(__x86_64__) || defined(__i386__)
#include "nerve/cpu/x86_intrinsics.hpp"
#endif
#include "nerve/core_types.hpp"

namespace nerve::distance
{

inline double euclideanAvx2Unrolled(const double *a, const double *b, nerve::Size dim) noexcept;

// Stores only upper triangle: n*(n-1)/2 doubles
// Access: d(i,j) with i<j -> index = i*n - i*(i+1)/2 + j - i - 1
// Memory: O(n^2/2) instead of O(n^2)  --  halves memory usage
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
        {
            return 0;
        }
        if (n > std::numeric_limits<nerve::Size>::max() / (n - 1))
        {
            throw std::length_error("distance matrix upper-triangle size overflows");
        }
        const nerve::Size count = (n * (n - 1)) / 2;
        if (count > std::vector<double>().max_size())
        {
            throw std::length_error("distance matrix upper-triangle size exceeds vector capacity");
        }
        return count;
    }

    nerve::Size index(nerve::Size i, nerve::Size j) const noexcept
    {
        // i < j guaranteed by caller
        return i * n_ - (i * (i + 1)) / 2 + j - i - 1;
    }
};

class CpuCapabilities
{
public:
    static bool hasAvx512() noexcept
    {
#if defined(__AVX512F__) && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("avx512f");
#else
        return true;
#endif
#else
        return false;
#endif
    }

    static bool hasAvx2() noexcept
    {
#if defined(__AVX2__) && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("avx2");
#else
        return true;
#endif
#else
        return false;
#endif
    }

    static bool hasSse4() noexcept
    {
#if defined(__SSE4_1__) && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("sse4.1");
#else
        return true;
#endif
#else
        return false;
#endif
    }

    static bool hasFma() noexcept
    {
#if defined(__FMA__) && (defined(__x86_64__) || defined(__i386__))
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("fma");
#else
        return true;
#endif
#else
        return false;
#endif
    }
};

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

inline double euclideanSse4(const double *a, const double *b, nerve::Size dim) noexcept
{
#ifdef __SSE4_1__
    __m128d sum = _mm_setzero_pd();
    nerve::Size k = 0;

    for (; k + 1 < dim; k += 2)
    {
        __m128d va = _mm_loadu_pd(a + k);
        __m128d vb = _mm_loadu_pd(b + k);
        __m128d diff = _mm_sub_pd(va, vb);
        __m128d sq = _mm_mul_pd(diff, diff);
        sum = _mm_add_pd(sum, sq);
    }

    sum = _mm_hadd_pd(sum, sum);
    double result = _mm_cvtsd_f64(sum);

    // Handle remaining element
    for (; k < dim; ++k)
    {
        double d = a[k] - b[k];
        result += d * d;
    }

    return std::sqrt(result);
#else
    return euclideanScalar(a, b, dim);
#endif
}

inline double euclideanAvx512Unrolled(const double *a, const double *b, nerve::Size dim) noexcept
{
#ifdef __AVX512F__
    __m512d sum0 = _mm512_setzero_pd();
    __m512d sum1 = _mm512_setzero_pd();
    nerve::Size k = 0;

    // Prefetch 4 cache lines ahead = 32 doubles = 256 bytes
    constexpr int PREFETCH_DIST = 32;

    for (; k + 16 <= dim; k += 16)
    {
        _mm_prefetch(reinterpret_cast<const char *>(a + k + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(b + k + PREFETCH_DIST), _MM_HINT_T0);

        __m512d va0 = _mm512_loadu_pd(a + k);
        __m512d vb0 = _mm512_loadu_pd(b + k);
        __m512d va1 = _mm512_loadu_pd(a + k + 8);
        __m512d vb1 = _mm512_loadu_pd(b + k + 8);

        __m512d d0 = _mm512_sub_pd(va0, vb0);
        __m512d d1 = _mm512_sub_pd(va1, vb1);

#ifdef __FMA__
        sum0 = _mm512_fmadd_pd(d0, d0, sum0);
        sum1 = _mm512_fmadd_pd(d1, d1, sum1);
#else
        __m512d sq0 = _mm512_mul_pd(d0, d0);
        __m512d sq1 = _mm512_mul_pd(d1, d1);
        sum0 = _mm512_add_pd(sq0, sum0);
        sum1 = _mm512_add_pd(sq1, sum1);
#endif
    }

    __m512d sum8 = _mm512_add_pd(sum0, sum1);

    double sum = _mm512_reduce_add_pd(sum8);

    for (; k < dim; ++k)
    {
        double d = a[k] - b[k];
        sum += d * d;
    }
    return std::sqrt(sum);
#else
    return euclideanAvx2Unrolled(a, b, dim);
#endif
}

inline double euclideanAvx2Unrolled(const double *a, const double *b, nerve::Size dim) noexcept;
inline double euclideanAvx2Unrolled(const double *a, const double *b, nerve::Size dim) noexcept
{
#ifdef __AVX2__
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd(); // 2 accumulators -> breaks dep chain
    nerve::Size k = 0;

    // Prefetch distance: 2 cache lines ahead = 16 doubles = 128 bytes
    constexpr int PREFETCH_DIST = 16;

    for (; k + 8 <= dim; k += 8)
    {
        // Issue prefetch for 2 iterations ahead  --  hides memory latency
        _mm_prefetch(reinterpret_cast<const char *>(a + k + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(b + k + PREFETCH_DIST), _MM_HINT_T0);

        __m256d va0 = _mm256_loadu_pd(a + k);
        __m256d vb0 = _mm256_loadu_pd(b + k);
        __m256d va1 = _mm256_loadu_pd(a + k + 4);
        __m256d vb1 = _mm256_loadu_pd(b + k + 4);

        __m256d d0 = _mm256_sub_pd(va0, vb0);
        __m256d d1 = _mm256_sub_pd(va1, vb1);

#ifdef __FMA__
        // FMA: sum += d*d  (one instruction, no extra rounding)
        sum0 = _mm256_fmadd_pd(d0, d0, sum0);
        sum1 = _mm256_fmadd_pd(d1, d1, sum1);
#else
        __m256d sq0 = _mm256_mul_pd(d0, d0);
        __m256d sq1 = _mm256_mul_pd(d1, d1);
        sum0 = _mm256_add_pd(sq0, sum0);
        sum1 = _mm256_add_pd(sq1, sum1);
#endif
    }

    __m256d sum4 = _mm256_add_pd(sum0, sum1);

    __m128d lo = _mm256_castpd256_pd128(sum4);
    __m128d hi = _mm256_extractf128_pd(sum4, 1);
    __m128d s2 = _mm_add_pd(lo, hi);
    __m128d s1 = _mm_hadd_pd(s2, s2);
    double sum = _mm_cvtsd_f64(s1);

    for (; k < dim; ++k)
    {
        double d = a[k] - b[k];
        sum += d * d;
    }
    return std::sqrt(sum);
#else
    return euclideanSse4(a, b, dim);
#endif
}

class DistanceComputer
{
public:
    using DistFn = double (*)(const double *, const double *, nerve::Size);

    DistanceComputer()
    {
        // Detect CPU capability ONCE at construction
        // No branching in hot path  --  pick the best available ISA
        if (CpuCapabilities::hasAvx512())
        {
            dist_fn_ = &euclideanAvx512Unrolled;
        }
        else if (CpuCapabilities::hasAvx2())
        {
            dist_fn_ = &euclideanAvx2Unrolled;
        }
        else if (CpuCapabilities::hasSse4())
        {
            dist_fn_ = &euclideanSse4;
        }
        else
        {
            dist_fn_ = &euclideanScalar;
        }
    }

    // Hot path: zero branches on capability
    double compute(const double *a, const double *b, nerve::Size dim) const noexcept
    {
        return dist_fn_(a, b, dim);
    }

    // Build full distance matrix: O(n^2 * dim)
    // Parallel over rows  --  each row is independent
    DistanceMatrix
    buildMatrix(const double *points, // row-major: points[i*dim + k] = point i, coord k
                nerve::Size n_points, nerve::Size dim, nerve::Size n_threads = 1) const
    {
        if (points == nullptr || n_points == 0 || dim == 0)
        {
            return DistanceMatrix(n_points);
        }
        if (n_points > std::numeric_limits<nerve::Size>::max() / dim)
        {
            throw std::length_error("distance matrix point coordinate count overflows");
        }

        DistanceMatrix mat(n_points);
        const nerve::Size coordinate_count = n_points * dim;
        for (nerve::Size k = 0; k < coordinate_count; ++k)
        {
            if (!std::isfinite(points[k]))
            {
                throw std::invalid_argument("distance matrix coordinates must be finite");
            }
        }

        // Cache-friendly: for each pair (i,j) with i<j, compute once
        // Loop order: outer=i, inner=j (row major  --  good cache behavior)

        if (n_threads == 1)
        {
            // Single-threaded version
            for (nerve::Size i = 0; i < n_points; ++i)
            {
                const double *pi = points + i * dim;
                for (nerve::Size j = i + 1; j < n_points; ++j)
                {
                    const double *pj = points + j * dim;
                    const double distance = compute(pi, pj, dim);
                    if (!std::isfinite(distance))
                    {
                        throw std::overflow_error(
                            "distance matrix computation produced a non-finite value");
                    }
                    mat(i, j) = distance;
                }
            }
        }
        else
        {
            // Multi-threaded row-partitioned execution with bounded workers.
            const nerve::Size worker_count =
                std::max<nerve::Size>(1, std::min<nerve::Size>(n_threads, n_points));
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            std::exception_ptr first_exception = nullptr;
            std::mutex exception_mutex;

            for (nerve::Size worker = 0; worker < worker_count; ++worker)
            {
                workers.emplace_back([&, worker]() {
                    try
                    {
                        for (nerve::Size i = worker; i < n_points; i += worker_count)
                        {
                            const double *pi = points + i * dim;
                            for (nerve::Size j = i + 1; j < n_points; ++j)
                            {
                                const double *pj = points + j * dim;
                                const double distance = compute(pi, pj, dim);
                                if (!std::isfinite(distance))
                                {
                                    throw std::overflow_error(
                                        "distance matrix computation produced a non-finite value");
                                }
                                mat(i, j) = distance;
                            }
                        }
                    }
                    catch (...)
                    {
                        std::lock_guard<std::mutex> lock(exception_mutex);
                        if (!first_exception)
                        {
                            first_exception = std::current_exception();
                        }
                    }
                });
            }

            for (auto &worker : workers)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }

            if (first_exception)
            {
                std::rethrow_exception(first_exception);
            }
        }

        return mat;
    }

    // Batch computation: compute multiple distances at once
    void computeBatch(const double *points, nerve::Size n_points, nerve::Size dim,
                      const std::vector<std::pair<nerve::Size, nerve::Size>> &pairs,
                      std::vector<double> &results) const
    {
        if (points == nullptr || dim == 0)
        {
            throw std::invalid_argument("distance batch points must be non-null with positive dim");
        }
        for (const auto &[i, j] : pairs)
        {
            if (i >= n_points || j >= n_points)
            {
                throw std::out_of_range("distance batch pair index exceeds point count");
            }
        }

        std::vector<double> computed;
        computed.reserve(pairs.size());
        for (nerve::Size k = 0; k < pairs.size(); ++k)
        {
            auto [i, j] = pairs[k];
            const double *pi = points + i * dim;
            const double *pj = points + j * dim;
            for (nerve::Size d = 0; d < dim; ++d)
            {
                if (!std::isfinite(pi[d]) || !std::isfinite(pj[d]))
                {
                    throw std::invalid_argument("distance batch coordinates must be finite");
                }
            }
            const double distance = compute(pi, pj, dim);
            if (!std::isfinite(distance))
            {
                throw std::overflow_error("distance batch computation produced a non-finite value");
            }
            computed.push_back(distance);
        }
        results = std::move(computed);
    }

private:
    DistFn dist_fn_ = nullptr;
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
        {
            return 1.0;
        }
        const double measured_speedup = baseline_ms / accelerated_ms;
        return std::isfinite(measured_speedup) && measured_speedup >= 0.0 ? measured_speedup : 1.0;
    }

    static DistanceBenchmark run(const double *points, nerve::Size n_points, nerve::Size dim,
                                 nerve::Size n_iterations = 10)
    {
        if (n_iterations == 0)
        {
            throw std::invalid_argument("distance benchmark iterations must be positive");
        }
        if (points == nullptr && n_points > 1)
        {
            throw std::invalid_argument("distance benchmark points pointer is null");
        }

        DistanceComputer simd_comp;

        // Benchmark scalar version
        auto start = std::chrono::high_resolution_clock::now();
        for (nerve::Size iter = 0; iter < n_iterations; ++iter)
        {
            for (nerve::Size i = 0; i < n_points; ++i)
            {
                for (nerve::Size j = i + 1; j < n_points; ++j)
                {
                    euclideanScalar(points + i * dim, points + j * dim, dim);
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
