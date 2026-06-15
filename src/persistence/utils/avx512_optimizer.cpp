// AVX-512 Optimizer for Persistent Homology

#include "nerve/persistence/utils/avx512_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace nerve::persistence::avx512
{

namespace
{

constexpr double AVX512_FULL_SPEEDUP = 8.0;
constexpr double AVX512_PARTIAL_SPEEDUP = 4.0;
constexpr double AVX512_NO_SPEEDUP = 1.0;
constexpr size_t AVX512_LARGE_MATRIX_THRESHOLD = 10000;
constexpr size_t AVX512_STREAMING_BYTES_THRESHOLD = 4ULL * 1024ULL * 1024ULL;
constexpr double AVX512_CACHE_FIT_SPEEDUP = 1.35;
constexpr double AVX512_CACHE_LIMITED_SPEEDUP = 1.15;
constexpr double AVX512_NON_TEMPORAL_SPEEDUP = 1.10;
constexpr int AVX512_ALIGNMENT_BYTES = 64;
constexpr int AVX512_MIN_WORDS = 8;

[[maybe_unused]] double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

bool checkAVX512F()
{
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f") != 0;
#else
    return false;
#endif
}

bool checkAVX512VL()
{
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512vl") != 0;
#else
    return false;
#endif
}

bool checkAVX512BW()
{
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512bw") != 0;
#else
    return false;
#endif
}

bool checkAVX512DQ()
{
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512dq") != 0;
#else
    return false;
#endif
}

[[maybe_unused]] bool checkAVX2()
{
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

bool compiledWithAVX512Kernels()
{
#ifdef __AVX512F__
    return true;
#else
    return false;
#endif
}

[[maybe_unused]] bool canRunAVX512Kernel(size_t num_words)
{
    return compiledWithAVX512Kernels() && num_words >= AVX512_MIN_WORDS &&
           detectAVX512Features().has_avx512f;
}

void addBitColumnsScalar(uint64_t *dest, const uint64_t *src, size_t num_words)
{
    if (!dest || !src || num_words == 0)
    {
        return;
    }

    for (size_t i = 0; i < num_words; ++i)
    {
        dest[i] ^= src[i];
    }
}

} // namespace

AVX512Features detectAVX512Features()
{
    AVX512Features features;

    features.has_avx512f = checkAVX512F();
    features.has_avx512vl = checkAVX512VL();
    features.has_avx512bw = checkAVX512BW();
    features.has_avx512dq = checkAVX512DQ();

    features.has_full_avx512 = features.has_avx512f && features.has_avx512vl &&
                               features.has_avx512bw && features.has_avx512dq;

    return features;
}

void addBitColumnsOptimized(uint64_t *dest, const uint64_t *src, size_t num_words)
{
#ifdef __AVX512F__
    if (canRunAVX512Kernel(num_words))
    {
        addBitColumnsAVX512(dest, src, num_words);
        return;
    }
#endif
    addBitColumnsScalar(dest, src, num_words);
}

AVX512Benchmark benchmarkAVX512(size_t num_words, int iterations)
{
    AVX512Benchmark bench{};
    if (num_words == 0 || iterations <= 0)
    {
        return bench;
    }

    std::vector<uint64_t> scalar_a(num_words, 0xAAAAAAAAAAAAAAAAULL);
    std::vector<uint64_t> scalar_b(num_words, 0x5555555555555555ULL);

    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (size_t i = 0; i < num_words; ++i)
        {
            scalar_a[i] ^= scalar_b[i];
        }
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    bench.scalar_time_ms =
        std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();

    volatile uint64_t scalar_checksum =
        std::accumulate(scalar_a.begin(), scalar_a.end(), uint64_t{0},
                        [](uint64_t lhs, uint64_t rhs) { return lhs ^ rhs; });
    (void)scalar_checksum;

    AVX512Features features = detectAVX512Features();
    if (compiledWithAVX512Kernels() && features.has_full_avx512)
    {
        bench.theoretical_speedup = AVX512_FULL_SPEEDUP;
    }
    else if (compiledWithAVX512Kernels() && features.has_avx512f)
    {
        bench.theoretical_speedup = AVX512_PARTIAL_SPEEDUP;
    }

#ifdef __AVX512F__
    if (canRunAVX512Kernel(num_words))
    {
        std::vector<uint64_t> avx512_a(num_words, 0xAAAAAAAAAAAAAAAAULL);
        std::vector<uint64_t> avx512_b(num_words, 0x5555555555555555ULL);

        auto start_avx512 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; ++iter)
        {
            addBitColumnsAVX512(avx512_a.data(), avx512_b.data(), num_words);
        }
        auto end_avx512 = std::chrono::high_resolution_clock::now();
        bench.avx512_time_ms =
            std::chrono::duration<double, std::milli>(end_avx512 - start_avx512).count();

        volatile uint64_t avx512_checksum =
            std::accumulate(avx512_a.begin(), avx512_a.end(), uint64_t{0},
                            [](uint64_t lhs, uint64_t rhs) { return lhs ^ rhs; });
        (void)avx512_checksum;

        bench.speedup = finiteBenchmarkSpeedup(bench.scalar_time_ms, bench.avx512_time_ms);
    }
#endif

    return bench;
}

VectorizationMode getOptimalVectorizationMode()
{
#ifdef __AVX512F__
    AVX512Features features = detectAVX512Features();
    if (features.has_full_avx512)
    {
        return VectorizationMode::AVX512_FULL;
    }
    else if (features.has_avx512f)
    {
        return VectorizationMode::AVX512_PARTIAL;
    }
#endif
#ifdef __AVX2__
    if (checkAVX2())
    {
        return VectorizationMode::AVX2;
    }
#endif
    return VectorizationMode::SCALAR;
}

AVX512Config getOptimalAVX512Config(size_t num_columns, int num_rows)
{
    AVX512Config config{};
    config.alignment = AVX512_ALIGNMENT_BYTES;
    config.min_words_for_avx512 = AVX512_MIN_WORDS;

    AVX512Features features = detectAVX512Features();
    const size_t row_words = num_rows > 0 ? (static_cast<size_t>(num_rows) + 63) / 64 : 0;

    config.use_avx512 = compiledWithAVX512Kernels() && features.has_avx512f &&
                        row_words >= config.min_words_for_avx512;

    const long double workload_bytes = static_cast<long double>(num_columns) *
                                       static_cast<long double>(row_words) *
                                       static_cast<long double>(sizeof(uint64_t));
    config.use_non_temporal =
        config.use_avx512 &&
        workload_bytes >= static_cast<long double>(AVX512_STREAMING_BYTES_THRESHOLD);

    return config;
}

AVX512SpeedupEstimate estimateAVX512Speedup(size_t num_words)
{
    AVX512SpeedupEstimate estimate{};
    AVX512Features features = detectAVX512Features();

    if (!compiledWithAVX512Kernels() || !features.has_avx512f || num_words < AVX512_MIN_WORDS)
    {
        return estimate;
    }

    estimate.base_speedup = features.has_full_avx512 ? AVX512_FULL_SPEEDUP : AVX512_PARTIAL_SPEEDUP;

    if (num_words > AVX512_LARGE_MATRIX_THRESHOLD)
    {
        estimate.memory_bandwidth_speedup = AVX512_CACHE_LIMITED_SPEEDUP;
        estimate.non_temporal_speedup = AVX512_NON_TEMPORAL_SPEEDUP;
    }
    else
    {
        estimate.memory_bandwidth_speedup = AVX512_CACHE_FIT_SPEEDUP;
        estimate.non_temporal_speedup = AVX512_NO_SPEEDUP;
    }

    estimate.total_speedup =
        estimate.base_speedup * estimate.memory_bandwidth_speedup * estimate.non_temporal_speedup;

    estimate.total_speedup = std::min(estimate.total_speedup, estimate.base_speedup);

    return estimate;
}

} // namespace nerve::persistence::avx512
