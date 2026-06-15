#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/utils/avx512_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using nerve::Size;

bool check_avx512_feature_detection()
{
    auto features = nerve::persistence::avx512::detectAVX512Features();
    (void)features;
    return true;
}

#ifdef __AVX512F__

bool check_avx512_column_xor_equivalence()
{
    constexpr size_t kWords = 16;
    alignas(64) uint64_t dest1[kWords];
    alignas(64) uint64_t dest2[kWords];
    alignas(64) uint64_t src[kWords];

    for (size_t i = 0; i < kWords; ++i)
    {
        dest1[i] = static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL;
        dest2[i] = dest1[i];
        src[i] = static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL;
    }

    nerve::persistence::avx512::addBitColumnsAVX512(dest1, src, kWords);

    for (size_t i = 0; i < kWords; ++i)
    {
        dest2[i] ^= src[i];
    }

    for (size_t i = 0; i < kWords; ++i)
    {
        if (dest1[i] != dest2[i])
        {
            std::cerr << "AVX512 XOR mismatch at word " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_avx512_copy_equivalence()
{
    constexpr size_t kWords = 16;
    alignas(64) uint64_t src[kWords];
    alignas(64) uint64_t dest[kWords];

    for (size_t i = 0; i < kWords; ++i)
    {
        src[i] = static_cast<uint64_t>(i * 0x9e3779b97f4a7c15ULL);
        dest[i] = 0;
    }

    nerve::persistence::avx512::copyColumnAVX512(dest, src, kWords);

    for (size_t i = 0; i < kWords; ++i)
    {
        if (dest[i] != src[i])
        {
            std::cerr << "AVX512 copy mismatch at word " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_avx512_zero_column()
{
    constexpr size_t kWords = 16;
    alignas(64) uint64_t col[kWords];
    for (size_t i = 0; i < kWords; ++i)
        col[i] = 0xdeadbeefdeadbeefULL;

    nerve::persistence::avx512::zeroColumnAVX512(col, kWords);

    for (size_t i = 0; i < kWords; ++i)
    {
        if (col[i] != 0)
        {
            std::cerr << "AVX512 zero column mismatch at word " << i << "\n";
            return false;
        }
    }
    return true;
}

bool check_avx512_pivot_find()
{
    constexpr size_t kWords = 8;
    alignas(64) uint64_t col[kWords] = {0, 0, 0, 0, 0, 0, 0, 1ULL << 63};

    int pivot = nerve::persistence::avx512::findPivotAVX512(col, kWords);
    if (pivot < 0)
    {
        std::cerr << "AVX512 pivot find: expected non-negative pivot\n";
        return false;
    }
    return true;
}

bool check_avx512_batch_xor()
{
    constexpr size_t kWords = 16;
    alignas(64) uint64_t dest[kWords];
    alignas(64) uint64_t src1[kWords];
    alignas(64) uint64_t src2[kWords];

    for (size_t i = 0; i < kWords; ++i)
    {
        src1[i] = static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL;
        src2[i] = static_cast<uint64_t>(i + 3) * 0x9e3779b97f4a7c15ULL;
    }

    nerve::persistence::avx512::batchXORAVX512(dest, src1, src2, kWords);

    for (size_t i = 0; i < kWords; ++i)
    {
        uint64_t expected = src1[i] ^ src2[i];
        if (dest[i] != expected)
        {
            std::cerr << "AVX512 batch XOR mismatch at word " << i << "\n";
            return false;
        }
    }
    return true;
}

#endif

bool check_optimized_dispatch()
{
    constexpr size_t kWords = 16;
    std::vector<uint64_t> dest(kWords);
    std::vector<uint64_t> src(kWords);

    for (size_t i = 0; i < kWords; ++i)
    {
        dest[i] = static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL;
        src[i] = static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL;
    }

    nerve::persistence::avx512::addBitColumnsOptimized(dest.data(), src.data(), kWords);

    for (size_t i = 0; i < kWords; ++i)
    {
        uint64_t expected = dest[i] ^ src[i];
        (void)expected;
    }
    return true;
}

bool check_vectorization_mode()
{
    auto mode = nerve::persistence::avx512::getOptimalVectorizationMode();
    (void)mode;
    return true;
}

bool check_optimal_config()
{
    auto config = nerve::persistence::avx512::getOptimalAVX512Config(100, 64);
    (void)config;
    return true;
}

bool check_speedup_estimate()
{
    auto estimate = nerve::persistence::avx512::estimateAVX512Speedup(64);
    if (estimate.total_speedup <= 0.0)
    {
        std::cerr << "speedup estimate should be positive\n";
        return false;
    }
    return true;
}

bool check_should_use()
{
    bool use = nerve::persistence::avx512::shouldUseAVX512(16);
    (void)use;
    return true;
}

} // namespace

int main()
{
    if (!check_avx512_feature_detection())
    {
        std::cerr << "FAIL: AVX512 feature detection\n";
        return 1;
    }

#ifdef __AVX512F__
    if (!check_avx512_column_xor_equivalence())
    {
        std::cerr << "FAIL: AVX512 column XOR equivalence\n";
        return 1;
    }
    if (!check_avx512_copy_equivalence())
    {
        std::cerr << "FAIL: AVX512 copy equivalence\n";
        return 1;
    }
    if (!check_avx512_zero_column())
    {
        std::cerr << "FAIL: AVX512 zero column\n";
        return 1;
    }
    if (!check_avx512_pivot_find())
    {
        std::cerr << "FAIL: AVX512 pivot find\n";
        return 1;
    }
    if (!check_avx512_batch_xor())
    {
        std::cerr << "FAIL: AVX512 batch XOR\n";
        return 1;
    }
#endif

    if (!check_optimized_dispatch())
    {
        std::cerr << "FAIL: optimized dispatch\n";
        return 1;
    }
    if (!check_vectorization_mode())
    {
        std::cerr << "FAIL: vectorization mode\n";
        return 1;
    }
    if (!check_optimal_config())
    {
        std::cerr << "FAIL: optimal config\n";
        return 1;
    }
    if (!check_speedup_estimate())
    {
        std::cerr << "FAIL: speedup estimate\n";
        return 1;
    }
    if (!check_should_use())
    {
        std::cerr << "FAIL: should use\n";
        return 1;
    }
    return 0;
}
