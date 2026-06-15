#include "nerve/persistence/utils/avx512_optimizer.hpp"

#ifdef __AVX512F__
#include "nerve/cpu/x86_intrinsics.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace nerve
{
namespace persistence
{
namespace avx512
{
namespace
{

constexpr int AVX512_WORDS_PER_ITERATION = 8;
constexpr int AVX512_ALIGNMENT_BYTES = 64;
constexpr int AVX512_MIN_WORDS = 8;

bool canRunAVX512Kernel(size_t num_words)
{
    return num_words >= AVX512_MIN_WORDS && detectAVX512Features().has_avx512f;
}

bool isAligned64(const void *ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) % AVX512_ALIGNMENT_BYTES) == 0;
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

void addBitColumnsAVX512(uint64_t *__restrict__ dest, const uint64_t *__restrict__ src,
                         size_t num_words)
{
    if (!dest || !src || num_words == 0)
    {
        return;
    }
    if (!canRunAVX512Kernel(num_words))
    {
        addBitColumnsScalar(dest, src, num_words);
        return;
    }

    size_t i = 0;
    for (; i + AVX512_WORDS_PER_ITERATION <= num_words; i += AVX512_WORDS_PER_ITERATION)
    {
        __m512i vdest = _mm512_loadu_si512(reinterpret_cast<const void *>(&dest[i]));
        __m512i vsrc = _mm512_loadu_si512(reinterpret_cast<const void *>(&src[i]));
        __m512i vresult = _mm512_xor_si512(vdest, vsrc);
        _mm512_storeu_si512(reinterpret_cast<void *>(&dest[i]), vresult);
    }
    for (; i < num_words; ++i)
    {
        dest[i] ^= src[i];
    }
}

void addBitColumnsAVX512Streaming(uint64_t *__restrict__ dest, const uint64_t *__restrict__ src,
                                  size_t num_words)
{
    if (!dest || !src || num_words == 0)
    {
        return;
    }
    if (!canRunAVX512Kernel(num_words))
    {
        addBitColumnsScalar(dest, src, num_words);
        return;
    }
    if (!isAligned64(dest))
    {
        addBitColumnsAVX512(dest, src, num_words);
        return;
    }

    size_t i = 0;
    for (; i + AVX512_WORDS_PER_ITERATION <= num_words; i += AVX512_WORDS_PER_ITERATION)
    {
        __m512i vdest = _mm512_loadu_si512(reinterpret_cast<const void *>(&dest[i]));
        __m512i vsrc = _mm512_loadu_si512(reinterpret_cast<const void *>(&src[i]));
        __m512i vresult = _mm512_xor_si512(vdest, vsrc);
        _mm512_stream_si512(reinterpret_cast<__m512i *>(&dest[i]), vresult);
    }
    for (; i < num_words; ++i)
    {
        dest[i] ^= src[i];
    }
    _mm_mfence();
}

void batchXORAVX512(uint64_t *__restrict__ dest, const uint64_t *__restrict__ src1,
                    const uint64_t *__restrict__ src2, size_t num_words)
{
    if (!dest || !src1 || !src2 || num_words == 0)
    {
        return;
    }
    if (!canRunAVX512Kernel(num_words))
    {
        for (size_t i = 0; i < num_words; ++i)
        {
            dest[i] = src1[i] ^ src2[i];
        }
        return;
    }

    size_t i = 0;
    for (; i + AVX512_WORDS_PER_ITERATION <= num_words; i += AVX512_WORDS_PER_ITERATION)
    {
        __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void *>(&src1[i]));
        __m512i v2 = _mm512_loadu_si512(reinterpret_cast<const void *>(&src2[i]));
        __m512i vresult = _mm512_xor_si512(v1, v2);
        _mm512_storeu_si512(reinterpret_cast<void *>(&dest[i]), vresult);
    }
    for (; i < num_words; ++i)
    {
        dest[i] = src1[i] ^ src2[i];
    }
}

int findPivotAVX512(const uint64_t *words, size_t num_words)
{
    if (!words || num_words == 0)
    {
        return -1;
    }
    if (!canRunAVX512Kernel(num_words))
    {
        for (int word_idx = static_cast<int>(num_words) - 1; word_idx >= 0; --word_idx)
        {
            if (words[word_idx] != 0)
            {
                uint64_t word = words[word_idx];
                int bit_idx = 63 - __builtin_clzll(word);
                return word_idx * 64 + bit_idx;
            }
        }
        return -1;
    }

    size_t end = num_words;
    while (end >= AVX512_WORDS_PER_ITERATION)
    {
        const size_t base = end - AVX512_WORDS_PER_ITERATION;
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void *>(&words[base]));
        __mmask8 nonzero_mask = _mm512_test_epi64_mask(v, v);
        if (nonzero_mask != 0)
        {
            int lane = 31 - __builtin_clz(static_cast<unsigned>(nonzero_mask));
            uint64_t word = words[base + static_cast<size_t>(lane)];
            int bit_idx = 63 - __builtin_clzll(word);
            return static_cast<int>((base + static_cast<size_t>(lane)) * 64 + bit_idx);
        }
        end = base;
    }

    for (int word_idx = static_cast<int>(end) - 1; word_idx >= 0; --word_idx)
    {
        if (words[word_idx] != 0)
        {
            uint64_t word = words[word_idx];
            int bit_idx = 63 - __builtin_clzll(word);
            return word_idx * 64 + bit_idx;
        }
    }
    return -1;
}

void zeroColumnAVX512(uint64_t *dest, size_t num_words)
{
    if (!dest || num_words == 0)
    {
        return;
    }
    if (!canRunAVX512Kernel(num_words))
    {
        std::fill(dest, dest + num_words, uint64_t{0});
        return;
    }

    size_t i = 0;
    __m512i zero = _mm512_setzero_si512();
    for (; i + AVX512_WORDS_PER_ITERATION <= num_words; i += AVX512_WORDS_PER_ITERATION)
    {
        _mm512_storeu_si512(reinterpret_cast<void *>(&dest[i]), zero);
    }
    for (; i < num_words; ++i)
    {
        dest[i] = 0;
    }
}

void copyColumnAVX512(uint64_t *__restrict__ dest, const uint64_t *__restrict__ src,
                      size_t num_words)
{
    if (!dest || !src || num_words == 0)
    {
        return;
    }
    if (!canRunAVX512Kernel(num_words))
    {
        std::memcpy(dest, src, num_words * sizeof(uint64_t));
        return;
    }

    size_t i = 0;
    for (; i + AVX512_WORDS_PER_ITERATION <= num_words; i += AVX512_WORDS_PER_ITERATION)
    {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void *>(&src[i]));
        _mm512_storeu_si512(reinterpret_cast<void *>(&dest[i]), v);
    }
    for (; i < num_words; ++i)
    {
        dest[i] = src[i];
    }
}

} // namespace avx512
} // namespace persistence
} // namespace nerve
#endif // __AVX512F__
