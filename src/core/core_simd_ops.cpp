#include "nerve/config.hpp"
#include "nerve/cpu/simd.hpp"
#include "nerve/cpu/x86_intrinsics.hpp"

#include <cmath>
#include <cstring>

namespace nerve::core
{

#if defined(NERVE_HAS_X86_INTRINSICS) && defined(__AVX2__)
static bool hasAvx2()
{
    static const bool has = cpu::simd::CPUFeatureDetector::hasAVX2();
    return has;
}
#endif

#if defined(NERVE_HAS_X86_INTRINSICS) && defined(__AVX512F__)
static bool hasAvx512()
{
    static const bool has = cpu::simd::CPUFeatureDetector::hasAVX512F();
    return has;
}
#endif

void simdMemcpy(void *dst, const void *src, Size bytes)
{
    Size i = 0;
    auto *d = static_cast<uint8_t *>(dst);
    const auto *s = static_cast<const uint8_t *>(src);
#if defined(__AVX512F__)
    if (hasAvx512())
    {
        for (; i + 64 <= bytes; i += 64)
        {
            _mm512_storeu_si512(d + i, _mm512_loadu_si512(s + i));
        }
    }
#endif
#if defined(__AVX2__)
    if (i + 32 <= bytes)
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + i),
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i)));
        i += 32;
    }
#endif
    for (; i < bytes; ++i)
        d[i] = s[i];
}

void simdMemset(void *dst, int value, Size bytes)
{
    Size i = 0;
    auto *d = static_cast<uint8_t *>(dst);
#if defined(__AVX512F__)
    if (hasAvx512())
    {
        __m512i v = _mm512_set1_epi8(static_cast<char>(value));
        for (; i + 64 <= bytes; i += 64)
            _mm512_storeu_si512(d + i, v);
    }
#endif
#if defined(__AVX2__)
    if (i + 32 <= bytes)
    {
        __m256i v = _mm256_set1_epi8(static_cast<char>(value));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + i), v);
        i += 32;
    }
#endif
    for (; i < bytes; ++i)
        d[i] = static_cast<uint8_t>(value);
}

void simdMemcpyAligned(void *dst, const void *src, Size bytes)
{
    if (reinterpret_cast<uintptr_t>(dst) % 64 == 0 && reinterpret_cast<uintptr_t>(src) % 64 == 0)
    {
        Size i = 0;
        auto *d = static_cast<uint8_t *>(dst);
        const auto *s = static_cast<const uint8_t *>(src);
#if defined(__AVX512F__)
        if (hasAvx512())
        {
            _mm_prefetch(s + 64, _MM_HINT_T0);
            for (; i + 64 <= bytes; i += 64)
            {
                _mm_prefetch(s + i + 128, _MM_HINT_T0);
                __m512i v = _mm512_load_si512(reinterpret_cast<const __m512i *>(s + i));
                _mm512_stream_si512(reinterpret_cast<__m512i *>(d + i), v);
            }
        }
#endif
        for (; i < bytes; ++i)
            d[i] = s[i];
    }
    else
    {
        simdMemcpy(dst, src, bytes);
    }
}

double simdReduceSum(const double *data, Size n)
{
    double sum = 0.0;
    Size i = 0;
#if defined(__AVX512F__)
    if (hasAvx512())
    {
        __m512d acc = _mm512_setzero_pd();
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(data + i);
            acc = _mm512_add_pd(acc, v);
        }
        sum = _mm512_reduce_add_pd(acc);
    }
#endif
#if defined(__AVX2__)
    if (i + 4 <= n)
    {
        __m256d acc = _mm256_loadu_pd(data + i);
        i += 4;
        for (; i + 4 <= n; i += 4)
        {
            __m256d v = _mm256_loadu_pd(data + i);
            acc = _mm256_add_pd(acc, v);
        }
        double tmp[4];
        _mm256_storeu_pd(tmp, acc);
        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
#endif
    for (; i < n; ++i)
        sum += data[i];
    return sum;
}

} // namespace nerve::core
