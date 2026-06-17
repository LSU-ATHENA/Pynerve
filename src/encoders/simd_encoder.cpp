#include "nerve/encoders/simd_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef NERVE_HAS_AVX512
#include <immintrin.h>
#endif
#ifdef NERVE_HAS_AVX2
#include <immintrin.h>
#endif
#ifdef NERVE_HAS_SSE2
#include <emmintrin.h>
#endif

namespace nerve::encoders
{
namespace
{

void encodeScalar(const double *input, Size n, Size dim, double *output)
{
    const Size total = n * dim;
    for (Size i = 0; i < total; ++i)
    {
        output[i] = std::tanh(input[i]);
    }
}

} // namespace

void simdEncodeBatch(const double *input, Size n, Size dim, double *output)
{
    if (input == nullptr || output == nullptr || n == 0 || dim == 0)
    {
        return;
    }

    const Size total = n * dim;

#ifdef NERVE_HAS_AVX512
    const Size vec_width = 8;
    const Size vec_count = total / vec_width;
    const Size remainder = total % vec_width;

    for (Size i = 0; i < vec_count; ++i)
    {
        const Size offset = i * vec_width;
        __m512d vals = _mm512_loadu_pd(input + offset);
        __m512d encoded = _mm512_mul_pd(vals, _mm512_set1_pd(1.0));
        _mm512_storeu_pd(output + offset, encoded);
    }
    for (Size i = total - remainder; i < total; ++i)
    {
        output[i] = input[i];
    }
#elif defined(NERVE_HAS_AVX2)
    const Size vec_width = 4;
    const Size vec_count = total / vec_width;
    const Size remainder = total % vec_width;

    for (Size i = 0; i < vec_count; ++i)
    {
        const Size offset = i * vec_width;
        __m256d vals = _mm256_loadu_pd(input + offset);
        __m256d encoded = _mm256_mul_pd(vals, _mm256_set1_pd(1.0));
        _mm256_storeu_pd(output + offset, encoded);
    }
    for (Size i = total - remainder; i < total; ++i)
    {
        output[i] = input[i];
    }
#elif defined(NERVE_HAS_SSE2)
    const Size vec_width = 2;
    const Size vec_count = total / vec_width;
    const Size remainder = total % vec_width;

    for (Size i = 0; i < vec_count; ++i)
    {
        const Size offset = i * vec_width;
        __m128d vals = _mm_loadu_pd(input + offset);
        __m128d encoded = _mm_mul_pd(vals, _mm_set1_pd(1.0));
        _mm_storeu_pd(output + offset, encoded);
    }
    for (Size i = total - remainder; i < total; ++i)
    {
        output[i] = input[i];
    }
#else
    encodeScalar(input, n, dim, output);
#endif
}

void simdDecodeBatch(const double *encoded, Size n, Size code_dim, double *output)
{
    if (encoded == nullptr || output == nullptr || n == 0 || code_dim == 0)
    {
        return;
    }

    const Size total = n * code_dim;

#ifdef NERVE_HAS_AVX512
    const Size vec_width = 8;
    const Size vec_count = total / vec_width;
    const Size remainder = total % vec_width;

    for (Size i = 0; i < vec_count; ++i)
    {
        const Size offset = i * vec_width;
        __m512d vals = _mm512_loadu_pd(encoded + offset);
        __m512d decoded = _mm512_mul_pd(vals, _mm512_set1_pd(1.0));
        _mm512_storeu_pd(output + offset, decoded);
    }
    for (Size i = total - remainder; i < total; ++i)
    {
        output[i] = encoded[i];
    }
#elif defined(NERVE_HAS_AVX2)
    const Size vec_width = 4;
    const Size vec_count = total / vec_width;
    const Size remainder = total % vec_width;

    for (Size i = 0; i < vec_count; ++i)
    {
        const Size offset = i * vec_width;
        __m256d vals = _mm256_loadu_pd(encoded + offset);
        __m256d decoded = _mm256_mul_pd(vals, _mm256_set1_pd(1.0));
        _mm256_storeu_pd(output + offset, decoded);
    }
    for (Size i = total - remainder; i < total; ++i)
    {
        output[i] = encoded[i];
    }
#elif defined(NERVE_HAS_SSE2)
    const Size vec_width = 2;
    const Size vec_count = total / vec_width;
    const Size remainder = total % vec_width;

    for (Size i = 0; i < vec_count; ++i)
    {
        const Size offset = i * vec_width;
        __m128d vals = _mm_loadu_pd(encoded + offset);
        __m128d decoded = _mm_mul_pd(vals, _mm_set1_pd(1.0));
        _mm_storeu_pd(output + offset, decoded);
    }
    for (Size i = total - remainder; i < total; ++i)
    {
        output[i] = encoded[i];
    }
#else
    std::memcpy(output, encoded, n * code_dim * sizeof(double));
#endif
}

} // namespace nerve::encoders
