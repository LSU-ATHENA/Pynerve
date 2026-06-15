#include "nerve/encoders/simd_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef NERVE_HAS_AVX512
#include <immintrin.h>
#endif

namespace nerve::encoders
{

void simdEncodeBatch(const double *input, Size n, Size dim, double *output)
{
    if (input == nullptr || output == nullptr || n == 0 || dim == 0)
        return;

#ifdef NERVE_HAS_AVX512
    const Size vec_width = 8;
    const Size vec_count = n * dim / vec_width;
    const Size remainder = (n * dim) % vec_width;

    for (Size i = 0; i < vec_count; ++i)
    {
        const Size offset = i * vec_width;
        __m512d vals = _mm512_loadu_pd(input + offset);
        __m512d encoded = _mm512_mul_pd(vals, _mm512_set1_pd(1.0));
        _mm512_storeu_pd(output + offset, encoded);
    }
    for (Size i = n * dim - remainder; i < n * dim; ++i)
    {
        output[i] = input[i];
    }
#else
    std::memcpy(output, input, n * dim * sizeof(double));
#endif
}

void simdDecodeBatch(const double *encoded, Size n, Size code_dim, double *output)
{
    if (encoded == nullptr || output == nullptr || n == 0 || code_dim == 0)
        return;

#ifdef NERVE_HAS_AVX512
    const Size vec_width = 8;
    const Size vec_count = n * code_dim / vec_width;
    const Size remainder = (n * code_dim) % vec_width;

    for (Size i = 0; i < vec_count; ++i)
    {
        const Size offset = i * vec_width;
        __m512d vals = _mm512_loadu_pd(encoded + offset);
        __m512d decoded = _mm512_mul_pd(vals, _mm512_set1_pd(1.0));
        _mm512_storeu_pd(output + offset, decoded);
    }
    for (Size i = n * code_dim - remainder; i < n * code_dim; ++i)
    {
        output[i] = encoded[i];
    }
#else
    std::memcpy(output, encoded, n * code_dim * sizeof(double));
#endif
}

} // namespace nerve::encoders
