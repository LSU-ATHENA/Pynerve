#include "nerve/compression/simd_compression.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef NERVE_HAS_AVX512
#include <immintrin.h>
#endif

namespace nerve::compression
{

void simdQuantize(const double *input, std::size_t n, int bits, std::uint8_t *output)
{
    if (input == nullptr || output == nullptr || n == 0)
        return;
    if (bits < 1 || bits > 8)
        return;

    const double scale = static_cast<double>((1 << bits) - 1);
    const double inv_scale = 1.0 / scale;

#ifdef NERVE_HAS_AVX512
    const std::size_t vec_width = 8;
    const std::size_t vec_count = n / vec_width;
    const std::size_t remainder = n % vec_width;
    const __m512d scale_vec = _mm512_set1_pd(scale);
    const __m512d zero = _mm512_setzero_pd();

    for (std::size_t i = 0; i < vec_count; ++i)
    {
        const std::size_t offset = i * vec_width;
        __m512d vals = _mm512_loadu_pd(input + offset);
        __m512d clipped = _mm512_max_pd(_mm512_min_pd(vals, _mm512_set1_pd(1.0)), zero);
        __m512d scaled = _mm512_mul_pd(clipped, scale_vec);
        __m512d rounded = _mm512_roundscale_pd(scaled, _MM_FROUND_TO_NEAREST_INT);
        __m512i int_vals = _mm512_cvtpd_epi32(rounded);
        alignas(64) int raw_ints[16];
        _mm512_store_si512(raw_ints, int_vals);
        for (std::size_t j = 0; j < vec_width; ++j)
        {
            output[offset + j] = static_cast<std::uint8_t>(raw_ints[j]);
        }
    }
    for (std::size_t i = n - remainder; i < n; ++i)
    {
        const double clipped = std::max(0.0, std::min(input[i], 1.0));
        output[i] = static_cast<std::uint8_t>(clipped * scale + 0.5);
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        const double clipped = std::max(0.0, std::min(input[i], 1.0));
        output[i] = static_cast<std::uint8_t>(clipped * scale + 0.5);
    }
#endif
}

void simdDequantize(const std::uint8_t *encoded, std::size_t n, int bits, double *output)
{
    if (encoded == nullptr || output == nullptr || n == 0)
        return;
    if (bits < 1 || bits > 8)
        return;

    const double scale = static_cast<double>((1 << bits) - 1);
    const double inv_scale = 1.0 / scale;

#ifdef NERVE_HAS_AVX512
    const std::size_t vec_width = 8;
    const std::size_t vec_count = n / vec_width;
    const std::size_t remainder = n % vec_width;
    const __m512d inv_scale_vec = _mm512_set1_pd(inv_scale);

    for (std::size_t i = 0; i < vec_count; ++i)
    {
        const std::size_t offset = i * vec_width;
        __m256i vals_8bit = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(encoded + offset));
        __m256i vals_32bit = _mm256_cvtepu8_epi32(_mm_set1_epi64x(0));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vals_32bit), vals_8bit);
        __m512i int_vals = _mm512_cvtepi32_epi64(
            _mm256_cvtepu8_epi32(_mm256_castsi128_si256(_mm256_castsi256_si128(vals_8bit))));
        __m512d vals = _mm512_cvtepi64_pd(int_vals);
        __m512d dequant = _mm512_mul_pd(vals, inv_scale_vec);
        _mm512_storeu_pd(output + offset, dequant);
    }
    for (std::size_t i = n - remainder; i < n; ++i)
    {
        output[i] = static_cast<double>(encoded[i]) * inv_scale;
    }
#else
    for (std::size_t i = 0; i < n; ++i)
    {
        output[i] = static_cast<double>(encoded[i]) * inv_scale;
    }
#endif
}

} // namespace nerve::compression
